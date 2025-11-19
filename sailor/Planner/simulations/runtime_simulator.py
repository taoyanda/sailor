import json
import math
from os.path import expanduser

from sailor.Planner.simulations.constants import GPU_MEMORY_GB
from sailor.Planner.simulations.utils import models
from sailor.Planner.sailor_planner.constants import MEMORY_BUCKET_DEEPSPEED_SIZE
from sailor.profiling.profile_utils import estimate_send_time, estimate_ar_time, find_bw, Zone, GPU_Type
from sailor.Planner.sailor_planner.utils import partition_uniform


class VMconfig:
    def __init__(self, gpu_type, gpus_per_node, zone: Zone) -> None:
        self.gpu_type = gpu_type
        self.gpus_per_node = gpus_per_node
        self.zone = zone


class TMPconfig:
    def __init__(self, vm_list: list[VMconfig], tmp: int) -> None:
        self.vm_list = vm_list
        self.tmp = tmp

    def serialize(self):
        vm_list_serialized = [(vm.gpu_type, vm.gpus_per_node, vm.zone) for vm in self.vm_list]
        return [vm_list_serialized, self.tmp]


class InterNodeNetworkInfo:
    def __init__(self, path: str) -> None:
        self.network_coeffs = {}
        self._available_zones = []
        self._available_gpu_types = []
        with open(path, 'r') as f:
            self.network_coeffs = json.load(f)

        # The following chunk assumes that the data is complete (i.e. every zone has all Zones and GPU types)
        for zone in self.network_coeffs.keys():
            self._available_zones.append(zone)
            for gpu_type in self.network_coeffs[zone].keys():
                self._available_gpu_types.append(gpu_type)

    def get_zone_coeffs(self, sender_zone: Zone, sender_gpu_type: GPU_Type, sender_gpu_count: int, receiver_zone: Zone, receiver_gpu_type: GPU_Type, receiver_gpu_count: int):
        # Get the network bw coeffs between two zones
        #print(sender_zone, sender_gpu_type, sender_gpu_count, receiver_zone, receiver_gpu_type, receiver_gpu_count)
        return self.network_coeffs[sender_zone][sender_gpu_type][str(sender_gpu_count)][receiver_zone][receiver_gpu_type][str(receiver_gpu_count)]


class Pipeline:
    def __init__(self, num_stages: int, layers_per_stage: list[list[int]], num_dp: list[int], tmp_configs: list[list[TMPconfig]]):
        self.num_stages = num_stages
        self.layers_per_stage = layers_per_stage
        self.num_dp = num_dp
        self.tmp_configs = tmp_configs

    def print_info(self):
        print(f"Num stages: {self.num_stages}")
        print(f"Layers_per_stage: {self.layers_per_stage}")
        print(f"TMP configs: {self.tmp_configs}")
        print(f"DP: {self.num_dp}")


class Plan:
    def __init__(self, pipeline_list: list[Pipeline], mbs: int, homogeneous_pipelines: bool, dp_ubatch: int) -> None:
        self.pipeline_list = pipeline_list
        self.num_pipelines = len(pipeline_list)
        self.mbs = mbs  # Might need to make it per-stage, check
        self.homogeneous_pipelines = homogeneous_pipelines
        self.dp_ubatch = dp_ubatch

    def to_dict(self):
        plan_to_dict = {}
        plan_to_dict["mbs"] = self.mbs
        pipelines = []

        for pipeline in self.pipeline_list:
            tmp_configs_serialized = []
            for tmp_config_stages in pipeline.tmp_configs:
                tmp_configs_per_stage = []
                for tmp_config in tmp_config_stages:
                    tmp_config_serial = tmp_config.serialize()
                    tmp_configs_per_stage.append(tmp_config_serial)
                tmp_configs_serialized.append(tmp_configs_per_stage)

            pipeline_dict = {
                "num_stages": pipeline.num_stages,
                "layers_per_stage": list([list(x) for x in pipeline.layers_per_stage]),
                "dp_degrees": pipeline.num_dp,
                "tmp_stages": tmp_configs_serialized
            }
            pipelines.append(pipeline_dict)

        plan_to_dict['pipelines'] = pipelines

        return plan_to_dict


def convert_homogeneous(plan: dict, training_config: dict):
    if 'pipeline_list' in plan:
        pipeline_list = []
        for pipeline_def in plan['pipeline_list']:
            num_stages = pipeline_def['num_stages']
            gpu_tmp_tuple_list = pipeline_def['tmp_per_stage']

            tmp_config_list = []
            dp_list = pipeline_def['dp']
            dp_degree = dp_list[0]  # TODO

            for tmp_stage in gpu_tmp_tuple_list:
                tmp_config_per_stage = []
                for gpu_tmp in tmp_stage:
                    vm_config_list = []
                    for vm_list in gpu_tmp[0]:
                        vm_config_list.append(
                            VMconfig(gpu_type=vm_list[0], gpus_per_node=vm_list[1], zone=vm_list[2]))
                    tmp_config_per_stage.append(
                        TMPconfig(vm_config_list, gpu_tmp[1])
                    )
                tmp_config_list.append(tmp_config_per_stage)

            layers_per_stage = pipeline_def['layers_per_stage']

            pipeline = Pipeline(num_stages, layers_per_stage, dp_list, tmp_config_list)
            pipeline_list.append(pipeline)
        homogeneous_pipelines = False
    else:
        # Homogeneous Plans: Varuna, AMP, Galvatron
        # They give a D*P*T grid
        num_layers = training_config['num_all_layers']
        num_stages = plan['P']
        layers_per_stage = partition_uniform(num_layers, num_stages, verbose=True)
        dp_degree = plan['D']

        vm_config = VMconfig(gpu_type=plan['gpu_type'], gpus_per_node=plan['num_gpus_per_node'], zone=plan['zone'])
        gpu_tmp_tuple_list = [[TMPconfig([vm_config], plan['T'])
                               for _ in range(dp_degree)] for _ in range(num_stages)]
        dp_degree_list = [dp_degree for _ in range(num_stages)]

        pipeline_list = [Pipeline(num_stages, layers_per_stage, dp_degree_list, gpu_tmp_tuple_list)]
        homogeneous_pipelines = True

    return Plan(pipeline_list, plan['mbs'], homogeneous_pipelines, dp_degree)


class Simulator():
    def __init__(self, sailor_path: str, training_config: dict, llm_info: dict, fp16: bool, profiles: dict, zone: Zone) -> None:
        self.training_config = training_config
        self.optimizer = self.training_config["optimizer"]

        self.model = training_config['model']
        self.model_mem_info = llm_info[self.model]
        self.model_config = models[self.model]

        self.fp16 = fp16
        self.float_size = 2 if fp16 else 4
        self.global_batch_size = training_config['global_batch_size']

        self.num_layers = training_config['num_all_layers']
        self.profiles = profiles[self.model]

        if '1' in self.model_mem_info:
            self.activation_sizes_per_layer = [self.model_mem_info['1'][str(
                layer)]['act_output_floats']*self.float_size for layer in range(self.num_layers)]
            self.weight_sizes_per_layer = [self.model_mem_info['1'][str(
                layer)]['params_floats']*self.float_size for layer in range(self.num_layers)]
        elif '2' in self.model_mem_info:
            self.activation_sizes_per_layer = [self.model_mem_info['2'][str(
                layer)]['act_output_floats']*self.float_size for layer in range(self.num_layers)]
            self.weight_sizes_per_layer = [self.model_mem_info['2'][str(
                layer)]['params_floats']*self.float_size*2 for layer in range(self.num_layers)]
        elif '4' in self.model_mem_info:
            self.activation_sizes_per_layer = [self.model_mem_info['4'][str(
                layer)]['act_output_floats']*self.float_size for layer in range(self.num_layers)]
            self.weight_sizes_per_layer = [self.model_mem_info['4'][str(
                layer)]['params_floats']*self.float_size*4 for layer in range(self.num_layers)]

        # inter-node
        self.inter_network_coeffs = InterNodeNetworkInfo(
            f'{sailor_path}/sailor/sailor/providers/multizone_bandwidths_het.json')

        with open(f'{sailor_path}/sailor/sailor/providers/intra_node_bandwidths.json', 'r') as f:
            intra_network_coeffs_dict = json.load(f)
        self.intra_network_coeffs = {}
        for gpu_type, coeffs_per_gpu in intra_network_coeffs_dict.items():
            for num_gpus, coeffs in coeffs_per_gpu.items():
                self.intra_network_coeffs[(gpu_type, int(num_gpus))] = coeffs
        with open(f'{sailor_path}/sailor/sailor/providers/gcp/communication_cost.json', 'r') as f:
            self.communication_cost = json.load(f)


    def get_memory_footprint_on_gpu(self, plan_dict: dict, stage_id: int, tmp: int):

        # computes memory footprint on a single GPU
        plan = convert_homogeneous(plan_dict, self.training_config)
        pipeline = plan.pipeline_list[0]

        mbs = plan.mbs

        # extra memory for kernel loading
        megatron_mem = 4.76 * 1e9 + 300*1e6 # the first term is for fused kernel loading, depends on GPU + platform
        if self.optimizer == 'sgd':
            # sgd saves only a copy of model parameters in fp32
            memory_multiplier_optim = 4*1  # bytes
        else:
            # this works for fp16
            memory_multiplier_optim = 4*2  # bytes - only 2 keys in state dict
        model_copy = 4  # keep model in fp32
        additional_ds_copies = 4  # Deepspeed creates 2 additional copies of the model (start of the training)
        gradients = 4
        comm = 4

        model_multiplier = memory_multiplier_optim + model_copy + gradients + comm + additional_ds_copies

        stage = pipeline.layers_per_stage[stage_id]

        # 1. Compute mem needed for parameters
        num_params = 0
        for layer in stage:
            num_params += self.model_mem_info[str(tmp)][str(layer)]['params_floats']
        mf = num_params * model_multiplier

        # 2. Compute mem needed for activations
        af_stage = 0
        for layer in stage:
            af_stage_layer = self.model_mem_info[str(tmp)][str(layer)]['act_mem_floats']
            af_stage += af_stage_layer

        af_stage = af_stage * mbs * self.float_size
        af_factor = (pipeline.num_stages - stage_id )

        reserved_mem = mf + af_stage * af_factor
        mem_used = reserved_mem + megatron_mem

        print(f"MODEL MULTIPLIER {model_multiplier}, NUM_PARAMS: {num_params}, MF: {mf}, AF_STAGE: {af_stage}, RESERVED: {reserved_mem}, TOTAL: {mem_used}")

        return mem_used

    def check_config_fits(self, plan_dict: dict):
        plan = convert_homogeneous(plan_dict, self.training_config)

        mbs = plan.mbs
        if self.optimizer == 'sgd':
            # sgd saves only a copy of model parameters in fp32
            memory_multiplier_optim = 4*1  # bytes
        else:
            # this works for fp16
            memory_multiplier_optim = 4*2  # bytes - only 2 keys in state dict
        model_copy = 4  # keep model in fp32
        additional_ds_copies = 4  # Deepspeed creates 2 additional copies of the model (start of the training)
        gradients = 4
        comm = 4

        if self.fp16:
            # model_multiplier = 16
            memory_multiplier_optim = 2*2  # bytes
            model_copy = 2 # model in fp16
            additional_ds_copies = 2  # Deepspeed creates 2 additional copies of the model (start of the training)
            gradients = 2 # gradients in fp16
            comm = 2 # communicatiion 
            
            
        model_multiplier = memory_multiplier_optim + model_copy + gradients + comm + additional_ds_copies
        all_fit = True

        # For each pipeline, make sure it fits in memory:
        for pipeline in plan.pipeline_list:
            for i, stage in enumerate(pipeline.layers_per_stage):

                # 1. Compute mem needed for parameters

                tmp_configs = pipeline.tmp_configs[i]
                for tmp_config in tmp_configs:

                    gpu_type = tmp_config.vm_list[0].gpu_type  # TODO: fix for Metis
                    tmp = tmp_config.tmp
                    megatron_mem = 2.0 * 1e9 if tmp==1 else 3.0 * 1e9  # extra mem needed by megatron, shown in nvidia-smi, but not as part of torch mem

                    if str(tmp) not in self.model_mem_info:
                        return False

                    num_params = 0
                    for layer in stage:
                        num_params += self.model_mem_info[str(tmp)][str(layer)]['params_floats']
                    mf = num_params * model_multiplier

                    # 2. Compute mem needed for activations
                    af_stage = 0
                    for layer in stage:
                        af_stage_layer = self.model_mem_info[str(tmp)][str(layer)]['act_mem_floats']
                        af_stage += af_stage_layer

                    af_stage = af_stage * mbs * self.float_size
                    mem_used = mf + af_stage * (pipeline.num_stages - i) + megatron_mem

                    gpu_mem = GPU_MEMORY_GB[gpu_type] * 1024 * 1024 * 1024

                    #print(i, mem_used, gpu_mem)
                    all_fit &= (mem_used <= gpu_mem)
                    if plan.homogeneous_pipelines:
                        break

        # print(f"all fit is {all_fit}")
        return all_fit

    def get_comm_cost_for_pipeline(self, pipeline: Pipeline, mbs: int, num_micro_batches=1):
        # fwd pass
        fwd_cost = 0
        for i in range(len(pipeline.layers_per_stage) - 1):
            stage = pipeline.layers_per_stage[i]
            activation_size = self.activation_sizes_per_layer[stage[-1]] * mbs

            for j in range(pipeline.num_dp[i]):
                # TODO: FIX case with different dp
                dp_prev = pipeline.num_dp[i+1]
                vmconfig_send = pipeline.tmp_configs[i][j].vm_list[0] # ignore metis case
                vmconfig_recv = pipeline.tmp_configs[i+1][j%dp_prev].vm_list[0] # ignore metis case
                fwd_cost += self.communication_cost[vmconfig_send.zone][vmconfig_recv.zone] * activation_size

        # backprop
        bp_cost = 0
        for i in range(1, len(pipeline.layers_per_stage)):
            stage = pipeline.layers_per_stage[i-1]
            activation_size = self.activation_sizes_per_layer[stage[-1]] * mbs
            for j in range(pipeline.num_dp[i]):
                # TODO: FIX case with different dp
                dp_prev = pipeline.num_dp[i-1]
                vmconfig_send = pipeline.tmp_configs[i][j].vm_list[0] # ignore metis case
                vmconfig_recv = pipeline.tmp_configs[i-1][j%dp_prev].vm_list[0] # ignore metis case
                bp_cost += self.communication_cost[vmconfig_send.zone][vmconfig_recv.zone] * activation_size

        # allreduce cost
        ar_cost = 0
        for i, stage in enumerate(pipeline.layers_per_stage):
            stage_size = sum([self.weight_sizes_per_layer[layer] for layer in stage])
            dp = pipeline.num_dp[i]
            for j in range(dp):
                vmconfig_send = pipeline.tmp_configs[i][j].vm_list[0]
                vmconfig_recv = pipeline.tmp_configs[i][(j + 1) % dp].vm_list[0]
                ar_cost += self.communication_cost[vmconfig_send.zone][vmconfig_recv.zone] * stage_size/dp * 2 * (dp-1)

        fwd_cost *= num_micro_batches
        bp_cost *= num_micro_batches

        print("COSTS: ", fwd_cost/1e9, bp_cost/1e9, ar_cost/1e9)
        return (fwd_cost + bp_cost + ar_cost) / 1e9


    def get_time_per_pipeline(
        self,
        pipeline: Pipeline,
        mbs: int,
        fwd_times_per_gpu_tmp: dict,
        bwd_times_per_gpu_tmp: dict,
        update_times_per_gpu_tmp: dict,
        activation_sizes_per_layer: list,
        num_micro_batches: int,
        async_pipe: bool = False
    ):

        comp_time_per_stage = []
        update_time_per_stage = []

        for i, stage in enumerate(pipeline.layers_per_stage):
            tmp_configs = pipeline.tmp_configs[i]
            update = 0.0
            fwd_bwd = 0.0

            # compute per stage
            for config in tmp_configs:
                fwd_bwd_config = 0.0
                for layer in stage:
                    fwd = get_time_layer(fwd_times_per_gpu_tmp, config, layer,
                                                     mbs, self.training_config, self.float_size, self.inter_network_coeffs, self.intra_network_coeffs)
                    bwd = get_time_layer(bwd_times_per_gpu_tmp, config, layer,
                                                     mbs, self.training_config, self.float_size, self.inter_network_coeffs, self.intra_network_coeffs)
                    fwd_bwd_config += (fwd + bwd)
                    #print(fwd+bwd, fwd_bwd_config)
                fwd_bwd = max(fwd_bwd, fwd_bwd_config)

                # print(stage)
                # print(update_times_per_gpu_tmp[config.vm_list[0].gpu_type][config.tmp])
                update_config = update_times_per_gpu_tmp[config.vm_list[0].gpu_type][config.tmp][stage[0]]
                update = max(update, update_config)

            update_time_per_stage.append(update)
            comp_time_per_stage.append(fwd_bwd)

        #comp_time_per_stage[1] += 1
        print(f"Pipeline stages is {pipeline.layers_per_stage}")
        print(f"comp_time_per_stage is {comp_time_per_stage}")

        update_time = max(update_time_per_stage)
        tot_computation_time = sum(comp_time_per_stage)

        per_stage_comm_times = estimate_p2p_pipeline_times(
            activation_sizes_per_layer, pipeline, mbs, self.inter_network_coeffs, self.intra_network_coeffs)
        print(f"per_stage_comm_times is {per_stage_comm_times}")
        tot_communication_time = sum(per_stage_comm_times)

        straggler_per_stage = [x+y for x, y in zip(comp_time_per_stage, per_stage_comm_times)]
        straggler = max(straggler_per_stage)
        straggler_overhead = (num_micro_batches - 1) * straggler

        print(f"num_micro_batches {num_micro_batches}, straggler: {straggler}, straggler_overhead: {straggler_overhead}, tot_communication_time: {tot_communication_time}, tot_computation_time: {tot_computation_time}, update_time: {update_time}")

        # total pipeline time
        t_pp = straggler_overhead + tot_communication_time + tot_computation_time + update_time
        return t_pp

    def simulate_iteration_time(self, plan_dict: dict):

        plan = convert_homogeneous(plan_dict, self.training_config)
        mbs = plan.mbs
        num_micro_batches = math.ceil(self.global_batch_size / (mbs * plan.dp_ubatch * plan.num_pipelines))

        # # TODO: what about this one?
        # if 'dg_per_stage' in plan_dict:
        #     dg_per_stage = plan['dg_per_stage']
        # else:
        #     dg_per_stage = [[] for _ in range(plan.num_stages)]

        fwd_times = {}
        bwd_times = {}
        update_times = {}

        # build times per layer for that specific mbs
        for gpu_type in self.profiles.keys():
            prof_gpu = self.profiles[gpu_type]

            if str(mbs) not in prof_gpu:
                continue

            fwd_times[gpu_type] = {}
            bwd_times[gpu_type] = {}
            update_times[gpu_type] = {}

            for tmp in prof_gpu["1"].keys():
                tmp_int = int(tmp)
                fwd_times[gpu_type][tmp_int] = []
                bwd_times[gpu_type][tmp_int] = []
                update_times[gpu_type][tmp_int] = []
                if tmp in prof_gpu[str(mbs)]:
                    for _, value in prof_gpu[str(mbs)][tmp].items():
                        fwd_times[gpu_type][tmp_int].append(value[0])
                        bwd_times[gpu_type][tmp_int].append(value[1])
                        update_times[gpu_type][tmp_int].append(value[2])
                else:
                    fwd_times[gpu_type][tmp_int] = [0.0 for _ in range(self.num_layers)]
                    bwd_times[gpu_type][tmp_int] = [0.0 for _ in range(self.num_layers)]
                    update_times[gpu_type][tmp_int] = [0.0 for _ in range(self.num_layers)]

        # get time for each pipeline
        pipeline_times = []
        pipeline_cost = 0
        for pipeline in plan.pipeline_list:
            pipeline_time = self.get_time_per_pipeline(
                pipeline, mbs, fwd_times, bwd_times, update_times, self.activation_sizes_per_layer, num_micro_batches)
            pipeline_times.append(pipeline_time)
            pipeline_cost += self.get_comm_cost_for_pipeline(pipeline, mbs, num_micro_batches)

        t_pp = max(pipeline_times)
        t_sync = self.estimate_sync_time(plan)

        iteration_time = t_pp + t_sync
        iteration_cost = pipeline_cost

        print(f"*********** T_pp is {t_pp}, T_sync is {t_sync}, Iteration time is {iteration_time}, Iteration cost is {iteration_cost}")

        reformed_plan_dict = plan.to_dict()

        return iteration_time, iteration_cost, reformed_plan_dict

    def estimate_sync_time(self, plan: Plan):

        pipeline = plan.pipeline_list[0]
        t_sync = 0.0

        # 1. time for each pipeline
        t_sync_intra_pipeline = []
        for pipeline in plan.pipeline_list:
            t_sync_pipeline = 0.0
            for i, stage in enumerate(pipeline.layers_per_stage):
                t_sync_stage = 0.0
                #print(f"Estimate sync time for stage {i}")
                tp_min = 4
                for tp_config in pipeline.tmp_configs[i]:
                    tp_min = min(tp_min, tp_config.tmp)
                stage_size = sum([self.weight_sizes_per_layer[layer]/tp_min for layer in stage])
                for j in range(pipeline.num_dp[i] - 1):
                    # get time assuming node j is the bottleneck
                    sender_gpu_node = pipeline.tmp_configs[i][j].vm_list[0]
                    sender_gpu_type = sender_gpu_node.gpu_type
                    sender_gpu_count = 1 #sender_gpu_node.gpus_per_node
                    sender_zone = sender_gpu_node.zone

                    recver_gpu_node = pipeline.tmp_configs[i][j+1].vm_list[0]
                    recver_gpu_type = recver_gpu_node.gpu_type
                    recver_gpu_count = 1 #recver_gpu_node.gpus_per_node
                    receiver_zone = recver_gpu_node.zone

                    #print(sender_gpu_type, recver_gpu_type)
                    network_coef = self.inter_network_coeffs.get_zone_coeffs(
                        sender_zone=sender_zone, receiver_zone=receiver_zone, sender_gpu_type=sender_gpu_type, receiver_gpu_type=recver_gpu_type, sender_gpu_count=sender_gpu_count, receiver_gpu_count=recver_gpu_count)[0]

                    t_sync_stage = get_ar_time_with_buckets(stage_size, pipeline.num_dp[i], network_coef)
                    t_sync_pipeline = max(t_sync_pipeline, t_sync_stage)
            t_sync_intra_pipeline.append(t_sync_pipeline)

        t_sync_intra_max = max(t_sync_intra_pipeline)
        t_sync_inter = 0.0

        # 2. time across pipelines (used for Oobleck and FlashFlex)

        # # find the pipeline with the least stages
        max_num_stages, pipeline_id = 0, 0
        for i, pipeline in enumerate(plan.pipeline_list):
            if pipeline.num_stages > max_num_stages:
                max_num_stages = pipeline.num_stages
                pipeline_id = i

        max_pipeline = plan.pipeline_list[pipeline_id]
        for i, stage in enumerate(max_pipeline.layers_per_stage):
            t_sync_stage = 0.0
            stage_size = sum([self.weight_sizes_per_layer[layer] for layer in stage])

            #print(i, pipeline_id, max_num_stages, max_pipeline.tmp_configs)
            for tmp_config in max_pipeline.tmp_configs[i]:
                gpu_node = tmp_config.vm_list[0] # TODO: Adjust estimation
                gpu_type = gpu_node.gpu_type
                gpu_count = 1 #gpu_node.gpus_per_node
                sender_zone = gpu_node.zone
                receiver_zone = gpu_node.zone

                network_coef = self.inter_network_coeffs.get_zone_coeffs(sender_zone=sender_zone, receiver_zone=receiver_zone,
                                                                sender_gpu_type=gpu_type, receiver_gpu_type=gpu_type, sender_gpu_count=gpu_count, receiver_gpu_count=gpu_count)[0]
                t_sync_stage = max(t_sync_stage, get_ar_time_with_buckets(stage_size, plan.num_pipelines, network_coef))
            t_sync_inter = max(t_sync_inter, t_sync_stage)


        t_sync = t_sync_intra_max + t_sync_inter

        return t_sync


def get_comm_time_megatron_ar(layer, mbs, training_config, float_size, tmp, vm_list, inter_network_coeffs, intra_network_coeffs):
    # used for metis
    # called when there is across-node communication
    sequence_length = training_config["sequence_length"]
    hidden_size = training_config["hidden_size"]
    num_layers = training_config["num_layers"]
    ar_size = mbs * sequence_length * hidden_size * float_size

    min_bandwidth = 100*1e9
    min_network_coef = None

    for vm in vm_list:
        gpu_type = vm.gpu_type
        gpu_count = vm.gpus_per_node
        zone = vm.zone # for metis, zone info should be the same!

        network_coef = inter_network_coeffs.get_zone_coeffs(sender_zone=zone, receiver_zone=zone,
                            sender_gpu_type=gpu_type, receiver_gpu_type=gpu_type, sender_gpu_count=gpu_count, receiver_gpu_count=gpu_count)[0]

        bandwidth = find_bw(ar_size, tmp, network_coef)
        if (bandwidth < min_bandwidth):
            min_bandwidth = bandwidth
            min_network_coef = network_coef

    if layer == 0:
        # EMBEDDING: 1 AR AT FWD, 1 AR AT BWD
        ar_time = estimate_ar_time(ar_size, tmp, min_network_coef)
    elif layer < num_layers-1:
        # TRANSFORMER: 2 AR AT FWD, 2 AR AT BWD
        ar_time = 2 * estimate_ar_time(ar_size, tmp, min_network_coef)
    else:
        ar_time = 0.0

    return ar_time


def get_time_layer(times, tmp_config, layer, mbs, training_config, float_size, inter_network_coeffs, intra_network_coeffs):
    if len(tmp_config.vm_list) == 1:
        gpu_type = tmp_config.vm_list[0].gpu_type
        tmp = tmp_config.tmp
        return times[gpu_type][tmp][layer]
    else:
        # metis case
        tmp = tmp_config.tmp
        layer_time = 0.0
        # SIMPLIFICATION: just add time to coordinate
        max_comp_time = 0
        for vm in tmp_config.vm_list:
            gpu_type = vm.gpu_type
            num_gpus = vm.gpus_per_node
            comp_time = times[gpu_type][num_gpus][layer] / tmp  # simplification
            max_comp_time = max(max_comp_time, comp_time)
        comm_time = get_comm_time_megatron_ar(layer, mbs, training_config, float_size, tmp,
                                              tmp_config.vm_list, inter_network_coeffs, intra_network_coeffs)
        layer_time += comm_time

        return layer_time


def get_ar_time_with_buckets(model_size, D, netw_coef_d):

    dp_time = 0
    remaining = model_size
    # print(f"model_size is {model_size}, remaining is {remaining}")
    while remaining > 0:
        bucket_size = MEMORY_BUCKET_DEEPSPEED_SIZE if remaining > MEMORY_BUCKET_DEEPSPEED_SIZE else remaining
        dp_time_bucket = estimate_ar_time(bucket_size, D, netw_coef_d)
        #print(f"bucket_size is {bucket_size}, estimate_ar_time is {dp_time_bucket}")
        dp_time += dp_time_bucket
        remaining -= bucket_size

    return dp_time


def is_homogeneous(x):
    return len(x) == len(set(x))


def estimate_p2p_pipeline_times(activation_sizes_per_layer, pipeline, mbs, inter_network_coeffs, intra_network_coeffs):
    sending_times = []
    num_stages = len(pipeline.layers_per_stage)
    for i, stage in enumerate(pipeline.layers_per_stage):
        max_send_time_act = 0
        max_send_time_grad = 0
        #print(f"activation_sizes_per_layer is {activation_sizes_per_layer}")

        # 1. TODO: should it be per-tp? where is ovelap happening?
        activation_size = activation_sizes_per_layer[stage[-1]] * mbs
        gradient_size = activation_sizes_per_layer[stage[0]] * mbs
        #print(f"stage {i}, activation_size {activation_size}, gradient size is {gradient_size}")
        if i < pipeline.num_stages-1:
            for j in range(pipeline.num_dp[i]):
                # stage i sends activation to stage i + 1 in PP
                sender_gpu_node = pipeline.tmp_configs[i][j].vm_list[0]
                sender_gpu_type = sender_gpu_node.gpu_type
                sender_gpu_count = min(pipeline.tmp_configs[i][j].tmp, sender_gpu_node.gpus_per_node)
                sender_zone = sender_gpu_node.zone

                # TODO: FIX case with different dp
                dp_next = pipeline.num_dp[i+1]
                recver_gpu_node = pipeline.tmp_configs[i + 1][j % dp_next].vm_list[0]
                recver_gpu_type = recver_gpu_node.gpu_type
                recver_gpu_count = min(pipeline.tmp_configs[i + 1][j % dp_next].tmp, recver_gpu_node.gpus_per_node)
                receiver_zone = recver_gpu_node.zone

                network_coef = inter_network_coeffs.get_zone_coeffs(
                    sender_zone=sender_zone, receiver_zone=receiver_zone, sender_gpu_type=sender_gpu_type,
                    receiver_gpu_type=recver_gpu_type, sender_gpu_count=sender_gpu_count,
                    receiver_gpu_count=recver_gpu_count)[0]
                #print(f"ACTS: {sender_gpu_type}, {recver_gpu_type}, network_coef is {network_coef}")
                send_time = estimate_send_time(activation_size, network_coef)
                max_send_time_act = max(send_time, max_send_time_act)
        if i > 0:
            for j in range(pipeline.num_dp[i]):
                # stage i sends gradient to stage i - 1 in PP
                sender_gpu_node = pipeline.tmp_configs[i][j].vm_list[0]
                sender_gpu_type = sender_gpu_node.gpu_type
                sender_gpu_count =  min(pipeline.tmp_configs[i][j].tmp, sender_gpu_node.gpus_per_node)
                sender_zone = sender_gpu_node.zone

                # TODO: FIX case with different dp
                dp_prev = pipeline.num_dp[i-1]
                recver_gpu_node = pipeline.tmp_configs[i - 1][j % dp_prev].vm_list[0]
                recver_gpu_type = recver_gpu_node.gpu_type
                recver_gpu_count = min(pipeline.tmp_configs[i - 1][j % dp_prev].tmp, recver_gpu_node.gpus_per_node)
                receiver_zone = recver_gpu_node.zone

                network_coef = inter_network_coeffs.get_zone_coeffs(
                    sender_zone=sender_zone, receiver_zone=receiver_zone, sender_gpu_type=sender_gpu_type,
                    receiver_gpu_type=recver_gpu_type, sender_gpu_count=sender_gpu_count,
                    receiver_gpu_count=recver_gpu_count)[0]
                #print(f"GRADS: {sender_gpu_type}, {recver_gpu_type}, network_coef is {network_coef}")
                send_time = estimate_send_time(gradient_size, network_coef)
                max_send_time_grad = max(send_time, max_send_time_grad)

        #total_times = max_send_time_act + max_send_time_grad
        #print(f"stage {i}, activation_size {activation_size}, gradient size is {gradient_size}, max_send_time_act is {max_send_time_act}, max_send_time_grad is {max_send_time_grad}")
        if i>0 and i<num_stages-1:
            total_times = max_send_time_act + max_send_time_grad
        elif i==num_stages-1:
            total_times = 2*max_send_time_grad
        elif i==0:
            if num_stages<=3:
                total_times = max_send_time_act
            else:
                total_times = 2*max_send_time_act
        sending_times.append(total_times)
    return sending_times