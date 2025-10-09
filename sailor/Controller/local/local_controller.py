import argparse
import time
from concurrent import futures
import grpc
import json
import os
import subprocess

from sailor.Controller.controller import Controller
from sailor.protos import orchestration_pb2, orchestration_pb2_grpc
from sailor.protos.orchestration_pb2_grpc import (
    WorkerAgentStub, MasterControllerStub
)
from sailor.Worker.elastic_worker_agent import TRAINING_START_PORT
from sailor.Planner.sailor_planner.cpp_src.planner import SailorPlanner
from sailor.Planner.simulations.utils import parse_trace

# if run on clariden, do export no_proxy=node_list

def convert_trace(input_trace_file):
    with open(input_trace_file, 'r') as f:
        input_trace = json.load(f)

    output_trace = []
    for input in input_trace:
        duration = input["duration_sec"]
        nodes = input["nodes"]
        trace_dict = convert_node_list(nodes)
        output = [duration, trace_dict]
        output_trace.append(output)
    return output_trace

def convert_node_list(node_list):
    trace_dict = {}
    for node in node_list:
        gpu_zone = f"{node[2]}_{node[1]}"
        num_gpus = node[3]
        if gpu_zone not in trace_dict:
            trace_dict[gpu_zone] = 0
        trace_dict[gpu_zone] += num_gpus
    return trace_dict

class LocalController(Controller):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.num_restarts = 0

    def get_available_nodes(self, all_nodes):
        node_list = []
        for node in all_nodes:
            endpoint = node[0]
            print(f"************ check endpoint {endpoint}")
            try:
                request = orchestration_pb2.CheckHealthRequest()
                with grpc.insecure_channel(endpoint) as channel:
                    stub = WorkerAgentStub(channel)
                    response = stub.CheckHealth(request)
                    print(response)
                    if not response.processes_ok:
                        continue
                node_list.append(node)
            except Exception as e:
                print(f"Host {endpoint} is not available")

        return node_list

    def update_nodes(self, cluster_config, training_config, node_list):
        if len(node_list) == 0:
            return

        num_stages=cluster_config['pipeline_parallelism']
        all_tp_degrees=cluster_config['tp_degrees']
        dp = cluster_config['data_parallelism']
        node_rank = 0
        ga_steps = training_config['global_batch_size'] // (dp * cluster_config['microbatch_size'])
        node_list_idx = 0
        num_layers_per_stage = cluster_config['num_layers_per_stage']

        # find globals
        world_size = 0
        max_tensor_parallelism = 1
        configs_all_stages = []

        for stage_id in range(num_stages):
            stage_config = []
            for replica_id in range(dp):
                max_tensor_parallelism = max(max_tensor_parallelism, len(all_tp_degrees[stage_id][replica_id]))
                stage_config.append(orchestration_pb2.ReplicaConfig(replica_ranks=all_tp_degrees[stage_id][replica_id]))
                world_size += len(all_tp_degrees[stage_id][replica_id])

            sconf = orchestration_pb2.StageConfig(stage_replicas=stage_config)
            configs_all_stages.append(sconf)

        # send to nodes
        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            for stage_id in range(num_stages):
                for replica_id in range(dp):
                    tmp_degrees = all_tp_degrees[stage_id][replica_id]
                    node = node_list[node_list_idx]
                    # SIMPLIFIED VERSION - no fault tolerance
                    tp_node = len(tmp_degrees)
                    worker_configuration = orchestration_pb2.WorkerConfiguration(
                        ranks=tmp_degrees,
                        world_size=world_size,
                        master_ip=node_list[0][0].split(":")[0],
                        master_port=str(int(TRAINING_START_PORT)+self.num_restarts),
                        pipeline_parallelism=num_stages,
                        tensor_parallelism=tp_node,
                        data_parallelism=dp,
                        max_tensor_parallelism=max_tensor_parallelism,
                        all_stages=configs_all_stages,
                        layers_per_stage=num_layers_per_stage
                    )

                    hparams = orchestration_pb2.HyperParams()
                    hparams.global_batch_size = training_config['global_batch_size']
                    hparams.micro_batch_size = cluster_config['microbatch_size'] # TODO: take this out of the hyperparams
                    hparams.num_stages = num_stages # not used in megatron
                    hparams.ga_steps = ga_steps

                    print(stage_id, worker_configuration, hparams)

                    host, port = node[0].split(":")
                    executor.submit(self.notify_topology_change,
                                   node, worker_configuration, hparams, host, port)
                    print(f"Sent training info to node: {node}")
                    node_rank += tp_node
                    node_list_idx += 1

        self.num_restarts += 1


    def check_ready(self, node):
        request = orchestration_pb2.CheckReadyRequest(is_ready=1)
        with grpc.insecure_channel(node) as channel:
            stub = WorkerAgentStub(channel)
            stub.CheckReady(request)


    def check_if_ready(self, node):
        try:
            self.check_ready(node)
        except Exception:
            #print(f"Worker {node} is not ready")
            return False
        return True


    def wait_all_ready(self, node_list):
        for node in node_list:
            print(f"Check if node {node} is ready")
            while (not self.check_if_ready(node[0])):
                pass


    def terminate_existing(self, prev_node_list, next_node_list, failed_nodes_list=[]):
        prev_nodes = [node[0] for node in prev_node_list]
        next_nodes = [node[0] for node in next_node_list]
        failed_nodes = [node[0] for node in failed_nodes_list]
        print(f"prev_nodes is {prev_nodes}, next_nodes is {next_nodes}")
        for node in prev_nodes:
            if node not in failed_nodes:
                print(f"Sending kill request to node {node}")
                hostname, port = node.split(":")
                self.send_kill_request(hostname, port, (node not in next_nodes))


def get_plan(planner, input_planner_config, training_config):

    sorted_plans = planner.get_sorted_plans(
        cluster_config=input_planner_config,
        training_config=training_config
    )

    if len(sorted_plans) == 0:
        print("Not valid plan found! Sorry")
        return {}

    plan = sorted_plans[0]['pipeline_list'][0]
    print(f"plan is {plan}")

    layers_per_stage = list(([list(x) for x in plan['layers_per_stage']]))
    print(layers_per_stage)

    num_layers_per_stage = [len(x) for x in layers_per_stage]
    print(num_layers_per_stage)

    tmp_per_stage = []
    rank = 0
    for stage_config in plan['tmp_per_stage']:
        stage_tmps = []
        for replica in stage_config:
            tp_replica = replica[1]
            tps_this_replica = list(range(rank, rank+tp_replica))
            stage_tmps.append(tps_this_replica)
            rank += tp_replica
        tmp_per_stage.append(stage_tmps)

    cluster_config = {}
    cluster_config['pipeline_parallelism'] = plan['num_stages']
    cluster_config['tp_degrees'] = tmp_per_stage
    cluster_config['data_parallelism'] = plan['dp'][0]
    cluster_config['microbatch_size'] = sorted_plans[0]['mbs']
    cluster_config['num_layers_per_stage'] = num_layers_per_stage
    return cluster_config

def main(args):
    with open(args.training_config_json, 'r') as f:
        training_config = json.load(f)

    os.environ['SAILOR_PATH'] = args.sailor_path
    controller = LocalController(args)
    prev_node_list = []
    planner = SailorPlanner(
            args.sailor_profile_file_dir,
            args.training_config_json,
            args.quotas_dict,
            args.objective,
            args.fp16
    )

    if args.trace_file: # trace-based
        with open(args.trace_file, 'r') as f:
            machine_info_trace = json.load(f)
        gpu_trace = convert_trace(args.trace_file)

        for idx, (duration, num_gpus_config) in enumerate(gpu_trace):
            input_planner_config = num_gpus_config
            input_planner_config["max_cost"] = args.max_cost
            input_planner_config["min_throughput"] = args.min_throughput
            print(duration, input_planner_config)

            plan_start = time.time()
            cluster_config = get_plan(planner, input_planner_config, training_config)
            print(cluster_config)
            node_list = machine_info_trace[idx]["nodes"]
            print(node_list)
            print(f"[RECONFIGURATION] Planner time is {time.time() - plan_start}")

            stop_start_time = time.time()
            # 2. notify that config will change
            if prev_node_list:
                controller.terminate_existing(prev_node_list, node_list)
            controller.wait_all_ready(node_list)
            print(f"[RECONFIGURATION] Stop and cleanup time is {time.time() - stop_start_time}")

            # 3. send updated config
            send_start_time = time.time()
            controller.update_nodes(cluster_config, training_config, node_list)
            print(f"[RECONFIGURATION] Worker Update time is {time.time() - send_start_time}")

            # # 4. sleep for now
            time.sleep(duration)
            prev_node_list = node_list
    elif args.initial_config_file:
        with open(args.initial_config_file, 'r') as f:
            machine_info_trace = json.load(f)

        while True:
            node_list = controller.get_available_nodes(machine_info_trace[0]["nodes"])
            failed_nodes = [node for node in prev_node_list if node not in node_list]
            print(node_list, failed_nodes)
            if node_list != prev_node_list:
                num_gpus_config = convert_node_list(node_list) # TODO
                print(f"num_gpus_config is {num_gpus_config}")
                input_planner_config = num_gpus_config
                input_planner_config["max_cost"] = args.max_cost
                input_planner_config["min_throughput"] = args.min_throughput

                cluster_config = get_plan(planner, input_planner_config, training_config)
                print(f"node_list is {node_list}")
                if prev_node_list:
                    controller.terminate_existing(prev_node_list, node_list, failed_nodes)

                print(f"cluster_config is {cluster_config}")
                controller.wait_all_ready(node_list)
                controller.update_nodes(cluster_config, training_config, node_list)
                prev_node_list = node_list
            time.sleep(10)
    else:
        print("Please provide a --trace_file or a --initial_config_file option")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trace controller')
    parser.add_argument('--training_config_json', type=str, required=True, help='Json file containing training info')
    parser.add_argument('--sailor_profile_file_dir', type=str, default='',
                        help='Directory for SAILOR profiling')
    parser.add_argument('--quotas_dict', type=str, default='',
                        help='Json file containing user quotas')
    parser.add_argument('--objective', type=str, default='throughput',
                        help='User objective (throughput, cost, or value (iters/USD))')
    parser.add_argument('--max_cost', type=float, default=0.0,
                        help='Max cost (USD/iteration)')
    parser.add_argument('--min_throughput', type=float, default=0.0,
                        help='Min througput (iters/sec)')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--sailor_path', type=str, required=True, help='Path to the sailor repo')
    parser.add_argument('--trace_file', type=str, required=False, help='GPU availability trace file. If specified, the controller will train using the availability from the file')
    parser.add_argument('--initial_config_file', type=str, required=False, help='Initial config. If specified, the controller will monitor the status of resources and scale up/down respectively')


    args = parser.parse_args()
    main(args)
