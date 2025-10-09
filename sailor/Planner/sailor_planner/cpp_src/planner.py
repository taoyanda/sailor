
import os
import json
import time
from typing import Dict
import subprocess

from sailor.profiling.profile_utils import Zone, GPU_Type


class SailorPlanner():
    def __init__(
        self,
        profile_file,
        training_config_path,
        quotas_path_dict,
        objective,
        fp16,
        heterogeneous=True,
    ) -> None:

        home_dir = os.environ.get('SAILOR_PATH')
        arch = subprocess.check_output(['uname', '-m']).decode('utf-8')[:-1]
        lib_path = f"{home_dir}/sailor/sailor/Planner/sailor_planner/cpp_src"

        os.system(f"cd {lib_path} && make clean && make libplanner.so")
        import sailor.Planner.sailor_planner.cpp_src.libplanner as libplanner

        network_coeff_path = f"{home_dir}/sailor/sailor/providers/multizone_bandwidths_het.json"
        model_mem_info = f"{home_dir}/sailor/sailor/Planner/llm_info.json"
        communication_cost_file = f"{home_dir}/sailor/sailor/providers/gcp/communication_cost.json"

        print(f"From Python, heterogeneous is {heterogeneous}, objective is {objective}")
        assert objective in ["throughput", "iteration_cost"]

        with open(quotas_path_dict, 'r') as f:
            quotas = json.load(f)
            self.available_gpus = sorted(list(quotas.keys()))

        self.planner = libplanner.SailorPlanner(
            profile_file,
            network_coeff_path,
            training_config_path,
            model_mem_info,
            communication_cost_file,
            heterogeneous,
            fp16,
            quotas_path_dict,
            objective
        )

    def get_sorted_plans(self, training_config={}, cluster_config={}):
        start = time.time()
        max_cost = cluster_config.pop("max_cost")
        min_throughput = cluster_config.pop("min_throughput")

        # cluster config of format {'A100-40_us-central1-a': 16, 'A100-40_us-central1-b': 16, ...}
        # max_gpus_per_zone {'zone' : { 'gpu_type' : gpu_count}}
        max_gpus_per_zone: Dict[Zone, Dict[GPU_Type, int]] = {}
        for key in cluster_config.keys():
            gpu_type, zone = key.split('_', 1)
            if max_gpus_per_zone.get(zone) is None:
                max_gpus_per_zone[zone] = {}
            if max_gpus_per_zone[zone].get(gpu_type) is None:
                max_gpus_per_zone[zone][gpu_type] = 0
            max_gpus_per_zone[zone][gpu_type] = cluster_config[key]

        configs = self.planner.get_sorted_plans(max_gpus_per_zone, max_cost, min_throughput)
        print(f"It took {time.time()-start}")

        # convert results
        new_start = time.time()
        py_configs = []
        for config in configs:
            Tpp = config.get_Tpp()
            configs_per_stage = config.get_config_per_stage()
            zones = config.get_zones()

            # here, TP==num_gpus_per_node
            config_updated = []
            used_gpus = {}

            for stage_config in configs_per_stage:
                stage_config_updated = []
                for tmp_config in stage_config:
                    gpu_type = self.available_gpus[tmp_config[0]]
                    zone = zones[tmp_config[2]]
                    # if zone=="us-central1-b":
                    #     zone="us-west1-b"
                    new_config = ([(gpu_type, tmp_config[1], zone)], tmp_config[1])
                    if gpu_type not in used_gpus:
                        used_gpus[gpu_type] = 0
                    used_gpus[gpu_type] += tmp_config[1]
                    stage_config_updated.append(new_config)
                config_updated.append(stage_config_updated)

            # if not ("V100-16" in used_gpus):
            #     continue
            # if not ("A100-40" in used_gpus):
            #     continue
            # if used_gpus["A100-40"] != 16 and used_gpus["V100-16"] != 16:
            #     continue

            dp_per_stage = [len(stage) for stage in configs_per_stage]
            num_stages = len(list(config.get_stages()))

            layers = config.get_stages()

            pipeline_list = [{
                'num_stages': num_stages,
                'layers_per_stage': layers,  # config.get_stages(),
                'tmp_per_stage': config_updated,
                'dp': dp_per_stage
            }]
            py_config = {
                'pipeline_list': pipeline_list,
                'mbs': config.get_mbs(),
                'estimated_throughput': 1/Tpp if Tpp != 0 else 0.0,
                'iter_time': Tpp,
                'estimated_cost': [config.get_cost(), config.get_comm_cost(), config.get_comp_cost()],
                'used_gpus': used_gpus
            }
            py_configs.append(py_config)
        print(f"Extra time is {time.time()-new_start}")
        return py_configs
