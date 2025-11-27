# this is a simulator to help evaluate various planning policies under GPU availability traces

import argparse
import json
import time
import pandas as pd
import os
from pathlib import Path
import copy

from sailor.Planner.sailor_planner.cpp_src.planner import SailorPlanner
from sailor.Planner.baselines.AMP.amp_planner import AMPPlanner
from sailor.Planner.baselines.Piper.piper_planner import PiperPlanner
try:
    from sailor.Planner.baselines.Oobleck.oobleck_planner import OobleckPlanner
except Exception:
    pass
from sailor.Planner.baselines.Varuna.varuna_planner import VarunaPlanner
from sailor.Planner.baselines.Metis.metis_planner import MetisPlanner
from sailor.Planner.baselines.Galvatron.galvatron_planner import GalvatronPlanner
from sailor.Planner.baselines.Aceso.aceso_planner import AcesoPlanner
from sailor.Planner.baselines.Atlas.atlas_planner import AtlasPlanner
from sailor.Planner.baselines.DTFM.dtfm_planner import DTFMPlanner
from sailor.Planner.baselines.FlashFlex.flashflex_planner import FlashFlexPlanner

from sailor.Planner.simulations.runtime_simulator import (
    Simulator,
)
from sailor.Planner.simulations.runtime_simulator_op import (
    SimulatorOP,
)
from sailor.Planner.simulations.utils import parse_trace
from sailor.Planner.simulations.constants import GPU_PRICES


def simulate_plan(simulator, plan, zone):
    used_gpus = plan['used_gpus']
    iteration_time, comm_cost, reformed_plan_dict = simulator.simulate_iteration_time(plan)
    throughput = 1/iteration_time
    comp_cost_per_sec = 0
    for gpu_type in used_gpus:
        comp_cost_per_sec += used_gpus[gpu_type] * (GPU_PRICES[gpu_type][zone]/3600)
    return throughput, comp_cost_per_sec, comp_cost_per_sec/throughput, comm_cost, used_gpus, reformed_plan_dict


def evaluate_trace_with_mem(
    sailor_path,
    llm_info,
    planner,
    gpu_trace,
    basic_cluster_config,
    training_config,
    profiles,
    zone='us-central1-a',
    fp16=False,
    condense=False,
    op_simulator=False,
    objective="throughput",
    max_cost=0.0,
    max_cost_file=None,
    min_throughput=0.0,
    min_throughput_file=None
):
    
    print("-------------------- Start evaluation --------------------")
    print(f"Using FP16: {fp16}, condense: {condense}, op_simulator: {op_simulator}")

    sim_result_list = []
    if op_simulator:
        simulator = SimulatorOP(training_config, llm_info, fp16, profiles)
    else:
        simulator = Simulator(sailor_path, training_config, llm_info, fp16, profiles, zone=basic_cluster_config['zone'])

    trace_size = len(gpu_trace)
    if not max_cost_file:
        all_max_costs = [max_cost]*trace_size
    else:
        with open(max_cost_file, 'r') as f:
            all_max_costs = json.load(f)

    if not min_throughput_file:
        all_min_thr = [min_throughput]*trace_size
    else:
        with open(min_throughput_file, 'r') as f:
            all_min_thr = json.load(f)

    for idx, (duration, num_gpus_config) in enumerate(gpu_trace):

        sim_result = {}
        oom_plans_case = 0
        print(
            f"---------------------------------------- Evaluate baselines with {num_gpus_config}  --------------------------------")

        cur_max_cost = all_max_costs[idx]
        cur_min_thr = all_min_thr[idx]

        #num_gpus_config["A100-40_us-central1-a"] *= 2
        #num_gpus_config["V100-16_us-central1-a"] *= 3
        if isinstance(planner, SailorPlanner):
            test_cluster_config = num_gpus_config
            test_cluster_config["max_cost"] = cur_max_cost
            test_cluster_config["min_throughput"] = cur_min_thr
        elif isinstance(planner, DTFMPlanner):
            test_cluster_config = copy.deepcopy(num_gpus_config)
            gpu_types_set = set()
            for key, val in num_gpus_config.items():
                gpu_type = key.split("_")[0]
                gpu_types_set.add(gpu_type)
            gpu_types = list(gpu_types_set)
            test_cluster_config["gpu_types"] = gpu_types
            test_cluster_config["gpus_per_node"] = {}
            for gpu_type in gpu_types:
                test_cluster_config["gpus_per_node"][gpu_type] = basic_cluster_config[gpu_type]["gpus_per_node"]
        elif isinstance(planner, MetisPlanner) or isinstance(planner, AMPPlanner) or isinstance(planner, FlashFlexPlanner):
            het_cluster_config = {}
            for gpu_type_zone, num_gpus in num_gpus_config.items():
                gpu_type, zone = gpu_type_zone.split("_")
                het_cluster_config[gpu_type] = {
                    "num_nodes": num_gpus // basic_cluster_config[gpu_type]['gpus_per_node'],
                    "gpus_per_node": basic_cluster_config[gpu_type]['gpus_per_node'],
                    "mem_per_gpu": basic_cluster_config[gpu_type]['mem_per_gpu'], # TODO
                }
            het_cluster_config["zone"] = zone # assume single-zone
            test_cluster_config = het_cluster_config
        elif isinstance(planner, AtlasPlanner):
            # single GPU, multi zone
            atlas_cluster_config = {}
            trace_keys = list(num_gpus_config.keys())
            key = trace_keys[0]
            gpu_type, zone = key.split("_")
            for key, val in num_gpus_config.items():
                if gpu_type in key:
                   atlas_cluster_config[key] = val
            atlas_cluster_config["gpus_per_node"] = basic_cluster_config[gpu_type]['gpus_per_node']
            atlas_cluster_config["gpu_type"] = gpu_type

            test_cluster_config = atlas_cluster_config
        else:
            baseline_cluster_config = {}

            # this considers all GPUs as the ones of min memory
            if condense:
                num_gpus = 0
                mem = 256
                gpu_type = ""
                for gpu_name, gpu_count in num_gpus_config.items():
                    gpu_mem = int(gpu_name.split("-")[1])
                    if gpu_mem < mem:
                        gpu_type = gpu_name
                        mem = gpu_mem
                    num_gpus += gpu_count
            else:
                key = list(num_gpus_config.keys())[0]
                gpu_type, zone = key.split("_")
                print(gpu_type, zone)
                num_gpus_config = {gpu_type: num_gpus_config[key]}
                num_gpus = num_gpus_config[gpu_type]


            baseline_cluster_config['gpu_type'] = gpu_type
            baseline_cluster_config['num_nodes'] = num_gpus//basic_cluster_config[gpu_type]['gpus_per_node']
            baseline_cluster_config["gpus_per_node"] =  basic_cluster_config[gpu_type]['gpus_per_node']
            baseline_cluster_config["mem_per_gpu"] = basic_cluster_config[gpu_type]['mem_per_gpu']
            baseline_cluster_config["zone"] = zone
            test_cluster_config = baseline_cluster_config

        print(f"test_cluster_config is {test_cluster_config}")

        start = time.time()
        valid_plan = None
        sorted_plans = planner.get_sorted_plans(cluster_config=test_cluster_config, training_config=training_config)
        search_time = time.time()-start

        for plan in sorted_plans:
            print(f"CHECK PLAN {plan}")
            if not simulator.check_config_fits(plan):
                oom_plans_case += 1
                continue

            valid_plan = plan

            # Check if there exists a valid plan
            #if oom_plans_case != len(sorted_plans):
            throughput, comp_cost, comp_cost_iter, comm_cost, used_gpus_plan, reformed_plan_dict = simulate_plan(
                simulator,
                valid_plan,
                zone
            )

            print(f"Throughput is {throughput}, cur_min_thr is {cur_min_thr}")

            # check cost and throughput objectives
            cost_passed = ((objective=="throughput") and (cur_max_cost==0.0 or (comp_cost_iter + comm_cost <= cur_max_cost)))
            thr_passed = ((objective=="iteration_cost") and (cur_min_thr==0.0 or (throughput >= cur_min_thr)))
            if (cost_passed or thr_passed):
                sim_result = {
                    "duration":  duration,
                    "num_gpus": num_gpus_config,
                    "used_gpus_plan": used_gpus_plan,
                    "throughput": throughput,
                    "estimated_throughput": valid_plan['estimated_throughput'] if 'estimated_throughput' in valid_plan else 0,
                    "estimated_cost":  valid_plan['estimated_cost'] if 'estimated_cost' in valid_plan else 0,
                    "cost_per_iteration": comp_cost_iter + comm_cost,
                    "computation_cost": comp_cost,
                    "communication_cost": comm_cost,
                    "plan": reformed_plan_dict,
                    "search_time": search_time,
                    "oom_plans": oom_plans_case
                }
                # break
                sim_result_list.append(sim_result)

        # no valid plan found
        if not sim_result:
            throughput = 0.0
            sim_result = {
                "duration":  duration,
                "num_gpus": num_gpus_config,
                "used_gpus_plan": 0,
                "throughput": 0.0,
                "estimated_throughput": 0.0,
                "estimated_cost":  0.0,
                "cost_per_iteration": 0.0,
                "computation_cost": 0.0,
                "communication_cost": 0.0,
                "plan": 0.0,
                "search_time": search_time,
                "oom_plans": oom_plans_case
            }

        print(f"Throughput is {throughput}, Search_time is {search_time}")
        sim_result_list.append(sim_result)
        #break

    return sim_result_list


def accumulate(gpu_trace, throughputs, costs_per_sec):
    # given different throughputs and costs, get overall number of completed iterations and overall cost
    iterations = 0
    total_cost = 0

    for (t, _), throughput, cost in zip(gpu_trace, throughputs, costs_per_sec):
        iterations += throughput * t
        total_cost += cost * t

    return iterations, total_cost


def evaluate(args):

    os.environ['SAILOR_PATH'] = args.sailor_path
    with open(args.basic_cluster_config_json, 'r') as f:
        cluster_config = json.load(f)
        zone = cluster_config['zone']

    # operator level info
    if args.planner in ["Aceso"]:
        with open(f"{args.sailor_path}/sailor/sailor/Planner/llm_info_aceso.json", 'r') as f:
            llm_info = json.load(f)
            op_simulator = True
    # layer level info
    else:
        with open(f"{args.sailor_path}/sailor/sailor/Planner/llm_info.json", 'r') as f:
            llm_info = json.load(f)
            op_simulator = False

    if args.planner == 'Varuna':
        planner = VarunaPlanner(args.planner_profile_file, args.objective)
    elif args.planner == 'Oobleck':
        planner = OobleckPlanner(args.planner_profile_file)
    elif args.planner == 'AMP':
        planner = AMPPlanner(args.planner_profile_file, args.objective)
    elif args.planner == 'Piper':
        planner = PiperPlanner(args.planner_profile_file)
    elif args.planner == 'Metis':
        planner = MetisPlanner(
            args.planner_profile_file,
            args.training_config_json,
        )
    elif args.planner == 'Galvatron':
        planner = GalvatronPlanner(args.training_config_json, args.planner_profile_file, args.objective)  # TODO
    elif args.planner == 'Aceso':
        planner = AcesoPlanner(args.planner_profile_file, fp16=args.fp16)
    elif args.planner == 'Atlas':
        planner = AtlasPlanner(
            args.planner_profile_file,
            args.training_config_json,
            llm_info,
            args.fp16,
            args.objective
        )
    elif args.planner == 'DTFM':
        planner = DTFMPlanner(
            args.planner_profile_file,
            args.training_config_json,
            llm_info,
            args.fp16
        )
    elif args.planner == 'FlashFlex':
        planner = FlashFlexPlanner(
            args.planner_profile_file
        )
    elif args.planner == 'SAILOR':
        planner = SailorPlanner(
            args.sailor_profile_file_dir,
            args.training_config_json,
            args.quotas_dict,
            args.objective,
            args.fp16
        )
    else:
        raise NotImplementedError

    with open(args.basic_cluster_config_json, 'r') as f:
        basic_cluster_config = json.load(f)

    with open(args.training_config_json, 'r') as f:
        training_config = json.load(f)

    with open(args.simulator_profile_file, 'r') as f:
        simulator_profile_file = json.load(f)

    gpu_trace = parse_trace(args.trace_file)

    sim_result_list = evaluate_trace_with_mem(
        args.sailor_path,
        llm_info,
        planner,
        gpu_trace,
        basic_cluster_config,
        training_config,
        simulator_profile_file,
        fp16=args.fp16,
        op_simulator=op_simulator,
        objective=args.objective,
        max_cost=args.max_cost,
        max_cost_file=args.max_cost_file,
        min_throughput=args.min_throughput,
        min_throughput_file=args.min_throughput_file
    )

    Path(args.result_dir_path).mkdir(parents=True, exist_ok=True)
    res_path = f"{args.result_dir_path}/{args.planner}_{training_config['model']}.json"
    print(f"-------------------- Save results in path {res_path}")
    with open(res_path, 'w') as f:
        json.dump(sim_result_list, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sailor_path', type=str, required=True, help='Path to the sailor repo')
    parser.add_argument('--trace_file', type=str, required=True, help='GPU availability trace file')

    parser.add_argument('--basic_cluster_config_json', type=str, required=True,
                        help='Json file containing info for the cluster setup.')
    parser.add_argument('--training_config_json', type=str, required=True, help='Json file containing training info')
    parser.add_argument('--simulator_profile_file', type=str, required=True,
                        help='JSON file containing fwd, bwd and update time for different models and microbatches')
    parser.add_argument('--result_dir_path', type=str, required=True, help='Path to store results')

    parser.add_argument('--planner', type=str, default='',
                        help='Type of resource planner to use. Can choose between [Varuna, Oobleck,  AMP, SAILOR]')

    # for baselines
    parser.add_argument('--planner_profile_file', type=str, default='',
                        help='File containing profiling information used by each planner')

    # for SAILOR
    parser.add_argument('--sailor_profile_file_dir', type=str, default='',
                        help='Directory for SAILOR profiling')
    parser.add_argument('--quotas_dict', type=str, default='',
                        help='Json file containing user quotas')
    parser.add_argument('--objective', type=str, default='',
                        help="User objective ('throughput' or 'iteration_cost')")
    parser.add_argument('--max_cost', type=float, default=0.0,
                        help='Max cost (USD/iteration)')
    parser.add_argument('--max_cost_file', type=str, default=None,
                        help='A json file corresponding to different cost limits for the given trace')
    parser.add_argument('--min_throughput', type=float, default=0.0,
                        help='Min througput (iters/sec)')
    parser.add_argument('--min_throughput_file', type=str, default=None,
                        help='A json file corresponding to different throughput for the given trace')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')

    args = parser.parse_args()

    evaluate(args)
