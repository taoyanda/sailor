import json
import sys
import glob
from os.path import expanduser
from pathlib import Path
import argparse
import os

MAX_VALUE = 1<<30
parse_nan = lambda x: MAX_VALUE if str(x)=="NaN" else x

def gather_profiles(profile_dir, gpu_type):
    all_profs_dict = {}
    mem_info_dict = {}
    sim_dict = {}

    print(profile_dir)

    smallest_tmp=str(8)
    for file in glob.glob(f"{profile_dir}/*.json"):
        filename = os.path.basename(file)
        filename_tokens = filename.split("_")
        tmp = filename_tokens[-2]
        smallest_tmp = min(smallest_tmp, tmp)
        print(tmp, smallest_tmp)

    for file in glob.glob(f"{profile_dir}/*.json"):
        filename = os.path.basename(file)
        filename_tokens = filename.split(".")[0].split("_")
        tmp = filename_tokens[-2]
        mbs = filename_tokens[-1]

        print(tmp, mbs)

        with open(file, 'r') as f:
            profile_dict = json.load(f, parse_constant=parse_nan)

        if (mbs=="1"):
            mem_info_dict[tmp] = profile_dict['memory']

        if mbs not in all_profs_dict:
            all_profs_dict[mbs] = {}

        timing_prof = profile_dict['timing']
        all_profs_dict[mbs][tmp] = timing_prof

        if mbs not in sim_dict:
            sim_dict[mbs] = {}

        sim_dict[mbs][tmp] = timing_prof

    return all_profs_dict, mem_info_dict, sim_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gather profiling information for SAILOR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-name', type=str, required=True, help='Model name')
    parser.add_argument('--gpu-type', type=str, required=True, help='Model name')
    parser.add_argument('--profile-dir', type=str, required=True, help='Model name')
    parser.add_argument('--sailor-parent-dir', type=str, required=True, help='Parent directory to sailor repo')

    args = parser.parse_args()

    model_name = args.model_name
    gpu_type = args.gpu_type
    profile_dir = args.profile_dir

    all_profs_dict, mem_info_dict, sim_dict = gather_profiles(profile_dir, gpu_type)
    home_dir = args.sailor_parent_dir

    print(mem_info_dict)

    # # # SAILOR, timing
    par_dir = f'{home_dir}/sailor/sailor/Planner/sailor_planner/profiles/{model_name}/{gpu_type}'
    Path(par_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{par_dir}/profile.json', 'w') as f:
         json.dump(all_profs_dict, f, indent=2)

    # # memory
    with open(f'{home_dir}/sailor/sailor/Planner/llm_info.json', 'r') as f:
        all_mem_info = json.load(f)

    with open(f'{home_dir}/sailor/sailor/Planner/llm_info.json', 'w') as f:
        all_mem_info[model_name] = mem_info_dict
        json.dump(all_mem_info, f, indent=2)

    # # # simulations, timing
    with open(f'{home_dir}/sailor/sailor/Planner/simulations/profiles_tmp.json', 'r') as f:
        all_sim_dict = json.load(f)

    with open(f'{home_dir}/sailor/sailor/Planner/simulations/profiles_tmp.json', 'w') as f:
        if model_name not in all_sim_dict:
            all_sim_dict[model_name] = {}
        all_sim_dict[model_name][gpu_type] = sim_dict
        json.dump(all_sim_dict, f, indent=2)
