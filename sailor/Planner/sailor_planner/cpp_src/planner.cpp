#include "planner.hpp"

using namespace std;
using namespace chrono;

SailorPlanner::SailorPlanner() {}

SailorPlanner::SailorPlanner(
    const char *profile_path,
    const char *network_coeff_path,
    const char *training_config_path,
    const char *model_mem_info_file,
    const char *communication_cost_file,
    bool heterogeneous,
    bool fp16,
    const char *quotas_dict_path,
    const char *objective)
{
    heterogeneous = heterogeneous;

    fp16 = fp16;
    objective_str = objective;

    cout << profile_path << endl;
    cout << network_coeff_path << endl;
    cout << training_config_path << endl;
    cout << model_mem_info_file << endl;
    cout << communication_cost_file << endl;

    float_size = fp16 ? 2 : 4;

    std::ifstream coeffs_file(network_coeff_path, std::ifstream::binary);
    Json::Value coeffs;
    coeffs_file >> coeffs;

    // get model-specific input
    auto training_config = read_basic_json(training_config_path);
    auto model = training_config["model"].asString();
    auto model_mem_info = read_basic_json(model_mem_info_file);

    quotas_dict = read_quotas(quotas_dict_path);
    string profile_path_str = profile_path;

    for (auto it : quotas_dict)
    {
        // quotas_dict: { 'gpu_type' : { 'zone' : gpu_count }}
        auto gpu_type = it.first;
        cout << "AT QUOTAS DICT: " << gpu_type << endl;
        available_gpu_types.push_back(gpu_type);
        cost_per_gpu.push_back(COSTS[gpu_type] / 3600);
        string profile_path_str = profile_path;
        auto profile_file = profile_path_str + gpu_type + "/profile.json";
        profiles_all_gpus.push_back(read_llm_profile(profile_file.c_str()));
        for (auto t : it.second)
        {
            string zone = t.first;
            string region = extract_region_from_zone(zone);
            if (zones_per_region.find(region) == zones_per_region.end())
                zones_per_region[region] = {};
            zones_per_region[region].insert(zone);
            if (quotas_per_zone.find(zone) == quotas_per_zone.end())
                quotas_per_zone[zone] = {};
            if (quotas_per_zone[zone].find(gpu_type) == quotas_per_zone[zone].end())
                quotas_per_zone[zone][gpu_type] = 0;

            quotas_per_zone[zone][gpu_type] += t.second;
            max_all_quotas += t.second;
        }
    }
    network_coeffs_full = read_full_network_coeffs(network_coeff_path, available_gpu_types);
    num_diff_gpu_types = available_gpu_types.size();
    comm_cost = read_comm_cost(communication_cost_file);

    training_info = convertLLMInput(
        profiles_all_gpus,
        model_mem_info[model],
        training_config,
        float_size);

    for (int idx = 0; idx < available_gpu_types.size(); idx++)
    {
        for (int mbs : training_info->mbs_set[idx])
        {
            all_possible_mbs_set.insert(mbs);
            max_mbs = max(max_mbs, mbs);
        }
    }

    auto start = high_resolution_clock::now();
    build_structs();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("build_structs took %ld ms\n", duration.count());
}

void SailorPlanner::build_structs()
{
    // compute all stages
    int num_layers = training_info->model.num_layers;
    all_stages.push_back({});

    // 1. partitioning based on different types of parallelism
    for (int pp = 1; pp < num_layers + 1; pp++)
    {
        auto stages = get_stages(num_layers, pp);
        all_stages.push_back(stages);
    }

    // 2. get memory consumption (in parameters) per stage
    for (int tp = 0; tp < 9; tp++)
    {
        params_all_stages.push_back({});
        bytes_per_stage.push_back({});
        activation_per_stage.push_back({});
    }

    for (int tp : {1, 2, 4})
    {

        for (auto stages : all_stages)
        {
            for (auto stage : stages)
            {
                auto key = make_pair(stage[0], stage.size());
                size_t stage_num_params = 0.0;
                size_t stage_activations = 0.0;
                for (auto layer : stage)
                {
                    stage_num_params += (training_info->model).layer_params[tp][layer];
                    stage_activations += (training_info->model).activation_params[tp][layer];
                }
                params_all_stages[tp][key] = stage_num_params;
                bytes_per_stage[tp][key] = stage_num_params * float_size;
                activation_per_stage[tp][key] = stage_activations;
            }
        }
    }
    auto gpus = network_coeffs_full.begin();
    for (int idx = 0; idx < available_gpu_types.size(); idx++)
    {
        auto gpu_type = available_gpu_types[idx];
        cout << "Check for GPU " << gpu_type << endl;

        min_tmps_per_gpu.push_back(16);
        max_tmps_vector_per_gpu.push_back({});
        min_tmps_vector_per_gpu.push_back({});
        possible_tmps_per_gpu.push_back({});

        // Initialize empty array of size microbatch count for gpu type
        for (int i = 0; i < *(training_info->mbs_set[idx].rbegin()) + 1; i++)
        {
            max_tmps_vector_per_gpu[idx].push_back({});
            min_tmps_vector_per_gpu[idx].push_back({});
        }
        for (auto key : gpus->second[idx])
        {
            possible_tmps_per_gpu[idx].insert(possible_tmps_per_gpu[idx].begin(), key.first);
            min_tmps_per_gpu[idx] = min(min_tmps_per_gpu[idx], key.first);
        }

        // For all Microbatch sizes available for the GPU
        for (int mbs : training_info->mbs_set[idx])
        {
            // For all PP values
            for (auto stages : all_stages)
            {
                // For all stages of a PP value
                for (auto stage : stages)
                {
                    auto key = make_pair(stage[0], stage.size());

                    // MAX TMP, based on scaling
                    double prev_exec_time_tmp = 1000.0; // just a random big number
                    for (auto tp : possible_tmps_per_gpu[idx])
                    {
                        double exec_time_tmp = 0.0;
                        // for all layers in stage
                        for (auto layer : stage)
                        {
                            auto mbs_tp = make_pair(mbs, tp);
                            // if key exists (i.e. microbatch size, tensor parallelism?)
                            if ((training_info->model).profiles_per_gpu[idx].find(mbs_tp) != (training_info->model).profiles_per_gpu[idx].end())
                                exec_time_tmp += (training_info->model).profiles_per_gpu[idx][mbs_tp].exec_times[layer];
                        }
                        double speedup = prev_exec_time_tmp / exec_time_tmp;
                        if (speedup >= SPEEDUP_THRESHOLD)
                        {
                            prev_exec_time_tmp = exec_time_tmp;
                            max_tmps_vector_per_gpu[idx][mbs][key] = tp;
                        }
                        else
                            break;
                    }
                }
            }
        }

        // MIN TMP, based on memory
        for (int pp = 1; pp < training_info->model.num_layers + 1; pp++)
        {
            vector<vector<int>> stages = all_stages[pp];
            for (int mbs : training_info->mbs_set[idx])
            {
                for (int stage_idx = 0; stage_idx < stages.size(); stage_idx++)
                {
                    auto stage = stages[stage_idx];
                    auto pp_key = tuple<int, int, int>(pp, stage[0], stage.size());
                    // cout << gpu_type << "," << pp << "," << mbs << "," << stage_idx << endl;
                    int min_tp = -1;
                    for (auto tp : possible_tmps_per_gpu[idx])
                    {
                        if (check_stage_fits(stages, params_all_stages[tp], activation_per_stage[tp], stage_idx, mbs, tp, gpu_type, float_size))
                        {
                            min_tp = tp;
                            break;
                        }
                    }
                    min_tmps_vector_per_gpu[idx][mbs][pp_key] = min_tp;
                }
            }
        }
    }

    // TODO: check this
    for (const auto &r_z : zones_per_region)
        for (auto sender_zone : r_z.second)
            for (auto receiver_zone : r_z.second)
                precompute_ar_times_bottleneck(sender_zone, receiver_zone);

    // initialize dp
    for (int mbs = 0; mbs < max_mbs + 1; mbs++)
    {
        dp.push_back({});
        for (int pp = 0; pp < num_layers + 1; pp++)
        {
            dp[mbs].push_back({});
            for (int start_layer = 0; start_layer < num_layers; start_layer++)
            {
                dp[mbs][pp].push_back({});
                for (int data_par = 0; data_par < max_all_quotas; data_par++)
                {
                    dp[mbs][pp][start_layer].push_back({});
                }
            }
        }
    }

    // data parallelism
    for (int i = 0; i < *(all_possible_mbs_set.rbegin()) + 1; i++)
    {
        mbs_dp_vector.push_back({});
    }
    for (auto mbs : all_possible_mbs_set)
    {
        int num_ubatches_current = ceil(training_info->global_batch_size * 1.0 / mbs);
        mbs_dp_vector[mbs].push_back(1);
        for (int dp = 2; dp < max_all_quotas; dp++)
        {
            int num_ubatches = ceil(training_info->global_batch_size * 1.0 / (mbs * dp));
            if (num_ubatches != num_ubatches_current)
            {
                mbs_dp_vector[mbs].push_back(dp);
                num_ubatches_current = num_ubatches;
            }
        }
        // TODO: what to do with reverse?
        if (objective_str == "throughput")
            std::reverse(mbs_dp_vector[mbs].begin(), mbs_dp_vector[mbs].end());
    }
}

void SailorPlanner::precompute_ar_times_bottleneck(string sender_zone, string receiver_zone)
{
    pair<string, string> sendrecv = make_pair(sender_zone, receiver_zone);
    for (int idx = 0; idx < available_gpu_types.size(); idx++)
    {
        vector<vector<map<pair<int, int>, double>>> ar_times_gpu;
        for (int tp = 0; tp <= possible_tmps_per_gpu[idx].back(); tp++)
        {
            vector<map<pair<int, int>, double>> ar_times_gpu_tp;
            if (network_coeffs_full[sender_zone][idx].find(tp) != network_coeffs_full[sender_zone][idx].end() &&
                network_coeffs_full[sender_zone][idx][tp][receiver_zone][idx].find(tp) != network_coeffs_full[sender_zone][idx][tp][receiver_zone][idx].end())
            {
                for (int num_peers = 0; num_peers < 1300; num_peers++)
                {
                    map<pair<int, int>, double> ar_per_stage = {};

                    for (auto stages : all_stages)
                    {
                        for (auto stage : stages)
                        {
                            auto key = make_pair(stage[0], stage.size());
                            double tsync_stage = get_ar_time_with_buckets(
                                bytes_per_stage[1][key],
                                MEMORY_BUCKET_DEEPSPEED_SIZE, num_peers,
                                network_coeffs_full[sender_zone][idx][tp][receiver_zone][idx][tp].first, known_times);
                            // double mul_factor = 2.0;
                            // if (num_peers % 2 != 0)
                            //     mul_factor = 3.0;
                            double mul_factor = 1.0;
                            ar_per_stage[key] = tsync_stage * mul_factor;
                        }
                    }
                    ar_times_gpu_tp.push_back(ar_per_stage);
                }
            }
            ar_times_gpu.push_back(ar_times_gpu_tp);
        }
        ar_times_bottleneck[sendrecv].push_back(ar_times_gpu);
    }
}

void debug_print(vector<int> input_vector)
{
    for (auto v : input_vector)
    {
        cout << v << ",";
    }
    cout << " ....";
    cout << endl;
}

vector<vector<int>> generate_all_combos_new(
    vector<int> &max_num_gpus, int id, int data_par, int n)
{
    /*
     *
     * @param max_num_gpus: vector containing per gpu type the maximum number of GPUs available
     * @param id: GPU idx, used for recursion purposes (start from 0)
     * @param data_par: data parallelism degree
     * @param n: number of GPUs, should be max_num_gpus.size() == n
     */
    int max_this_gpu = min(data_par, max_num_gpus[id]);
    if ((id == n - 1))
    {
        if (max_this_gpu == data_par)
            return {{max_this_gpu}};
        else
            return {{}};
    }

    vector<vector<int>> new_res;
    for (int num_gpu = 0; num_gpu <= max_this_gpu; num_gpu++)
    {
        auto res = generate_all_combos_new(max_num_gpus, id + 1, data_par - num_gpu, n);
        for (int j = 0; j < res.size(); j++)
        {
            if (!res[j].empty())
                res[j].insert(res[j].begin(), num_gpu);
        }
        new_res.insert(new_res.end(), res.begin(), res.end());
    }
    return new_res;
}

struct ParallelismConfig *SailorPlanner::solve_dp(
    int mbs,
    int pp,
    int data_par,
    int num_layers,
    int og_stage_idx,
    const int curr_region,
    const string &dp_resource_hash_string,
    const vector<string> &regions_list,
    unordered_map<string, vector<pair<string, vector<int>>>> &resources,
    string max_cur_budget,
    vector<vector<int>> &stages,
    vector<vector<int>> &tmp_degrees)
{

    int start_layer = stages[og_stage_idx][0];
    // printf("Inside solve_dp, mbs is %d, pp is %d, start_layer is %d, max_iter_time is %f\n", mbs, pp, start_layer, max_iter_time);
    //  cout << "max budget is: " << max_cur_budget << endl;
    //  cout << "dp_resource_hash_string is " << dp_resource_hash_string << endl;
    if (dp[mbs][pp][start_layer][data_par].find(dp_resource_hash_string) != dp[mbs][pp][start_layer][data_par].end())
    {
        if (dp[mbs][pp][start_layer][data_par][dp_resource_hash_string].find(max_cur_budget) != dp[mbs][pp][start_layer][data_par][dp_resource_hash_string].end())
            return &(dp[mbs][pp][start_layer][data_par][dp_resource_hash_string][max_cur_budget]);
    }

    vector<int> stage = stages[og_stage_idx];
    auto key = make_pair(stage[0], stage.size());
    vector<vector<int>> resources_combo;
    int num_gpu_types = available_gpu_types.size();

    // get number of VMs for each gpu type at the current region
    vector<int> rec_id(num_gpu_types, 0);
    string region = regions_list[curr_region];
    for (int i = 0; i < num_gpu_types; i++)
    {
        for (auto zone : resources[region])
        {
            if ((zone.second[i] > 0) && (tmp_degrees[i].size() > 0))
            {
                rec_id[i] += zone.second[i] / tmp_degrees[i][og_stage_idx];
            }
        }
    }
    resources_combo = generate_all_combos_new(rec_id, 0, data_par, num_gpu_types);
    struct ParallelismConfig dummy_config;
    dp[mbs][pp][start_layer][data_par][dp_resource_hash_string][max_cur_budget] = dummy_config;

    // convert cost back to float
    float max_cur_budget_float = stof(max_cur_budget);

    // auto next_start = high_resolution_clock::now();
    if (pp == 1)
    {
        double prev_Tpp = 1000000.0;
        float prev_cost = 1000000.0;
        struct ParallelismConfig prev_pconfig;

        vector<pair<int, int>> prev_all_tp(num_gpu_types, make_pair(0, -1));
        for (auto resource_combo : resources_combo)
        {
            if (resource_combo.size() < num_gpu_types)
                continue;

            // printf("PP %d, Check resource combo: ", pp);
            // debug_print(resource_combo);

            vector<tuple<int, int, int>> gpu_tp_pairs = {};
            vector<pair<int, int>> all_tp(num_gpu_types, make_pair(0, -1));
            int dp_degree = 0;
            double comp_cost = 0.0;
            set<string> all_zones = {};

            for (int idx = 0; idx < num_gpu_types; idx++)
            {
                int curr_zone = 0; // TODO: keep it here or move it out?
                if (resource_combo[idx] == 0)
                    continue;
                string czone = resources[region][curr_zone].first;

                int tp = tmp_degrees[idx][og_stage_idx];
                all_tp[idx] = make_pair(tp, zone_to_id[czone]);
                int dp_gpu_type = resource_combo[idx];
                int residual_gpus = resources[region][curr_zone].second[idx];
                int dp_gpu = 0;
                while (dp_gpu < dp_gpu_type)
                {
                    // printf("dp_gpu is %d, curr_zone is %d, residual_gpus is %d, tp is %d\n", dp_gpu, curr_zone, residual_gpus, tp);
                    if (residual_gpus < tp)
                    { // If not enough GPUs left in zone, drop zone and move on (multizone dp)
                        curr_zone++;
                        if (curr_zone == resources[region].size())
                        { // no more GPUs in the region
                            break;
                        }
                        residual_gpus = resources[region][curr_zone].second[idx];
                        czone = resources[region][curr_zone].first;
                    }
                    // cout << "------------------ " << czone << "," << zone_to_id[czone] << endl;
                    gpu_tp_pairs.push_back(make_tuple(idx, tp, zone_to_id[czone]));
                    all_zones.insert(czone);
                    residual_gpus -= tp;
                    dp_gpu++;
                }
                if (dp_gpu != dp_gpu_type)
                    break;
                dp_degree += dp_gpu;
                comp_cost += dp_gpu * tp * cost_per_gpu[idx];
            }

            if (dp_degree != data_par)
                continue;

            struct PipelineInfo pinfo = simulate_time_single_stage(
                stage,
                all_tp,
                all_zones,
                id_to_zone,
                training_info,
                mbs,
                network_coeffs_full,
                ar_times_bottleneck,
                params_all_stages[1][key] * float_size,
                data_par);

            float total_cost = pinfo.Tpp * comp_cost + pinfo.ar_cost;
            // printf("Tpp is %f, prev is %f, total_cost is %f, prev_cost is %f\n", pinfo.Tpp, prev_Tpp, total_cost, prev_cost);

            // 2. Set at dp array
            if (objective_str == "throughput")
            {
                bool in_budget = ((max_cur_budget_float == 0.0) || total_cost <= max_cur_budget_float);
                if ((pinfo.Tpp < prev_Tpp) && in_budget)
                {
                    struct StageConfig config(dp_degree, {}, gpu_tp_pairs);
                    struct ParallelismConfig new_config(
                        {config},
                        pinfo.Tpp,
                        pinfo.ar_cost,
                        comp_cost,
                        pinfo);
                    dp[mbs][pp][start_layer][data_par][dp_resource_hash_string][max_cur_budget] = new_config;
                    prev_Tpp = pinfo.Tpp;
                    prev_pconfig = new_config;
                }
                else
                {
                    dp[mbs][pp][start_layer][data_par][dp_resource_hash_string][max_cur_budget] = prev_pconfig;
                }
            }
            else
            {
                bool in_throughput = ((max_iter_time == 0.0) || pinfo.Tpp <= max_iter_time);
                if (((total_cost <= prev_cost) && in_throughput))
                {
                    struct StageConfig config(dp_degree, {}, gpu_tp_pairs);
                    struct ParallelismConfig new_config(
                        {config},
                        pinfo.Tpp,
                        pinfo.ar_cost,
                        comp_cost,
                        pinfo);
                    dp[mbs][pp][start_layer][data_par][dp_resource_hash_string][max_cur_budget] = new_config;
                    prev_cost = total_cost;
                    prev_pconfig = new_config;
                }
                else
                {
                    dp[mbs][pp][start_layer][data_par][dp_resource_hash_string][max_cur_budget] = prev_pconfig;
                }
            }
            prev_all_tp = all_tp;
        }
    }
    else
    {
        double best_Tpp = 1000000.0;
        float best_cost = 100000.0;
        double prev = 1000000.0;
        struct ParallelismConfig best_pconfig;
        vector<int> best_combo;

        long int sum = 0.0;
        for (auto resource_combo : resources_combo)
        {
            int next_region = curr_region;
            if (resource_combo.size() < num_gpu_types)
                continue;

            // printf("------------- PP %d, Region is %d, Regions list size is %d\n ", pp, next_region, regions_list.size());
            // debug_print(resource_combo);

            set<string> all_zones = {};
            vector<tuple<int, int, int>> gpu_tp_pairs = {};
            vector<pair<int, int>> all_tp(num_gpu_types, make_pair(0, -1));
            int dp_degree = 0;
            double comp_cost = 0.0;

            unordered_map<string, vector<pair<string, vector<int>>>> new_resources = resources;
            for (int idx = 0; idx < num_gpu_types; idx++)
            {
                int curr_zone = 0; // TODO: keep it here or move it out?
                if (resource_combo[idx] == 0)
                    continue;
                string czone = resources[region][curr_zone].first;
                all_zones.insert(czone);

                int tp = tmp_degrees[idx][og_stage_idx];
                all_tp[idx] = make_pair(tp, zone_to_id[czone]);
                int dp_gpu_type = resource_combo[idx];
                int residual_gpus = resources[region][curr_zone].second[idx];
                int dp_gpu = 0;
                while (dp_gpu < dp_gpu_type)
                {
                    // printf("----------------------- dp_gpu is %d, curr_zone is %d, total_zones is %d, residual_gpus is %d, tp is %d\n", dp_gpu, curr_zone, resources[region].size(), residual_gpus, tp);
                    while (residual_gpus < tp)
                    { // If not enough GPUs left in zone, drop zone and move on (multizone dp)
                        new_resources[region][curr_zone].second[idx] = max(residual_gpus, 0);
                        curr_zone++;
                        if (curr_zone == resources[region].size())
                        { // no more GPUs in the region
                            break;
                        }
                        residual_gpus = resources[region][curr_zone].second[idx];
                        czone = resources[region][curr_zone].first;
                    }
                    gpu_tp_pairs.push_back(make_tuple(idx, tp, zone_to_id[czone]));
                    residual_gpus -= tp;
                    if (residual_gpus < 0)
                        break;
                    dp_gpu++;
                }
                if (dp_gpu != dp_gpu_type)
                    break;
                if (curr_zone < resources[region].size())
                    new_resources[region][curr_zone].second[idx] = max(residual_gpus, 0);
                comp_cost += dp_gpu * tp * cost_per_gpu[idx];
                dp_degree += dp_gpu;
            }

            if (dp_degree != data_par)
                continue;

            // Count max possible dp given new resources in all zones
            string new_hash_string = region;
            vector<int> new_gpu_counts(num_gpu_types, 0);
            int next_stage_max_dp = 0;
            for (auto zone : new_resources[region])
            {
                for (int idx = 0; idx < num_gpu_types; idx++)
                {
                    int gpu_count = zone.second[idx];

                    if (tmp_degrees[idx].size() > 0 && tmp_degrees[idx][og_stage_idx + 1])
                    {
                        next_stage_max_dp += gpu_count / tmp_degrees[idx][og_stage_idx + 1];
                    }
                    new_gpu_counts[idx] += gpu_count;
                }
            }

            if (next_stage_max_dp < data_par)
            {
                next_region++; // change region here (but only once)
                assert(next_region == curr_region + 1);
                if (next_region >= regions_list.size())
                    continue; // if all regions were utilized try next combo
                // Update hash string
                auto new_region = regions_list[next_region];
                new_hash_string = new_region;
                for (int idx = 0; idx < num_gpu_types; idx++)
                    new_gpu_counts[idx] = 0;
                for (auto zone : resources[new_region])
                {
                    for (int idx = 0; idx < zone.second.size(); idx++)
                    {
                        new_gpu_counts[idx] += zone.second[idx];
                    }
                }
            }

            for (int i = 0; i < num_gpu_types; i++)
            {
                new_hash_string += "_" + to_string(i) + "-" + to_string(new_gpu_counts[i]);
            }

            struct StageConfig config(dp_degree, {}, gpu_tp_pairs);
            // 1. Get time giving it i VMs
            struct PipelineInfo pinfo = simulate_time_single_stage(
                stage,
                all_tp,
                all_zones,
                id_to_zone,
                training_info,
                mbs,
                network_coeffs_full,
                ar_times_bottleneck,
                params_all_stages[1][key] * float_size,
                data_par);

            float total_cost = pinfo.Tpp * comp_cost + pinfo.ar_cost;
            bool in_budget = (objective_str == "throughput") && ((max_cur_budget_float == 0.0) || total_cost <= max_cur_budget_float);
            if (!in_budget)
                continue;
            if ((objective_str == "throughput") && (pinfo.Tpp == prev))
                continue;
            prev = pinfo.Tpp;

            float remaining_budget = max(max_cur_budget_float - total_cost, (float)(0.0));
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << remaining_budget;
            string remaining_budget_str = stream.str();

            double new_Tpp = 0.0;
            float new_comm_cost = 0.0;
            float new_comp_cost = 0.0;
            float new_cost = 0.0;

            vector<struct StageConfig> config_array;

            do
            {
                config_array.clear();
                // 1. Get time giving the rest VMs to the rest stages
                auto best_rem_combo = solve_dp(
                    mbs,                  // int mbs
                    pp - 1,               // int pp
                    data_par,             // int data_par
                    num_layers,           // int num_layers
                    og_stage_idx + 1,     // int og_stage_idx
                    next_region,          // const int curr_region
                    new_hash_string,      // const string &dp_resource_hash_string
                    regions_list,         // const vector<string> &regions_list
                    new_resources,        // unordered_map<string, unordered_map<string, vector<int>>> &resources
                    remaining_budget_str, // remaining budget
                    stages,               // vector<vector<int>> &stages
                    tmp_degrees           // vector<vector<int>> &tmp_degrees
                );
                struct ParallelismConfig rest_parallel_config = *best_rem_combo;
                if ((rest_parallel_config.Tpp == 0.0) || (rest_parallel_config.config_per_stage[0].dp != dp_degree))
                    break;

                config_array.push_back(config);
                config_array.insert(
                    config_array.end(),
                    rest_parallel_config.config_per_stage.begin(),
                    rest_parallel_config.config_per_stage.end());

                pair<double, double> Tpp_cost = merge_stages_get_time(
                    &pinfo,
                    &(rest_parallel_config.pp_info),
                    training_info,
                    dp_degree,
                    mbs,
                    float_size,
                    0,
                    stages.size(),
                    stages,
                    config_array,
                    network_coeffs_full,
                    known_times,
                    comm_cost,
                    id_to_zone);

                new_comm_cost = Tpp_cost.second + pinfo.ar_cost + best_rem_combo->comm_cost;
                new_comp_cost = comp_cost + best_rem_combo->comp_cost;
                new_cost = new_Tpp * new_comp_cost + new_comm_cost;
                // printf("Comm cost1: %f, Comm cost2: %f, PP cost: %f\n", pinfo.ar_cost, best_rem_combo->comm_cost, Tpp_cost.second);

                if (objective_str == "throughput")
                {
                    // is the found config in budget? If yes, valid solution
                    bool total_config_in_budget = ((max_cur_budget_float == 0.0) || new_cost <= max_cur_budget_float);
                    if (total_config_in_budget)
                    {
                        new_Tpp = Tpp_cost.first;
                        break;
                    }
                    // if not:
                    // is the current stage the straggler? If so, the limit was correct, no other solution possible, continue
                    // if not, test with new cost bound
                    if (rest_parallel_config.pp_info.straggler <= pinfo.straggler)
                        break;
                    else
                    {
                        remaining_budget = max(max_cur_budget_float - new_cost, (float)(0.0));
                        std::stringstream stream;
                        stream << std::fixed << std::setprecision(2) << remaining_budget;
                        remaining_budget_str = stream.str();
                    }
                }
                else
                {
                    if (max_iter_time == 0.0 || (Tpp_cost.first <= max_iter_time))
                        new_Tpp = Tpp_cost.first;
                }

            } while (remaining_budget > 0);

            if ((new_Tpp > 0.0) && ((objective_str == "throughput" && (new_Tpp < best_Tpp)) || (objective_str == "iters_dollar" && (new_cost < best_cost))))
            {
                struct ParallelismConfig new_config(config_array, new_Tpp, new_comm_cost, new_comp_cost, pinfo);
                best_Tpp = new_Tpp;
                best_cost = new_cost;
                best_pconfig = new_config;
                dp[mbs][pp][start_layer][data_par][dp_resource_hash_string][max_cur_budget] = best_pconfig;
            }
        }
    }

    // auto next_stop = high_resolution_clock::now();
    // auto next_duration = duration_cast<microseconds>(next_stop - next_start);
    // printf("P3 took %ld us\n", next_duration.count());

    return &(dp[mbs][pp][start_layer][data_par][dp_resource_hash_string][max_cur_budget]);
}

vector<vector<int>> SailorPlanner::get_pp_tp_combos(int pp, int gpu_id) {
    if (pp==0) {
        return {{}};
    }
    vector<vector<int>> result;
    vector<vector<int>> combos = get_pp_tp_combos(pp-1, gpu_id);
    for (auto combo: combos) {
        for (int tp: possible_tmps_per_gpu[gpu_id]) {
            vector<int> combo_copy = combo;
            combo_copy.push_back(tp);
            result.push_back(combo_copy);
        }
    }

    return result;

}

vector<vector<vector<int>>> SailorPlanner::get_tmp_degrees_for_all(int pp, int num_types) {

    assert (num_types <= 2); // just for experimentation

    vector<vector<vector<int>>> result = {};
    vector<vector<vector<int>>> per_gpu_tp = {};
    for (int id=0; id<num_types; id++) {
        vector<vector<int>> gpu_tps = get_pp_tp_combos(pp, id); // TODO
        per_gpu_tp.push_back(gpu_tps);
    }

    if (num_types==1) {
        for (int i=0; i<per_gpu_tp[0].size(); i++) {
            result.push_back({per_gpu_tp[0][i], {}});
        }
    }
    else {
        for (int i=0; i<per_gpu_tp[0].size(); i++) {
            for (int j=0; j<per_gpu_tp[1].size(); j++) {
                result.push_back({per_gpu_tp[0][i], per_gpu_tp[1][j]});
            }
        }
    }

    return result;

}

void SailorPlanner::regions_sorting(
    const unordered_map<string, vector<pair<string, vector<int>>>> &given_resources)
{
    regions = {};
    region_gpu_count = {};
    zones_per_region_combo = {};

    int gpu_types = available_gpu_types.size();
    for (const auto &region_entry : given_resources)
    {
        const string &region = region_entry.first;
        const auto &zones = region_entry.second;
        zones_per_region_combo[region] = {};
        vector<int> total_gpus_per_type(gpu_types, 0);
        for (const auto &zone_entry : zones)
        {
            zones_per_region_combo[region].push_back(zone_entry.first);
            const vector<int> &gpu_counts = zone_entry.second;
            for (int i = 0; i < gpu_types && i < (int)gpu_counts.size(); ++i)
            {
                total_gpus_per_type[i] += gpu_counts[i];
            }
        }

        regions.push_back(region);
        region_gpu_count[region] = total_gpus_per_type;
    }

    for (const auto &r1 : regions)
    {
        const string &zone1 = zones_per_region_combo[r1][0]; // TODO should it take the first zone or the maximum TP zone?
        for (const auto &r2 : regions)
        {
            if (r1 == r2)
                continue;
            const string &zone2 = zones_per_region_combo[r2][0];
            if (!zone1.empty() && !zone2.empty())
            {
                // Using 1 gpu count per type, sufficient results for comparison
                double th = get_max_throughput(
                    this->network_coeffs_full,
                    zone1, 1, zone2, 1);
                region_throughput[r1][r2] = th;
            }
            else
            {
                region_throughput[r1][r2] = 0.0;
            }
        }
    }
}

void SailorPlanner::get_plans_no_heuristics(unordered_map<string, vector<pair<string, vector<int>>>> &given_resources)
{

    regions_sorting(given_resources);
    int region_count = 1;
    int max_regions = given_resources.size();
    int num_layers = training_info->model.num_layers;
    for (int pp = 1; pp < num_layers + 1; pp++)
    {
        printf("-------------------------------------------------------------------------------------------------------- Check for PP: %d\n", pp);
        for (int mbs : all_possible_mbs_set)
        {
            region_count = 1;
            printf("**************************************************** Check for MBS: %d\n", mbs);

            set<int> valid_gpus = {};
            vector<vector<int>> stages = all_stages[pp];
            // vector<vector<int>> tmp_degrees = find_tmp_degrees(
            //     stages,
            //     training_info,
            //     max_tmps_vector_per_gpu,
            //     min_tmps_vector_per_gpu,
            //     possible_tmps_per_gpu,
            //     mbs,
            //     available_gpu_types.size(),
            //     float_size);

            // permutations:
            vector<vector<vector<int>>> tmp_degrees_all = get_tmp_degrees_for_all(pp, 1); // TODO

            for (auto tmp_degrees : tmp_degrees_all)
            {

                printf("size is %d\n", available_gpu_types.size());
                for (int id=0; id < available_gpu_types.size(); id++) {
                    printf("GPU %d: \n", id);
                    cout << available_gpu_types[id] << endl;
                    debug_print(tmp_degrees[id]);
                }

                printf("HERE!\n");

                // do regular search for that combo
                vector<pair<string, vector<string>>> region_list = get_regions_list(
                    given_resources,
                    region_gpu_count,
                    zones_per_region_combo,
                    regions,
                    region_throughput,
                    this->network_coeffs_full,
                    tmp_degrees);

                printf("Permutation size is %d\n", region_list.size());
                bool regions_bottleneck = false;

                for (auto region_combo : region_list)
                {
                    int max_dp = 0;
                    cout << "++++++++++++++++++++++++++++++++++++++++ Check for regions: ";
                    for (string region : region_combo.second)
                        cout << region << ", ";
                    cout << endl;

                    int max_dp_mbs = -1;

                    for (int i = 0; i < num_diff_gpu_types; i++)
                    {
                        int num_gpus_pipeline = 0;
                        for (auto tp : tmp_degrees[i])
                        {
                            num_gpus_pipeline += tp;
                        }
                        if (num_gpus_pipeline == 0)
                            continue;
                        int total_resources = 0;
                        for (string reg : region_combo.second)
                        {
                            for (auto zone_resources : given_resources[reg])
                            {
                                total_resources += zone_resources.second[i];
                            }
                        }
                        int region_dp = total_resources / num_gpus_pipeline;
                        max_dp += region_dp;
                    }

                    printf("MAX_DP is %d\n", max_dp);

                    double prev_dp_Tpp = 1000000.0;
                    float prev_total_cost = 1000.0;
                    if (max_dp_mbs < 0)
                        max_dp_mbs = max_dp;

                    bool in_budget_config = false;
                    bool in_throughput_config = false;
                    for (auto data_par : mbs_dp_vector[mbs])
                    {

                        // if (data_par > max_dp)
                        //     continue;

                        // if (data_par > max_dp_mbs)
                        //     continue;

                        printf("----------- CHECK WITH D %d, max_dp is %d\n", data_par, max_dp);

                        extra_cost = 0.0;
                        std::stringstream stream;
                        stream << std::fixed << std::setprecision(2) << max_budget;
                        string max_cur_budget = stream.str();
                        cout << "max cur budget is " << max_cur_budget << endl;

                        auto start = high_resolution_clock::now();
                        auto best_combo = solve_dp(
                            mbs,                 // int mbs
                            pp,                  // int pp
                            data_par,            // int data_par
                            num_layers,          // int num_layers
                            0,                   // int og_stage_idx
                            0,                   // const int curr_region
                            region_combo.first,  // const string &dp_resource_hash_string
                            region_combo.second, // const vector<string> &regions_list
                            given_resources,     // unordered_map<string, unordered_map<string, vector<int>>> &resources
                            max_cur_budget,      // max budget for the whole config
                            stages,              // vector<vector<int>> &stages
                            tmp_degrees          // vector<vector<int>> &tmp_degrees
                        );
                        auto stop = high_resolution_clock::now();
                        auto duration = duration_cast<microseconds>(stop - start);

                        dp_solve_total += duration.count();
                        struct ParallelismConfig config = *best_combo;

                        printf("----------------------------------------------- Tpp is %f, solve_dp duration is %ld us, extra_cost is %lu\n", config.Tpp, duration.count(), extra_cost);
                        if (config.Tpp > 0.0)
                        {
                            float total_cost = config.comm_cost + config.comp_cost * config.Tpp;
                            printf("TOTAL COST IS %f, comm_cost is %f, comp_cost is %f, max budget is %f, Tpp is %f\n", total_cost, config.comm_cost, config.comp_cost, max_budget, config.Tpp);

                            if (objective_str == "throughput")
                            {

                                if (prev_dp_Tpp > config.Tpp)
                                    if (max_budget == 0.0)
                                        prev_dp_Tpp = config.Tpp;
                                    else
                                    {
                                        if (total_cost < max_budget)
                                        {
                                            prev_dp_Tpp = config.Tpp;
                                            in_budget_config = true;
                                        }
                                    }
                                else
                                {
                                    regions_bottleneck = true;
                                    // if a cost limit is given, it is worth visiting configs with smaller DP
                                    // break if a config that a good config is found
                                    // part of H3
                                    // if ((max_budget == 0.0) || in_budget_config)
                                    //     break;
                                }
                            }
                            else
                            {
                                printf("config.Tpp is %f, max_iter_time is %f\n", config.Tpp, max_iter_time);

                                // we optimize for cost - stop when cost stops improving
                                if (prev_total_cost > total_cost)
                                {
                                    if (max_iter_time == 0.0)
                                        prev_total_cost = total_cost;
                                    else
                                    {
                                        if (config.Tpp <= max_iter_time)
                                        {
                                            prev_total_cost = total_cost;
                                            in_throughput_config = true;
                                        }
                                    }
                                }
                                else
                                {
                                    if ((max_iter_time == 0.0) || in_throughput_config)
                                        break;
                                }
                            }

                            struct Config new_config(
                                stages,
                                mbs,
                                config.Tpp,
                                total_cost,
                                config.comm_cost,
                                config.comp_cost,
                                config.config_per_stage,
                                id_to_zone);
                            all_configs.push_back(new_config);
                        }
                        else
                        {
                            if (objective_str == "throughput")
                                max_dp_mbs = data_par - 1;
                        }
                    }
                    // if (regions_bottleneck) // dp bottleneck found, dont look for more regions
                    //     break;
                }
            }
        }
    }
}

void SailorPlanner::get_plans_num_gpus_dp(unordered_map<string, vector<pair<string, vector<int>>>> &given_resources)
{

    regions_sorting(given_resources);
    int region_count = 1;
    int max_regions = given_resources.size();
    int num_layers = training_info->model.num_layers;
    for (int pp = 1; pp < num_layers + 1; pp++)
    {
        printf("-------------------------------------------------------------------------------------------------------- Check for PP: %d\n", pp);
        for (int mbs : all_possible_mbs_set)
        {
            region_count = 1;
            printf("**************************************************** Check for MBS: %d\n", mbs);

            set<int> valid_gpus = {};
            vector<vector<int>> stages = all_stages[pp];
            vector<vector<int>> tmp_degrees = find_tmp_degrees(
                stages,
                training_info,
                max_tmps_vector_per_gpu,
                min_tmps_vector_per_gpu,
                possible_tmps_per_gpu,
                mbs,
                available_gpu_types.size(),
                float_size,
                homogeneous
            );

            // permutations:

            vector<pair<string, vector<string>>> region_list = get_regions_list(
                given_resources,
                region_gpu_count,
                zones_per_region_combo,
                regions,
                region_throughput,
                this->network_coeffs_full,
                tmp_degrees);

            printf("Permutation size is %d\n", region_list.size());
            bool regions_bottleneck = false;

            for (auto region_combo : region_list)
            {
                int max_dp = 0;
                cout << "++++++++++++++++++++++++++++++++++++++++ Check for regions: ";
                for (string region : region_combo.second)
                    cout << region << ", ";
                cout << endl;

                int max_dp_mbs = -1;

                for (int i = 0; i < num_diff_gpu_types; i++)
                {
                    int num_gpus_pipeline = 0;
                    for (auto tp : tmp_degrees[i])
                    {
                        num_gpus_pipeline += tp; // get min
                    }
                    if (num_gpus_pipeline == 0)
                        continue;
                    int total_resources = 0;
                    for (string reg : region_combo.second)
                    {
                        for (auto zone_resources : given_resources[reg])
                        {
                            total_resources += zone_resources.second[i];
                        }
                    }
                    int region_dp = total_resources / num_gpus_pipeline;
                    //printf("GPU %d, region_dp %d, total_resources %d, num_gpus_pipeline %d\n", i, region_dp, total_resources, num_gpus_pipeline);
                    max_dp += region_dp;
                }

                printf("MAX_DP is %d\n", max_dp);

                double prev_dp_Tpp = 1000000.0;
                float prev_total_cost = 1000.0;
                if (max_dp_mbs < 0)
                    max_dp_mbs = max_dp;

                bool in_budget_config = false;
                bool in_throughput_config = false;
                for (auto data_par : mbs_dp_vector[mbs])
                {

                    // if (data_par > max_dp)
                    //     continue;

                    // if (data_par > max_dp_mbs)
                    //     continue;

                    printf("----------- CHECK WITH D %d, max_dp is %d\n", data_par, max_dp);

                    extra_cost = 0.0;
                    std::stringstream stream;
                    stream << std::fixed << std::setprecision(2) << max_budget;
                    string max_cur_budget = stream.str();
                    cout << "max cur budget is " << max_cur_budget << endl;

                    auto start = high_resolution_clock::now();
                    auto best_combo = solve_dp(
                        mbs,                 // int mbs
                        pp,                  // int pp
                        data_par,            // int data_par
                        num_layers,          // int num_layers
                        0,                   // int og_stage_idx
                        0,                   // const int curr_region
                        region_combo.first,  // const string &dp_resource_hash_string
                        region_combo.second, // const vector<string> &regions_list
                        given_resources,     // unordered_map<string, unordered_map<string, vector<int>>> &resources
                        max_cur_budget,      // max budget for the whole config
                        stages,              // vector<vector<int>> &stages
                        tmp_degrees          // vector<vector<int>> &tmp_degrees
                    );
                    auto stop = high_resolution_clock::now();
                    auto duration = duration_cast<microseconds>(stop - start);

                    dp_solve_total += duration.count();
                    struct ParallelismConfig config = *best_combo;

                    printf("----------------------------------------------- Tpp is %f, solve_dp duration is %ld us, extra_cost is %lu\n", config.Tpp, duration.count(), extra_cost);
                    if (config.Tpp > 0.0)
                    {
                        float total_cost = config.comm_cost + config.comp_cost * config.Tpp;
                        printf("TOTAL COST IS %f, comm_cost is %f, comp_cost is %f, max budget is %f, Tpp is %f\n", total_cost, config.comm_cost, config.comp_cost, max_budget, config.Tpp);

                        if (objective_str == "throughput")
                        {

                            if (prev_dp_Tpp > config.Tpp)
                                if (max_budget == 0.0)
                                    prev_dp_Tpp = config.Tpp;
                                else
                                {
                                    if (total_cost < max_budget)
                                    {
                                        prev_dp_Tpp = config.Tpp;
                                        in_budget_config = true;
                                    }
                                }
                            else
                            {
                                regions_bottleneck = true;
                                // if a cost limit is given, it is worth visiting configs with smaller DP
                                // break if a config that a good config is found
                                // part of H3
                                if ((max_budget == 0.0) || in_budget_config)
                                    break;
                            }
                        }
                        else
                        {
                            printf("config.Tpp is %f, max_iter_time is %f\n", config.Tpp, max_iter_time);

                            // we optimize for cost - stop when cost stops improving
                            if (prev_total_cost > total_cost)
                            {
                                if (max_iter_time == 0.0)
                                    prev_total_cost = total_cost;
                                else
                                {
                                    if (config.Tpp <= max_iter_time)
                                    {
                                        prev_total_cost = total_cost;
                                        in_throughput_config = true;
                                    }
                                }
                            }
                            else
                            {
                                if ((max_iter_time == 0.0) || in_throughput_config)
                                    break;
                            }
                        }

                        struct Config new_config(
                            stages,
                            mbs,
                            config.Tpp,
                            total_cost,
                            config.comm_cost,
                            config.comp_cost,
                            config.config_per_stage,
                            id_to_zone);
                        all_configs.push_back(new_config);
                    }
                    else
                    {
                        if (objective_str == "throughput")
                            max_dp_mbs = data_par - 1;
                    }
                }
                // if (regions_bottleneck) // dp bottleneck found, dont look for more regions
                //     break;
            }
            // if (max_dp_mbs <= 0)
            //     break;
        }
    }
}

vector<struct Config> SailorPlanner::get_sorted_plans(unordered_map<string, unordered_map<string, int>> max_num_gpus, float max_budget_arg, double min_throughput_arg)
{
    /**
     * Solve dp and get plans sorted according to the defined objective
     * @params: max_num_gpus {'zone' : {'gpu_type' : gpu_count}}
     * @returns: all_configs vector sorted according to objective
     */
    // {'region' : [(zone1,  gpu_count), (zone2, gpu_count), ...]}
    max_budget = max_budget_arg;
    if (min_throughput_arg > 0.0)
        max_iter_time = 1 / min_throughput_arg;
    else
        max_iter_time = 0.0;
    unordered_map<string, vector<pair<string, vector<int>>>> given_resources = {};
    string hash_string = "";
    int zone_idx = 0;
    vector<bool> valid_gpus(available_gpu_types.size(), false);
    for (auto it : max_num_gpus)
    {
        string zone = it.first;
        string region = extract_region_from_zone(zone);
        vector<int> count_per_gpu_type(available_gpu_types.size());

        if (given_resources.find(region) == given_resources.end())
            given_resources[region] = {};

        int i = 0;
        string tmp_hash = zone;
        bool found_gpus = false;
        for (auto gpu_type : available_gpu_types)
        {
            int gpu_count = 0;
            // if (max_num_gpus[zone].find(gpu_type) != max_num_gpus[zone].end())
            // {
            // }
            gpu_count = max_num_gpus[zone][gpu_type];
            if (gpu_count > 0) { // if max_num_gpus in zone are 0 we don't want that zone
                found_gpus = true;
                valid_gpus[i] = true;
            }
            count_per_gpu_type[i++] = gpu_count;
            tmp_hash += "_" + gpu_type + "_" + to_string(gpu_count);
        }
        if (found_gpus)
        {
            given_resources[region].push_back(make_pair(zone, count_per_gpu_type));
            hash_string += tmp_hash;
        }
        zone_to_id[zone] = zone_idx;
        id_to_zone.push_back(zone);
        zone_idx++;
    }

    int num_valid_gpus = 0;
    for (auto x: valid_gpus) {
        if (x) {

            num_valid_gpus++;
        }
    }
    printf("NUM_VALID_GPUS is %d\n", num_valid_gpus);
    if (num_valid_gpus==1)
        homogeneous = true;
    else
        homogeneous = false;

    if (found_configs.find(hash_string) != found_configs.end())
    {
        return {found_configs[hash_string]};
    }

    // Sort the zones for each region based on total GPUs
    for (auto &[region, zones] : given_resources)
    {
        sort(zones.begin(), zones.end(), [](const auto &a, const auto &b)
             {
                 int total_gpus_a = accumulate(a.second.begin(), a.second.end(), 0);
                 int total_gpus_b = accumulate(b.second.begin(), b.second.end(), 0);
                 return total_gpus_a > total_gpus_b; // Descending order
             });
    }

    cout << "hash_string is " << hash_string << endl;
    for (auto it : given_resources)
    {
        cout << "region: " << it.first << endl;
        for (auto zt : it.second)
        {
            cout << "zone: " << zt.first << endl;
            for (auto gt : zt.second)
            {
                cout << gt << ",";
            }
            cout << endl;
        }
    }

    // get plans!
    auto start = high_resolution_clock::now();
    get_plans_num_gpus_dp(given_resources);
    //get_plans_no_heuristics(given_resources);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "--------------- get_plans_num_gpus_dp took " << duration.count() << " us" << endl;
    cout << "--------------- dp_solve_total is " << dp_solve_total << " us" << endl;

    vector<struct Config> new_configs = {};
    for (auto config : all_configs)
    {
        if (objective_str == "throughput")
        {
            if (max_budget > 0.0 && (config.cost > max_budget))
                continue;
        }
        else
        {
            if ((max_iter_time > 0) && (config.Tpp > max_iter_time))
                continue;
        }

        auto config_zones = config.get_zones();
        // check validity
        unordered_map<string, unordered_map<string, int>> used_gpus;
        auto configs_per_stage = config.get_config_per_stage();
        for (auto stage_config : configs_per_stage)
        {
            for (auto tp_config : stage_config)
            {
                auto gpu_type = available_gpu_types[get<0>(tp_config)];
                auto zone = config_zones[get<2>(tp_config)];
                if (used_gpus.find(zone) == used_gpus.end())
                {
                    used_gpus[zone] = {};
                }
                if (used_gpus[zone].find(gpu_type) == used_gpus[zone].end())
                {
                    used_gpus[zone][gpu_type] = 0;
                }
                used_gpus[zone][gpu_type] += get<1>(tp_config);
            }
        }

        bool invalid = false;
        for (auto zone_info : used_gpus)
        {
            auto zone = zone_info.first;
            if (max_num_gpus.find(zone) == max_num_gpus.end())
            {
                invalid = true;
                break;
            }
            for (auto gpu_info : used_gpus[zone])
            {
                auto gpu = gpu_info.first;
                if (
                    (max_num_gpus[zone].find(gpu) == max_num_gpus[zone].end()) ||
                    (used_gpus[zone][gpu] > max_num_gpus[zone][gpu])
                )
                {
                    invalid = true;
                    break;
                }
            }
        }

        if (invalid)
            continue;

        new_configs.push_back(config);
    }

    if (objective_str == "throughput")
    {

        vector<struct Config> new_configs = {};
        if (max_budget > 0.0)
        {
            for (auto config : all_configs)
            {
                if (config.cost < max_budget)
                    new_configs.push_back(config);
            }
        }
        else
        {
            new_configs = all_configs;
        }

        sort(new_configs.begin(), new_configs.end(), compareByThroughput);
        //std::reverse(new_configs.begin(), new_configs.end());
        if (!new_configs.empty())
            found_configs[hash_string] = new_configs[0];


        return new_configs;
    }
    else
    {
        vector<struct Config> new_configs;
        if (max_iter_time > 0)
        {
            for (auto config : all_configs)
            {
                if (config.Tpp < max_iter_time)
                    new_configs.push_back(config);
            }
        }
        else
        {
            new_configs = all_configs;
        }
        sort(new_configs.begin(), new_configs.end(), compareByCost);
        if (!new_configs.empty())
            found_configs[hash_string] = new_configs[0];

        return new_configs;
    }

    return {};
}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(vector<int>);
PYBIND11_MAKE_OPAQUE(vector<vector<int>>);
// PYBIND11_MAKE_OPAQUE(vector<string>);

PYBIND11_MODULE(libplanner, m)
{
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def("get_stages", &Config::get_stages)
        .def("get_mbs", &Config::get_mbs)
        .def("get_Tpp", &Config::get_Tpp)
        .def("get_cost", &Config::get_cost)
        .def("get_comm_cost", &Config::get_comm_cost)
        .def("get_comp_cost", &Config::get_comp_cost)
        .def("get_config_per_stage", &Config::get_config_per_stage)
        .def("get_zones", &Config::get_zones);

    py::bind_vector<vector<int>>(m, "VectorInt");
    py::bind_vector<vector<vector<int>>>(m, "VectorVectorInt");

    py::class_<SailorPlanner>(m, "SailorPlanner")
        .def(py::init<const char *, const char *, const char *, const char *, const char *, bool, bool, const char *, const char *>())
        .def("get_sorted_plans", &SailorPlanner::get_sorted_plans);
}