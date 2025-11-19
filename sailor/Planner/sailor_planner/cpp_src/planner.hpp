#include <stdio.h>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <map>
#include <fstream>
#include <tuple>
#include <sstream>
#include <iomanip>

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pybind11.h>

#include <jsoncpp/json/json.h>

#include "planner_utils.hpp"
#include "utils/read_json.hpp"

#include <cassert>
#include <set>

struct ParallelismConfig {
    vector<struct StageConfig> config_per_stage;
    double Tpp;
    double comm_cost;
    double comp_cost;
    struct PipelineInfo pp_info;

    ParallelismConfig():
        config_per_stage({}), Tpp(0.0), comm_cost(0.0), comp_cost(0.0), pp_info({}) {};

    ParallelismConfig(
        vector<struct StageConfig> config_per_stage_arg,
        double Tpp_arg,
        double comm_cost_arg,
        double comp_cost_arg,
        struct PipelineInfo pp_info_arg
    ):

        config_per_stage(config_per_stage_arg),
        Tpp(Tpp_arg), comm_cost(comm_cost_arg), comp_cost(comp_cost_arg), pp_info(pp_info_arg) {};
};

struct Config {
    vector<vector<int>> stages;
    int mbs;
    double Tpp;
    double cost;
    double comm_iter_cost;
    double comp_iter_cost;
    vector<struct StageConfig> config_per_stage;
    vector<string> zones;

    Config():
        stages({}), mbs(0),  Tpp(0.0), cost(0.0), comm_iter_cost(0.0), comp_iter_cost(0.0), config_per_stage({}), zones({}) {};

    Config(
        vector<vector<int>> stages_arg,
        int mbs_arg,
        double Tpp_arg,
        double cost_arg,
        double comm_iter_cost_arg,
        double comp_iter_cost_arg,
        vector<struct StageConfig> config_per_stage_arg,
        vector<string> zones_args
    ):

    stages(stages_arg), mbs(mbs_arg), Tpp(Tpp_arg), cost(cost_arg),
    config_per_stage(config_per_stage_arg), zones(zones_args),
    comm_iter_cost(comm_iter_cost_arg), comp_iter_cost(comp_iter_cost_arg) {};

    vector<vector<int>> get_stages() {return stages;}
    int get_mbs() {return mbs;}
    double get_Tpp() {return Tpp;}
    double get_cost() {return cost;}
    double get_comm_cost() {return comm_iter_cost;}
    double get_comp_cost() {return comp_iter_cost;}
    vector<string> get_zones() {return zones;}

    vector<vector<tuple<int, int, int>>> get_config_per_stage() {
        vector<vector<tuple<int, int, int>>> all_configs_per_stage = {};
        for (auto config: config_per_stage) {
            all_configs_per_stage.push_back(config.get_dp_pairs());
        }
        return all_configs_per_stage;
    }
};


class SailorPlanner {
    public:
        SailorPlanner(
            const char* profile_file,
            const char* network_coeff_path,
            const char* training_config_path,
            const char* model_mem_info_file,
            const char *communication_cost_file,
            bool heterogeneous,
            bool fp16,
            const char* quotas_dict_path,
            const char* objective
        );

        SailorPlanner();
        vector<struct Config> get_sorted_plans(unordered_map<string, unordered_map<string, int>> max_num_gpus, float max_budget_arg, double min_throughput_arg);
        void get_plans_num_gpus_dp(unordered_map<string, vector<pair<string, vector<int>>>> &given_resources);
        void get_plans_no_heuristics(unordered_map<string, vector<pair<string, vector<int>>>> &given_resources);

        // only for experimentation!
        vector<vector<vector<int>>> get_tmp_degrees_for_all(int pp, int num_types);
        vector<vector<int>> get_pp_tp_combos(int pp, int gpu_id);


    private:
        bool heterogeneous = true;
        bool homogeneous = false;
        bool fp16;
        string gpu_type_str;
        string zone_str;
        string objective_str;
        int float_size;

        int nquotas;
        int max_num_gpus=0;
        int max_mbs=0;
        vector<double> cost_per_gpu;

        unordered_map<string, double> COSTS = {
            {"A100-40", 3.74},
            {"V100-16", 0},
            {"A100-80", 0},
            {"V100-32", 0},
            {"T4", 0},
            {"GH-96", 11.06}
        };

        struct TrainingInfo* training_info;
        vector<struct ParallelismConfig> configs;
        vector<struct Config> all_configs = {};

        float max_budget = 0.0;
        double max_iter_time = 0.0;

        // vector of {'(mbs, tmp)': {'layer': [fwd, bwd, update]}}
        vector<map<pair<int, int>, map<int, vector<double>>>> profiles_all_gpus;

        // vector of {'zone': coeffs}
        NETWORK_COEFFS_TYPE network_coeffs_full;

        // map of {'zone': {'zone': comm cost pet GB} }
        unordered_map<string, unordered_map<string, double>> comm_cost;

        map<string, set<string>> zones_per_region;

        // map of {'gpu_type' : {'zone': quotas}}
        map<string, unordered_map<string, int>> quotas_dict;
        // map of {'region' : {'gpu_type' : 'quota'}}
        map<string, unordered_map<string, int>> quotas_per_region;
        // map of {'zone' : {'gpu_type_idx' : 'quota'}}
        map<string, unordered_map<string, int>> quotas_per_zone;

        vector<map<pair<int, int>, double>> bytes_per_stage;
        vector<map<pair<int, int>, double>> activation_per_stage;
        vector<map<pair<int,int>, size_t>> params_all_stages;

        vector<vector<vector<int>>> all_stages; // all_stages[pp]: stages with pipeline parallelism degree pp
        vector<vector<map<pair<int, int>, int>>> max_tmps_vector_per_gpu;
        vector<vector<int>> possible_tmps_per_gpu;
        vector<int> min_tmps_per_gpu;
        vector<vector<map<tuple<int, int, int>, int>>> min_tmps_vector_per_gpu;

        // ar_times of different stages for which this gpu is a bottleneck
        // key: GPU type
        // vector[i][j]: i represents machine size, j nr of peers
        // vector[i][j] is map of stages to ar_times
        map<pair<string, string>, vector<vector<vector<map<pair<int, int>, double>>>>> ar_times_bottleneck;

        vector<vector<int>> tmps_vector;
        vector<vector<int>> mbs_dp_vector;

        unordered_map<size_t, double> known_times;
        vector<string> available_gpu_types;
        int num_diff_gpu_types;
        int max_all_quotas=0;

        set<int> all_possible_mbs_set;
        map<string, struct Config> found_configs; // keep best solution for each config

        // mbs, pp, start_layer, data_par, resources, budget
        // TODO: as alternative
        vector<vector<vector<vector<map<string, map<string, struct ParallelismConfig>>>>>> dp;
        size_t extra_cost = 0.0;

        // optimizations
        vector<string> regions;
        unordered_map<string, vector<int>> region_gpu_count;
        unordered_map<string, vector<string>> zones_per_region_combo;
        unordered_map<string, unordered_map<string, double>> region_throughput;
        unordered_map<string, int> zone_to_id;
        vector<string> id_to_zone;
        size_t dp_solve_total = 0;

        void build_structs();
        void precompute_ar_times_bottleneck(string sender_zone, string receiver_zone);
        void regions_sorting(const unordered_map<string,vector<pair<string, vector<int>>>> &given_resources);

        struct ParallelismConfig* solve_dp(
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
            vector<vector<int>> &tmp_degrees
        );
        double get_cost(vector<struct StageConfig> &config_per_stage);


};

bool compareByThroughput(const Config& p1, const Config& p2) {
    return p1.Tpp < p2.Tpp;
}

bool compareByCost(const Config& p1, const Config& p2) {
    return p1.cost < p2.cost;
}

bool compareByValue(const Config& p1, const Config& p2) {
    double p1_value = (1/p1.Tpp)/p1.cost;
    double p2_value = (1/p2.Tpp)/p2.cost;
    return p1_value > p2_value;
}