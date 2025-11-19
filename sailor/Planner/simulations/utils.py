import pandas as pd


class Model:
    def __init__(self, model_size, activation_params, params_per_layer) -> None:
        self.model_size_gb = model_size/1e9
        self.activation_params = activation_params/1e9
        self.number_of_parameters = model_size//4
        self.params_per_layer = params_per_layer


models = {
    'VGG-19': Model(576 * 1e6, 0, []),
    'ConvNext-Large': Model(788 * 1e6, 0, []),
    'VIT-H-14': Model(2528 * 1e6, 0, []),
    'BLOOM-7': Model(28000*1e6, 1, []),  # TODO: fix activations
    'OPT-350': Model(1400*1e6, 2097152, [28362752] + [12596224] * 24 + [26263552]),
    'OPT-1.3': Model(5290*1e6, 4194304, []),
    'OPT-6.7': Model(26800*1e6, 8388608, []),
    'GPT.2_6B': Model(2780451840*4, 5242880, [134021120]+[78676480]*32+[128783360]),
    'GPT.6_7B': Model(6864642048*4, 8388608, [214433792]+[201379840]*32+[206053376]),
    'GPT.13B': Model(13729285120*4, 16777216, [429061888]+[402759168]*40+[412719360]),
    'OPT-30': Model(120000*1e6, 14680064, [375044096] + [616655872] * 48 + [360364032]),
    'LLAMA-3-8': Model(32000*1e6, 8392704, [525336576] + [218112000] * 32 + [525340672]),
    'GPT-Neo-2.7': Model(10800*1e6, 5242880, [133900800] + [78676480] * 32 + [128663040])
}


def parse_trace(trace_file):
    # given a trace file in the form of {timestamp, gputype-count_zone}
    # return a trace of {duration (in sec), num_gpus}

    trace = pd.read_csv(trace_file, index_col=0)
    times = list(trace['Timestamp'])
    trace.pop('Timestamp')

    diff_gpus = list(trace.columns.values)
    gpu_configs = []
    for _, row in trace.iterrows():
        config = {}
        for gpu_name in diff_gpus:
            config[gpu_name] = int(row[gpu_name])
        gpu_configs.append(config)

    current_time = times[0]
    current_num_gpus_list = gpu_configs[0]

    new_trace = []

    for timestamp, gpu_count_list in zip(times, gpu_configs):
        if gpu_count_list == current_num_gpus_list:
            continue
        # change in trace found
        new_trace.append([timestamp-current_time, current_num_gpus_list])

        current_time = timestamp
        current_num_gpus_list = gpu_count_list

    new_trace.append([times[-1]-current_time, current_num_gpus_list])

    sum_gpus_first = 0
    for _, gpu_count in new_trace[0][1].items():
        sum_gpus_first += gpu_count

    if sum_gpus_first == 0:
        new_trace.pop(0)

    return new_trace