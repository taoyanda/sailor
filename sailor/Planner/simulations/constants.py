COST_SAME_AZ = 0
COST_2_AZ = 0.01
COST_2_REGIONS_US = 0.02
COST_2_REGIONS_EU = 0.02
COST_US_EU = 0.05

V100_US_CENTRAL1_A = 2
V100_US_CENTRAL1_B = 2
V100_US_WEST1_B = 2
V100_EU_WEST4_A = 2.08

A100_US_CENTRAL1_A = 3.67
A100_US_CENTRAL1_B = 3.67
A100_US_EAST1_B = 3.67
A100_US_WEST1_B = 3.67
A100_EU_WEST4_A = 3.75

GPU_PRICES = {
    'A100-40': {
        'europe-west4-a': 3.673385,
        'us-east1-b': 3.673385,
        'us-central1-a': 3.673385,
        'us-central1-b': 3.673385,
        'us-west1-b': 3.673385
    },
    'V100-16': {
        'europe-west4-a': 2.859998,
        'us-west1-b': 2.859998,
        'us-central1-b': 2.859998,
        'us-central1-a': 2.859998
    },
    'T4-16': {
        'us-central1-a': 1,
    },
    'GH-96': {
        'europe-west4-a': 11.06,
        'us-west1-b': 11.06,
        'us-central1-b': 11.06,
        'us-central1-a': 11.06
    }

}

GPU_MEMORY_GB = {
    'V100-16': 16,
    'T4-16': 16,
    'A100-40': 40,
    'A100-80': 80,
    'GH-96': 96,
    'Titan-RTX': 24,
    'RTX-2080': 11,
    'RTX-3090': 24
}
