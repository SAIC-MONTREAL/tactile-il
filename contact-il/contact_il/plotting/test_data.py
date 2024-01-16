import os
import json
import numpy as np
import pandas as pd


######## Options ########
MAIN_DIR = os.environ['CIL_DATA_DIR']
MAIN_PLOT_DIR = os.path.join(MAIN_DIR, 'plots')

# all
orb_close_ds_substr = "apr22_random_fix"
orb_close_ds05_substr = "apr22_better_fixed"

# main performance
main_ds_substr = "apr20_new_sensor"
main_train_substr = "apr20_"
main_test_substr = "apr20"

# obs variant performance
obs_ds_substr = "apr20_new_sensor"
obs_train_substr = "apr20_"
obs_test_substr = "apr20"

# data variant performance
data_ds_substr = "apr20_new_sensor"
data_flat_ds0_substr = "apr22_redo"
data_train_substr = "apr20_"
data_test_substr = "apr20"

test_ep_amt = 10

#########################


TASKS = [
    'PandaTopBlackFlatHandleNewPos',
    'PandaTopBlackFlatHandleNewPosClose',
    'PandaBottomGlassOrb',
    'PandaBottomGlassOrbClose',
]

TASK_PLOT_NAMES = [
    'Flat Handle',
    'Flat Handle Close',
    'Glass Knob',
    'Glass Knob Close'
]

def get_dataset_substr(idx, sub_name):
    dataset_substrs = [
        f"/{sub_name}",
        f"TactileOnly/{sub_name}",
        f"VisualOnly/{sub_name}",
        f"/{sub_name}_simple_contact",
        f"/{sub_name}_no_fa",
        f"VisualOnly/{sub_name}_no_fa",
        f"TactileOnly/{sub_name}_no_fa"
    ]
    return dataset_substrs[int(idx)]

EXTRA_DATA_AMOUNTS = ['15', '10', '5']

OBS_LIST_STRS = [
    'wrist_rgb-sts_raw_image-pose-prev_pose',
    'wrist_rgb-pose-prev_pose',
    'sts_raw_image-pose-prev_pose',
    'wrist_rgb',
    'sts_raw_image',
    'pose-prev_pose',
    'wrist_rgb-sts_raw_image',
]

DATASET_NAMES = [
    'Mode Switching, Force Matching',
    'Tactile Only, Force Matching',
    'Visual Only, Force Matching',
    'Mode Switching, Binary Force Matching',
    'Mode Switching, No Force Matching',
    'Visual Only, No Force Matching',
    'Tactile Only, No Force Matching',
]

DATASET_NAMES_SHORT = [
    'MS-FM',
    'TO-FM',
    'VO-FM',
    'MS-BFM',
    'MS',
    'VO',
    'TO',
]

OBS_NAMES = [
    'Wrist, STS, Rel Pose',
    'Wrist, Rel Pose',
    'STS, Rel Pose',
    'Wrist',
    'STS',
    'Rel Pose',
    'Wrist, STS',
]

OBS_NAMES_SHORT = [
    'WSR',
    'WR',
    'SR',
    'W',
    'S',
    'R',
    'WS',
]

def get_name_from_idxes(idxes, short=False):
    # two idxes, first is dataset, second is included observations
    if type(idxes) == str:
        idxes = idxes.split(' ')

    fixed = []
    for i in idxes:
        fixed.append(int(i))
    idxes = fixed

    if short:
        return f"{DATASET_NAMES_SHORT[idxes[0]]}, {OBS_NAMES_SHORT[idxes[1]]}"
    else:
        return f"{DATASET_NAMES[idxes[0]]}, {OBS_NAMES[idxes[1]]}"


################ Performance ################

COLOR_MODE_MAP={
    "0 0": 0,
    "1 0": 2,
    "2 0": 4,
    "0 2": 6,
    "1 2": 8,
    "2 2": 10,
    "4 0": 12,
    "5 1": 14,
    "2 1": 16,
    "2 3": 18,
    "0 4": 1,
    "2 5": 3,
    "0 6": 5,
}

OBS_ONLY_MAP={
    'WSR': 'Wrist, STS, Rel Pose',
    'WR': 'Wrist, Rel Pose',
    'SR': 'STS, Rel Pose',
    'W': 'Wrist',
    'S': 'STS',
    'R': 'Rel Pose',
    'WS': 'Wrist, STS',
}

MAIN_VARIANT_IDX_COMBOS=(
    "0 0",
    "1 0",
    "2 0",
    "0 2",
    "1 2",
    "2 2",
    "4 0",
    "5 1"
)

OBS_VARIANT_IDX_COMBOS=(
    "0 0",
    "2 1",
    "0 2",
    "2 3",
    "0 4",
    "2 5",
    "0 6",
)

DATA_VARIANT_IDX_COMBOS=(
    "0 0",
    "0 2",
    "2 1",
    "5 1",
)

SEEDS = [1, 2, 3]

MAIN_VARIANT_NAMES_SHORT = [get_name_from_idxes(idxes, short=True) for idxes in MAIN_VARIANT_IDX_COMBOS]
DATA_VARIANT_NAMES_SHORT = [get_name_from_idxes(idxes, short=True) for idxes in DATA_VARIANT_IDX_COMBOS]

DF_COLUMNS = ['task', 'variant', 'amount', 'suc_mean', 'suc_std']

def get_main_perf_data(fig_name, allow_data_missing=False):
    if fig_name == "main":
        idx_combos = MAIN_VARIANT_IDX_COMBOS
        train_substr = main_train_substr
        test_substr = main_test_substr
        data_amounts = [20]
        assert len(data_amounts) == 1
        df_columns = DF_COLUMNS
    elif fig_name == "obs_variant":
        idx_combos = OBS_VARIANT_IDX_COMBOS
        train_substr = obs_train_substr
        test_substr = obs_test_substr
        data_amounts = [20]
        assert len(data_amounts) == 1
        df_columns = DF_COLUMNS
    elif fig_name == 'data_variant':
        idx_combos = DATA_VARIANT_IDX_COMBOS
        train_substr = data_train_substr
        test_substr = data_test_substr
        data_amounts = [5, 10, 15, 20]
        df_columns = DF_COLUMNS
    else:
        raise NotImplementedError()

    variant_names_short = [get_name_from_idxes(idxes, short=True) for idxes in idx_combos]

    num_missing = 0
    perf_raw = dict()
    perf_data_only = dict()
    perf_data_by_variant = dict.fromkeys(variant_names_short)
    perf_data_for_df = dict.fromkeys(df_columns)
    for k in perf_data_for_df:
        perf_data_for_df[k] = []
    for va in variant_names_short:
        perf_data_by_variant[va] = dict()
        perf_data_by_variant[va]['suc_means'] = []
        perf_data_by_variant[va]['suc_stds'] = []
        perf_data_by_variant[va]['suc_all'] = []

    for ta, ta_name in zip(TASKS, TASK_PLOT_NAMES):
        perf_raw[ta] = dict()
        perf_data_only[ta] = dict()
        task_dict = perf_raw[ta]
        data_only_task_dict = perf_data_only[ta]
        data_only_task_dict['suc_means'] = []
        data_only_task_dict['suc_stds'] = []
        data_only_task_dict['suc_stds'] = []
        for va, va_idxes_str in zip(variant_names_short, idx_combos):

            va_idxes = va_idxes_str.split(' ')
            task_dict[va] = dict()
            va_dict = task_dict[va]

            for amount in data_amounts:
                seed_suc_rates = []

                for se in SEEDS:
                    va_dict[se] = dict()

                    if "BlackFlat" in ta:
                        if int(va_idxes[0]) == 0 and amount in [5, 10, 15] or int(va_idxes[1]) == 6:
                            ds_substr = data_flat_ds0_substr
                        else:
                            ds_substr = main_ds_substr
                    else:
                        if int(va_idxes[0]) in [0, 5] and "Close" in ta:
                            ds_substr = orb_close_ds05_substr
                        else:
                            ds_substr = orb_close_ds_substr

                    va_dict[se]['loc'] = os.path.join(
                        MAIN_DIR, 'tests', ta + get_dataset_substr(int(va_idxes[0]), ds_substr), f"{amount}_eps",
                        OBS_LIST_STRS[int(va_idxes[1])], train_substr, str(se), f"{test_ep_amt}_test_eps",
                        test_substr)

                    # load data
                    try:
                        with open(os.path.join(va_dict[se]['loc'], 'performance.json'), 'r') as f:
                            perf_data = json.load(f)
                        va_dict[se]['suc_rate'] = perf_data['success_mean']
                    except FileNotFoundError as e:
                        if not allow_data_missing:
                            raise FileNotFoundError(e)
                        else:
                            num_missing += 1
                            print(f"Warning: no data found at {perf_raw[ta][va][se]['loc']}..using -0.1.")
                            va_dict[se]['suc_rate'] = -0.1

                    seed_suc_rates.append(va_dict[se]['suc_rate'])
                seed_suc_rates = np.array(seed_suc_rates)
                mean = seed_suc_rates.mean()
                std = seed_suc_rates.std()
                data_only_task_dict['suc_means'].append(mean)
                data_only_task_dict['suc_stds'].append(std)

                perf_data_by_variant[va]['suc_means'].append(mean)
                perf_data_by_variant[va]['suc_stds'].append(std)
                perf_data_by_variant[va]['suc_all'].append(seed_suc_rates)

                perf_data_for_df['task'].append(ta_name)
                perf_data_for_df['variant'].append(va)
                perf_data_for_df['amount'].append(amount)
                perf_data_for_df['suc_mean'].append(mean)
                perf_data_for_df['suc_std'].append(std)

    perf_data_df = pd.DataFrame(perf_data_for_df)

    for va, va_idxes_str in zip(variant_names_short, idx_combos):
        perf_data_by_variant[va]['suc_all'] = np.concatenate(perf_data_by_variant[va]['suc_all'])
        suc_all = perf_data_by_variant[va]['suc_all']
        perf_data_by_variant[va]['suc_all_mean'] = suc_all.mean()
        perf_data_by_variant[va]['suc_all_std'] = suc_all.std()

    if num_missing > 0:
        print(f"Total num missing: {num_missing}")

    return perf_data_df, perf_raw, perf_data_only, perf_data_by_variant
