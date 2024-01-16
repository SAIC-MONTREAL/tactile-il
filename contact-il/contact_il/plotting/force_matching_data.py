import os
import json
import numpy as np
import pandas as pd


######## Options ########
MAIN_DIR = os.environ['CIL_DATA_DIR']
MAIN_PLOT_DIR = os.path.join(MAIN_DIR, 'plots')

# all
TASK_DATA_DICT = {
    'PandaTopBlackFlatHandleNewPosTactileOnly': {
        'ds_substr': 'apr20_new_sensor',
        'no_fa_ds_substr': 'apr20_new_sensor_no_fa',
        'ep': 3,
        'rep_ts': 18,
        'img_tl_br_corners': [(143, 11), (213, 92)]  # 70 x 81
    },
    'PandaTopBlackFlatHandleNewPosCloseTactileOnly': {
        'ds_substr': 'apr20_new_sensor',
        'no_fa_ds_substr': 'apr20_new_sensor_no_fa',
        'ep': 0,
        'rep_ts': 15,
        # 'rep_ts': 13,
        'img_tl_br_corners': [(131, 15), (201, 96)]
    },
    'PandaBottomGlassOrbTactileOnly': {
        'ds_substr': 'apr22_random_fix',
        'no_fa_ds_substr': 'apr22_random_fix_no_fa',
        'ep': 3,
        'rep_ts': 20,
        'live_xlim': [-0.375, 0.02],
        'live_ylim': [-0.01, 0.375],
        'img_tl_br_corners': [(116, 18), (186, 99)]  # 70 x 81 but cuts off a bit of bottom
    },
    'PandaBottomGlassOrbCloseTactileOnly': {
        'ds_substr': 'apr22_random_fix',
        'no_fa_ds_substr': 'apr22_random_fix_no_fa',
        'ep': 3,
        'rep_ts': 37,
        'live_xlim': [-0.02, 0.375],
        'live_ylim': [-0.08, 0.28],
        'img_tl_br_corners': [(143, 32), (213, 113)]
    },
}



#########################

TRAJ_PLOT_NAMES = [
    'Demonstrator',
    'Force-Matched',
    'Not Force-Matched'
]

TRAJ_PLOT_NAMES_SHORT = [
    'Demonstrator',
    'FM Replay',
    'No FM Replay'
]

TRAJ_PLOT_NAMES_VERY_SHORT = [
    'Demo',
    'FM',
    'No FM'
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