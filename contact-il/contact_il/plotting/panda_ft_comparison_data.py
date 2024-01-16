import os
import json
import numpy as np
import pandas as pd


######## Options ########
MAIN_DIR = os.environ['CIL_DATA_DIR']
MAIN_PLOT_DIR = os.path.join(MAIN_DIR, 'plots')

TOY_TASK_DATA_DICT = {
    'PandaCabinetOneFinger6DOFROS': {
        'ds_substr': 'circles',
        'ep': 0,  # moves in all 3 directions to a degree, while 1 is only in x + z (y is normal dir)
        'contact_ts': None,
        'max_ts': 30,
        'traj_xlim': None,
        'live_traj_xlim': [-0.06, .16],
        'live_traj_ylim': [-0.12, 0.12],
        'panda_force_scale': 15,
        'sts_force_scale': 15,
        'traj_arrow_inds': [10, 20],
        'traj_ts_labels': [0, 29],
        'traj_ts_labels_xyadds': [[0.04, 0.04], [-0.09, -0.07]],
        'arrow_arcs': [0.2, -0.2]
    },
    'PandaBottomGlassOrbROSTactileOnly': {
        'ds_substr': 'push',
        'ep': 0,
        'contact_ts': (9, 26),  # 25 is the last true contact
        'max_ts': 35,
        'traj_xlim': [-0.05, 0.05],
        'live_traj_xlim': [-0.05, 0.05],
        'live_traj_ylim': [-0.01, 0.075],
        'panda_force_scale': 12,
        'sts_force_scale': 13,
        'traj_arrow_inds': [4, 30],
        'traj_ts_labels': [0, 9, 26, 34],
        'traj_ts_labels_xyadds': [[-0.05, 0.01], [-0.05, -0.01], [0.02, -0.01], [0.01, 0.01]],
        'arrow_arcs': [0.2, -0.2, 0.2, -0.2]
    }
}

TASK_DATA_DICT = {
    'PandaTopBlackFlatHandleROSTactileOnly': {
        'ds_substr': 'demos',
        'ep': 1,
    },
    'PandaTopBlackFlatHandleROSCloseTactileOnly': {
        'ds_substr': 'demos',
        'ep': 0,
    },
    'PandaBottomGlassOrbROSTactileOnly': {
        'ds_substr': 'demos',
        'ep': 0,
    },
    'PandaBottomGlassOrbROSCloseTactileOnly': {
        'ds_substr': 'demos',
        'ep': 0,
    },
}


#########################

TASK_PLOT_NAMES = [
    'Flat Handle',
    'Flat Handle Close',
    'Glass Knob',
    'Glass Knob Close'
]