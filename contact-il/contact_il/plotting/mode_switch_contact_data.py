import os
import json
import numpy as np
import pandas as pd


MAIN_DIR = os.environ['CIL_DATA_DIR']
MAIN_PLOT_DIR = os.path.join(MAIN_DIR, 'plots')

TASK_PLOT_NAMES = [
    'Flat Handle',
    'Flat Handle Close',
    'Glass Knob',
    'Glass Knob Close'
]

DEMO_DATA_DICT = {
    'PandaTopBlackFlatHandleNewPos': {
        'ds_substr': 'apr20_new_sensor',
        'raw_contact_ts': [12, 11, 20, 14, 15, 13, 12, 24, 12, 19, 14, 13, 15, 15, 14, 15, 14, 11, 13, 10],
        'contact_ts':     [15, 13, 21, 15, 18, 13, 13, 26, 15, 22, 16, 15, 15, 16, 14, 16, 15, 12, 14, 11],
    },
    'PandaTopBlackFlatHandleNewPosClose': {
        'ds_substr': 'apr20_new_sensor',
        'raw_contact_ts': [10,  8, 13, 13, 10, 13, 10, 10, 12, 10, 11, 10, 10, 11, 10, 10, 10,  8, 11,  8],
        'contact_ts':     [10,  8, 13, 13, 11, 14, 10, 11, 12, 11, 11, 11, 10, 11,  9, 10, 10,  8, 11,  9],
    },
    'PandaBottomGlassOrb': {
        'ds_substr': 'apr22_random_fix',
        'raw_contact_ts': [ 8,  9,  8,  9, 12,  9,  9,  7,  8, 12,  9, 10, 11,  9, 15, 12,  9,  8,  9, 11],
        'contact_ts':     [ 9, 10,  9, 10, 13, 10, 10,  8, 10, 13, 10, 11, 12, 10, 16, 12, 10,  9, 10, 12],
    },
    'PandaBottomGlassOrbClose': {
        'ds_substr': 'apr22_better_fixed',
        'raw_contact_ts': [16, 14, 14, 14, 17, 16, 16, 13, 12, 20, 18, 21, 13, 18, 16, 16, 16, 16, 16, 14],
        'contact_ts':     [15, 15, 14, 16, 18, 16, 16, 13, 12, 20, 18, 15, 13, 25, 15, 14, 16, 21, 15, 13],
    },
}

TEST_DATA_DICT = {
    'PandaTopBlackFlatHandleNewPos': {
        'ds_substr': 'apr20_new_sensor/20_eps/wrist_rgb-sts_raw_image-pose-prev_pose/apr20_/',
        'postseed_str': '10_test_eps/apr20',
        'contact_ts': [
            [14, 12, 15, 12, 13, 17, 14, 14, 15, 12],
            [14, 12, 16, 14, 13, 20, 14, 14, 15, 12],
            [14, 11, 16, 14, 12, 23, 14, 13, 17, 13],
        ],
    },
    'PandaTopBlackFlatHandleNewPosClose': {
        'ds_substr': 'apr20_new_sensor/20_eps/wrist_rgb-sts_raw_image-pose-prev_pose/apr20_/',
        'postseed_str': '10_test_eps/apr20',
        # 'contact_ts': [
        #     [10,  9, 10, 12,  9, 12, 11,  9,  9,  9],
        #     [ 9,  9, 11, 13,  9, 15, 14, 14, 11, 15],
        #     [10,  8, 11, 12,  9, 14, 13, 13, 12, 12],
        # ],
        'contact_ts': [
            [10,  9, 10, 12,  8, 12, 11,  9,  9,  9],
            [ 9,  9, 11, 13,  9, 15, 13, 13, 11, 15],
            [10,  8, 11, 12,  9, 14, 12, 12, 12, 11],
        ],
    },
    'PandaBottomGlassOrb': {
        'ds_substr': 'apr22_random_fix/20_eps/wrist_rgb-sts_raw_image-pose-prev_pose/apr20_/',
        'postseed_str': '10_test_eps/apr20',
        'contact_ts': [
            [ 9,  9, 11, 10,  9, 10,  9, 10, 13, 10],
            [ 9,  8,  9, 10,  9,  9,  9, 10,  9, 10],
            [ 9,  8,  9, 10,  8, 10, 12,  9, 10, 10],
        ],
    },
    'PandaBottomGlassOrbClose': {
        'ds_substr': 'apr22_better_fixed/20_eps/wrist_rgb-sts_raw_image-pose-prev_pose/apr20_/',
        'postseed_str': '10_test_eps/apr20',
        'contact_ts': [
            [14, 13, 15, 14, 12, 16, 14, 13, 15, 15],
            [14, 13, 15, 15, 13, 16, 16, 22, 15, 16],
            [14, 13, 14, 14, 12, 16, 14, 15, 15, 15],
        ],
    },
}