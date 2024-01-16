import cv2
import os
import time
import argparse

from pysts.processing import STSProcessor


parser = argparse.ArgumentParser()
parser.add_argument('--initial_mode', type=str, default='tactile', help="initial mode of sensor")
parser.add_argument('--allow_both_modes', action='store_true', help="allow mode switching")
parser.add_argument('--config_dir', type=str, default=os.environ['STS_CONFIG'], help="config dir")
parser.add_argument('--source_vid', type=str, default='', help="video file with sts data, if simulated")
parser.add_argument('--resolution', default='', type=str, help="resolution to use, defaults to source")
parser.add_argument('--filter_markers', type=str, default='', help="average or kalman or empty string.")
parser.add_argument('--time_debug', action='store_true', help="print time debugging info.")
parser.add_argument('--test_mode_switch', action='store_true', help="test mode switches at interval.")
parser.add_argument('--view_force', action='store_true', help="Show force output.")

args = parser.parse_args()

mode_switch_type = "displacement" if not args.test_mode_switch else "none"

sts_p = STSProcessor(
    config_dir=args.config_dir,
    allow_both_modes=args.allow_both_modes,
    resolution=args.resolution,
    time_debug=args.time_debug,
    sensor_sim_vid=args.source_vid,
    filter_markers=args.filter_markers,
    mode_switch_opts={
        'initial_mode': args.initial_mode,
        'mode_switch_type': mode_switch_type,  # none (string), displacement, depth, contact, internal_ft
        'mode_switch_req_ts': 4,
        # 'tac_thresh': 0.5,
        # 'tac_thresh': 0.7,
        'tac_thresh': 1.5,
        'vis_thresh': 0.25,
        'tactile_mode_object_flow_channel': 'b',
        'mode_switch_in_thread': True,
    }
)

i = 0

displays=[
    'marker_dots',
    'marker_image',  # uses kalman filter if set on
    'raw_image',
    'marker_displacement',
    'depth_image',
    # 'depth_surface'
    # 'flow',
    # 'marker_flow'
]
if args.view_force:
    displays.extend(['force', 'avg_force'])

while True:
    start = time.time()
    img_dict = sts_p.get_processed_sts(
        modes={
            'raw_image',
            'depth_image',
            'flow',
            'marker_flow',
            'marker_dots',
            'marker_image',
            'marker_displacement',
            'force',
            'avg_force'
        },
        displays=tuple(displays)
    )

    if args.test_mode_switch:
        start_switch = time.time()
        if i % 50 == 0:
            if i % 100 == 0:
                sts_p.set_mode('tactile')
            else:
                sts_p.set_mode('visual')
            print(f"MODE SWITCH TIME: {time.time() - start_switch}")
    i += 1
