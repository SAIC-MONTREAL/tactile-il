import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse
# from contact_panda_envs.envs.cabinet.cabinet_contact import PandaCabinetOneFinger2DOF, PandaTopGlassOrbNoSTSSwitch
from contact_panda_envs.envs import *


parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, help="sts config dir")
parser.add_argument('--sts_config_dir', type=str, default=os.environ['STS_CONFIG'], help="sts config dir")
parser.add_argument('--sim', action='store_true', help="whether the robot is in sim")
parser.add_argument('--sts_sim_vid', type=str, default="", help="sts vid file for sim if desired")

args = parser.parse_args()


env = globals()[args.env](
    sts_config_dir=args.sts_config_dir,
    sim_override=args.sim,
    sts_source_vid=args.sts_sim_vid
)

# env = PandaCabinetOneFinger2DOF(
#     sts_config_dir=os.environ["STS_CONFIG"],
#     sim_override=True,
#     sts_source_vid=os.environ['STS_SIM_VID'])
obs = env.reset()

obss = []
obss.append(obs)

# for ts in range(100):
for ts in range(10):
    cmd = np.zeros(env.action_space.shape)
    if ts < 50:
        cmd[1] = .005
    else:
        cmd[1] = -.005
    next_obs, rew, term, trunc, info = env.step(cmd)
    obss.append(next_obs)
    print(f"timestep: {ts}")

import ipdb; ipdb.set_trace()
