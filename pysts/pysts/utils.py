import time
import os
import datetime
import shutil

import cv2

from sts.scripts.helper import read_json


class Rate:
    def __init__(self, frequency: float) -> None:
        """
        Maintains a constant control rate for the POMDP loop. Replacement for rate object in ROS.

        :param frequency: Polling frequency, in Hz.
        """
        self.period, self.last = 1.0 / frequency, time.time()

    def sleep(self) -> None:
        current_delta = time.time() - self.last
        sleep_time = max(0.0, self.period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self.last = time.time()

def sas_list_to_vids(ep_i, sas_list, keys, top_dir, vid_subdir, rate=10):
    img_dict = sas_list_to_img_dict(sas_list, keys)
    for k in keys:
        vid_subdir_w_key = os.path.join(vid_subdir, k)
        os.makedirs(os.path.join(top_dir, vid_subdir_w_key), exist_ok=True)
        img_list_to_vid(img_dict[k], rate=rate, top_dir=top_dir,
            vid_dir_name=vid_subdir_w_key, vid_name=f"ep{ep_i}", save_img_list=True)

def sas_list_to_img_dict(sas_list, keys):
    data = {}
    for k in keys:
        data[k] = []
    for sas in sas_list:
        for k in keys:
            state = sas[0]
            if k == 'wrist_rgb' and not 'wrist_rgb' in state.keys() and 'wrist_rgbd' in state.keys():
                data[k].append(state['wrist_rgbd'][:, :, :3].astype('uint8'))
            elif k == 'wrist_depth' and not 'wrist_depth' in state.keys() and 'wrist_rgbd' in state.keys():
                rgb_of_depth = cv2.cvtColor(state['wrist_rgbd'][:, :, 3], cv2.COLOR_GRAY2BGR)
                rgb_of_d_uint8 = cv2.convertScaleAbs(rgb_of_depth)
                data[k].append(rgb_of_d_uint8)
            else:
                data[k].append(state[k])

    return data

def img_list_to_vid(imgs, rate, top_dir='.', vid_dir_name='recordings', vid_name=None, save_img_list=False):
    if vid_name is None:
        vid_name = datetime.now().strftime("%m-%d-%y-%H_%M_%S")

    vid_dir_name = os.path.join(top_dir, vid_dir_name)
    img_dir_name = os.path.join(vid_dir_name, f"{vid_name}_imgs")
    os.makedirs(vid_dir_name, exist_ok=True)
    os.makedirs(img_dir_name, exist_ok=True)
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(img_dir_name, str(i).zfill(5) + ".png"), img)

    os.system(f"/usr/bin/ffmpeg -r {rate} -y -i {img_dir_name}/%05d.png -pix_fmt yuv420p "\
              f"{vid_dir_name}/{vid_name}.mp4 >/dev/null 2>&1")

    if not save_img_list:
        shutil.rmtree(img_dir_name)

    print(f"Generated video at {vid_dir_name}/{vid_name}.mp4.")


def eul_rot_to_mat(eul_rxyz):
    from transform_utils.pose_transforms import PoseTransformer
    pt = PoseTransformer(pose=[0, 0, 0, *eul_rxyz], rotation_representation='euler', axes='rxyz')
    ee_to_sts_rot_mat = pt.get_matrix()[:3, :3]
    return ee_to_sts_rot_mat


def eul_rot_json_to_mat(config_dir):
    ee_to_sts_eul_rxyz = read_json(os.path.join(config_dir, "ft_adapted_replay.json"))["ee_to_sts_eul_rxyz"]
    return eul_rot_to_mat(ee_to_sts_eul_rxyz)