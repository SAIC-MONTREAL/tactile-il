import cv2
import numpy as np
import os

from sts.scripts.helper import read_json

class STSSlip:
    """Process slip detection on the STS"""
    def __init__(self, config):
        self._config = config
        self._slip_counter = 0

    def detect(self, Vx, Vy, MVx, MVy, logger=None):
        Dx = (Vx - MVx)[0]
        Dy = (Vy - MVy)[0]
        slip_flow = np.zeros((Dx.shape[0], Dx.shape[1], 2))
        slip_flow[:,:,0] = Dx
        slip_flow[:,:,1] = Dy

        total_flow = np.linalg.norm(slip_flow, axis=2)
        flow_indices = np.where(total_flow > 2)[0]
        slip_total = np.nan_to_num(np.average(total_flow[flow_indices]))
        if logger:
            logger.info(f'slip total {slip_total}')
        if slip_total > self._config['thresh']:
            self._slip_counter += 1
        else:
            self._slip_counter = 0

        if self._slip_counter > self._config['min_count']:
            is_slip = True
        else:
            is_slip = False

        return is_slip, Dx, Dy

    def get_slip_flow(self, img, Dx, Dy, type='quiver'):
        if type=='quiver':
            from sts.scripts.flow import STSOpticalFlow
            return STSOpticalFlow.plot_quiver(img, Dx, Dy, 
                    spacing=15, 
                    margin=0, 
                    scaling=2, 
                    color=(0, 255, 255), 
                    centroids=None, 
                    thresh=5)
        else:
            raise ValueError(f"Unexpected slip flow type argument: {type}")    
