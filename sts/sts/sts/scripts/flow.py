import cv2
import numpy as np
from copy import deepcopy
from sts.scripts.helper import read_json

class STSOpticalFlow:
    """Process optical flow on the STS"""
    def __init__(self, config):
        self._config = config
        self._prev_img = None

    def detect(self, img, centroids=None):
        """Detect flow from previous image to this one. On frame 0, no flow will be reported"""
        if len(img.shape) == 3:
            rgb_img = img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # On first iteration, initialize previous image
        if self._prev_img is None:
            self._prev_img = img

        # Compute flow
        Vx, Vy = STSOpticalFlow.compute_flow(self._prev_img, img, self._config["window"])
        flow_img = self._get_flow_img(rgb_img, Vx, Vy, centroids=centroids)

        self._prev_img = img

        return flow_img, Vx, Vy

    def _get_flow_img(self, img, Vx, Vy, centroids=None):
        return STSOpticalFlow.plot_quiver(deepcopy(img), Vx[0], Vy[0], spacing=15, color=self._config["color"], centroids=centroids)

    @staticmethod
    def compute_flow(image_old, image_now, window=35):
        """ Compute optical flow from two cv2 images"""
        Vx_list = []
        Vy_list = []

        flow = cv2.calcOpticalFlowFarneback(
            image_old.astype("uint8"),
            image_now.astype("uint8"),
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=window,
            iterations=1,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        Vx_list.append(flow[:, :, 0])
        Vy_list.append(flow[:, :, 1])

        return np.array(Vx_list), np.array(Vy_list)

    @staticmethod
    def plot_quiver( flow_img, Vx, Vy, spacing, margin=0, scaling=2, size=3, color=(0, 0, 255), centroids=None, thresh=10e-4, **kwargs):
        """quiver plot"""
        if centroids is not None:
            if centroids.shape[0]==0:
                centroids = np.array([[0,0]])
        if centroids is None:
             """Add vector field to img to display optical flow"""
             (h, w) = Vx.shape

             nx = int((w - 2 * margin) / spacing)
             ny = int((h - 2 * margin) / spacing)

             xs = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
             ys = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

             for x in xs:
                 for y in ys:
                     disp_x = scaling * Vx[y, x]
                     disp_y = scaling * Vy[y, x]
                     if np.linalg.norm(np.array([disp_x, disp_y])) > thresh:
                         flow_img = cv2.arrowedLine( flow_img, (x, y), (int(x + disp_x), int(y + disp_y)), color, size)

           

        else:
            xs = centroids[:, 0].astype("int")
            ys = centroids[:, 1].astype("int")
            for x, y in zip(xs, ys):
                disp_x = scaling * Vx[y, x]
                disp_y = scaling * Vy[y, x]
                flow_img = cv2.arrowedLine( flow_img, (x, y), (int(x + disp_x), int(y + disp_y)), color, size)
        return flow_img

#
#    @staticmethod
#    def plot_flow(images, Vxs, Vys, quiver=True):
#        """ Display optical flow results""""
#
#        for counter in range(len(images)):
#            img = images[counter]
#            Vx = Vxs[counter]
#            Vy = Vys[counter]
#            magnitude, angle = cv2.cartToPolar(Vx, Vy)
#            hsv = np.zeros((img.shape[0], img.shape[1], 3))
#            hsv[..., 0] = angle * 180 / np.pi / 2
#            hsv[..., 1] = 255
#            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#            flow_img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
#            img_color = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_GRAY2BGR)
#            if quiver:
#                img_color = plot_quiver(img_color, Vx, Vy, spacing=15)
#            frame = np.concatenate((img_color), axis=1)
#        # cv2.imshow("test", frame)
#        cv2.imshow("test", img_color)
#        # cv2.waitKey(0)
#        ### If the user presses ESC then exit the program
#        key = cv2.waitKey(1)
#        if key == 27:
#            cv2.destroyAllWindows()
#            break
#
