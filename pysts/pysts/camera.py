import json
import cv2
import threading
import queue
import time
import platform
from functools import partial

from pysts.utils import Rate


class FakeCamera:
    def __init__(self,
        source,
        camera_resolution="",
        loop=True,
        rate=None  # no delay in grabbing images if None, otherwise simulate camera delay
    ):
        self.count = 0
        self.loop_count = 0
        self.source = source
        self.cam = cv2.VideoCapture(self.source)
        self.num_frames = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))

        self._cv2_resolution_modifier = None
        if len(camera_resolution) > 0:  # handles both lists and strings
            width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cur_res_str = f"{width}x{height}"
            if type(camera_resolution) == list:
                camera_resolution = f"{camera_resolution[0]}x{camera_resolution[1]}"
            if camera_resolution != cur_res_str:
                w, h = camera_resolution.split('x')
                self._cv2_resolution_modifier = partial(cv2.resize, dsize=(int(w), int(h)))

        self.loop = loop
        if rate is not None:
            self.rate = Rate(rate)
        else:
            self.rate = None

    def reset_video(self, source=None):
        self.count = 0
        # optionally allow selecting a new video
        if source is not None:
            self.source = source
        self.cam = cv2.VideoCapture(self.source)
        print("Simulated camera video reset.")

    def get_image(self):
        if self.rate is not None:
            self.rate.sleep()
        success, frame = self.cam.read()
        if not success:
            self.cam.release()
            if self.loop:
                # print(f'looping video')
                self.reset_video()
                self.loop_count += 1
                success, frame = self.cam.read()
                if not success:
                    print(f'cv2 unable to loop video')

        self.count += 1

        if self._cv2_resolution_modifier is not None:
            frame = self._cv2_resolution_modifier(frame)

        return frame


class Camera:
    def __init__(self, config_dir, camera_resolution=""):
        # parameters
        with open(f"{config_dir}/camera_resolutions.json") as fd:
            self._config = json.load(fd)
        with open(f"{config_dir}/tactile.json") as fd:
            self.tactile_config = json.load(fd)

        if platform.system() == 'Darwin':  # for MacOS, no /dev/video
            self._camera = cv2.VideoCapture(int(self.tactile_config['device']))
        else:
            self._camera = cv2.VideoCapture("/dev/video" + str(self.tactile_config['device']))

        # since 160x120 doesn't get set properly, we'll use cv2 to manually resize images when we want 160x120
        self._cv2_resolution_modifier = None

        if len(camera_resolution) == 0:  # handle both lists and strings
            self.set_resolution(self.get_default_resolution())
        else:
            if type(camera_resolution) == list:
                camera_resolution = f"{camera_resolution[0]}x{camera_resolution[1]}"
            self.set_resolution(camera_resolution)

        # add thread so we can always grab only the latest image
        self._latest_img = None
        self._image_updated = False
        self.lock = threading.Lock()
        self.cam_q = queue.Queue()
        self.thread = threading.Thread(target=self._reader, args=(self.cam_q,))
        self.thread.daemon = True
        self.thread.start()

        while self._latest_img is None:
            time.sleep(.0001)

    def _reader(self, q: queue.Queue):
        while True:
            _, img = self._camera.read()
            if self._cv2_resolution_modifier is not None:
                img = self._cv2_resolution_modifier(img)
            self._latest_img = img
            while not q.empty():  # clear so get_image only gets latest
                q.get(block=False)
            q.put(img)

    def get_image(self, block_until_latest=True):
        if block_until_latest:
            img = self.cam_q.get()
            return img
        else:
            return self._latest_img

    def get_default_resolution(self):
        """Get the default resolution"""
        return self._config['default']

    def set_resolution(self, s):
        """Set resolution to form widthxheight"""
        if s not in self._config['resolutions']:
            self.log_print(f"Camera does not support resolution {s}")
            return
        if s == '160x120':
            s = '320x240'
            self._cv2_resolution_modifier = partial(cv2.resize, dsize=(160, 120))
        elif s== '212x120':  # the 16:9 aspect ratio version
            s = '424x240'
            self._cv2_resolution_modifier = partial(cv2.resize, dsize=(212, 120))

        try:
            wh = s.split('x')
            w = int(wh[0])
            h = int(wh[1])
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            width = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w != width or h != height:
                self.log_print(f'Camera unable to set resolution {w}x{h}')
            self.log_print(f'Camera set resolution to {width}x{height}')
            if self._cv2_resolution_modifier is not None:
                true_width, true_height = self._cv2_resolution_modifier.keywords['dsize']
                self.log_print(f'Camera resolution modifier enabled, resolution will be {true_width}x{true_height}')
        except KeyError:
            self.log_print(f'Camera resolution {s} not found')

    def log_print(self, string):
        print(f"Camera: {string}")