import os
import cv2
import json
import numpy as np
from subprocess import PIPE, run


class FakeCamera:
    def __init__(self,
        source,
        loop=True
    ):
        self.count = 0
        self.source = source
        self.cam = cv2.VideoCapture(self.source)
        self.num_frames = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.loop = loop

    def get_frame(self):
        success, frame = self.cam.read()
        if not success:
            self.cam.release()
            if self.loop:
                # print(f'looping video')
                self.cam = cv2.VideoCapture(self.source)
                success, frame = self.cam.read()
                if not success:
                    print(f'cv2 unable to loop video')

        self.count += 1

        return success, frame

def plot(img_dict, plot_keys, name='Plot', delay=1):
   if len(plot_keys)==0:
        name = 'empty'
        frame = np.zeros((640, 480))
   else:
       row_counter = 0
       column_counter = 0
       frame_list = [None]
       for i, key in enumerate(plot_keys):
           column_counter += 1
           if column_counter%4==0:
               row_counter += 1
               column_counter = 0
               frame_list.append(None)
           if len(img_dict[key].shape)==3:
               img = img_dict[key]

           else:
               img = cv2.cvtColor(img_dict[key], cv2.COLOR_GRAY2BGR)
           img = draw_text(img, plot_keys[i])

#           img = cv2.putText(img, plot_keys[i], (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
#                   1, (0, 255, 0), 2, cv2.LINE_AA)
           if frame_list[row_counter] is None:
               frame_list[row_counter] = img
           else:
               frame_list[row_counter] = np.concatenate((frame_list[row_counter], img), axis=1)
       if frame_list[0].shape[1]!=frame_list[-1].shape[1]:
           frame_list[row_counter] = np.concatenate((frame_list[row_counter], np.zeros_like(img)), axis=1)

       frame = None
       for _frame in frame_list:
           if frame is None:
               frame = _frame
           else:
               frame = np.concatenate((frame, _frame), axis=0)
   #cv2.setWindowProperty('test', cv2.WND_PROP_FULLSCREEN, 1)


   y_size = 900
   size_factor = int(y_size / frame.shape[0])
   x_size = int(frame.shape[1] * size_factor)
   if frame.shape[1]<1000:
       frame = cv2.resize(frame, (y_size, x_size))
   cv2.imshow(name, frame)
   key = cv2.waitKey(delay)
   if key == 27:  # press ESC to quit
       cv2.destroyAllWindows()
       return True
   return key

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_COMPLEX,
          pos=(0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    img = cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return img

def split_channels(img, color=True):
    img_list = []
    for i in range(3):
        img_tmp = np.zeros(img.shape).astype('uint8')
        if color:
            img_tmp[:,:,i]= img[:,:,i]
        else:
            for j in range(3):
                img_tmp[:,:,j]= img[:,:,i]
        img_list.append(img_tmp)
    return img_list

def filter_image(img, filter_type='BGR'):
    img_list = []
    if filter_type=='BGR':
        return split_channels(img)
    if filter_type=="HSV" or filter_type=="LAB":
        if filter_type=="HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif filter_type=="LAB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return split_channels(img, color=False)


def normalize_img(arr):
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

def nothing(x):
    pass

def bash_out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

def get_package_path(package_name='sts'):
    return bash_out('ros2 pkg prefix ' + package_name)[:-1]

def get_ws_path():
    output = get_package_path('sts').split('/')
    return '/' + os.path.join(*output[:-2])

def mask_image(img, mask):
    return img * mask

def warp_image(img, M):
    return cv2.warpPerspective(img.astype('uint8'), np.array(M), (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def read_json(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    for key in data:
        if isinstance(data[key], str):
            if data[key][0]=='.':
                data[key] = '/' + os.path.join(*filepath.split('/')[:-1], data[key])
    return data

def write_json(filepath, dict):
    t = json.dumps(dict, indent=4)
    print(f"writing {t}")
    print(f"to {filepath}")
    with open(filepath, 'w') as fp:
        fp.write(t)
