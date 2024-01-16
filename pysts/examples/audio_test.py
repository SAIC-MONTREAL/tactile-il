import numpy as np
import vlc
import time
import cv2
 
from pysts.sts import PySTS
from sts.scripts.sts_transform import STSTransform
from sts.scripts.marker_detection.marker_detection import MarkerDetectionCreator


config_dir = '/home/t.ablett/panda_ros2_ws/src/sts-cam-ros2/configs/demo'
sts = PySTS(config_dir=config_dir)
sts_transform = STSTransform(config_dir)
marker_detector = MarkerDetectionCreator(config_dir).create_object()
sts.set_mode('tactile')


media_player = vlc.MediaPlayer()
media = vlc.Media("/home/t.ablett/panda_ros2_ws/src/sts-cam-ros2/video.mp4")
media_player.set_media(media)
media_player.set_rate(1) 
 
# start playing video
media_player.play()
volume = 50
speed = 1

while True:
    img = sts.get_image()
    transformed_img = sts_transform.transform(img)
    img_dict, vals = marker_detector.detect(transformed_img)
    vals, disp_image = marker_detector.filter_markers_kalman(transformed_img,  vals)
    centroid_img_only = marker_detector.img_from_centroids(0*img_dict['mask'], vals, color=[255,255,255])
    centroid_img_overlay = marker_detector.img_from_centroids(img_dict['mask'], vals, color=[0,0,255])
    img_dict = {}
    img_dict['img'] = img
    img_dict['filtered_on_img'] = centroid_img_overlay                                                                                                                        
    img_dict['displacement'] = disp_image
    avg_disp = np.average(vals - marker_detector.markers_initial, axis=0)
    delta_volume = avg_disp[0]
    delta_speed = -avg_disp[1]
    command_dict = {}
    max_volume = 20
    max_speed = 20
    if np.abs(delta_volume) > np.abs(delta_speed):
        command_dict['type'] = 'volume'
        command_dict['value'] = np.clip(delta_volume / max_volume, -1, 1)
        value = command_dict['value']#100* ((command_dict['value']+1) / 2)
        volume += value * 5
        volume = np.clip(volume, 0, 100)
        print('volume: ', volume)
        media_player.audio_set_volume(int(volume))
        media_player.set_rate(1)
    else:
        command_dict['type'] = 'speed'
        command_dict['value'] = delta_speed
        command_dict['value'] = np.clip(delta_speed / max_speed, -1, 1)
        speed = 1 + command_dict['value'] * .5
        print('speed: ', speed)
        media_player.set_rate(speed)
        media_player.audio_set_volume(int(volume))
    window_name = 'Image'
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2
    volume_text = 'volume: ' + str(volume)[0:5]
    speed_text = 'speed: ' + str(speed)[0:4]
    image = cv2.putText(img_dict['displacement'], volume_text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, speed_text, (50,100), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("frame", image)
    cv2.waitKey(1)

