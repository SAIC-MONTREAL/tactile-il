# Reference frame manipulation

This README provides visualization to help understand the inputs of the key reference frame conversions in pose_transforms.py

## convert_reference_frame

There are 3 reference frames of interest, the source frame, the target frame, and the object pose.

<img src="https://github.sec.samsung.net/f-hogan/utils/blob/main/assets/convert_frames_1.png" width="300"/>

The convert_reference_frame converts the point of view on the object from the source frame to the target frame.

<img src="https://github.sec.samsung.net/f-hogan/utils/blob/main/assets/convert_frames_2.png" width="300"/>

The pose of the source and target frames are resolves in a common "world" frame. The frame of the object pose is resolve in the source frame.

<img src="https://github.sec.samsung.net/f-hogan/utils/blob/main/assets/convert_frames_3.png" width="300"/>

Often times, it is convenient to define the source pose with unit pose, that is to say superimposed with the world frame.


## transform_local_body

This function  transforms the pose of an object with a defined transformation in the local body frame. There are 2 reference frames of interest: 1) pose_source_world: pose of the object in a given frame 2) the desired pose transformation, defined relative to the object's body pose (not the world frame). The function returns the pose of the transformed object in the same given frame as pose_source_world. 

<img src="https://github.sec.samsung.net/f-hogan/utils/blob/main/assets/transform_local_body.png" width="300"/>
