import os

import contact_panda_envs.envs.panda_generic as panda_generic
from contact_panda_envs.envs.panda_generic import PandaGeneric


CABINET_DEFAULTS = dict(
    img_in_state=True,
    depth_in_state=False,
    state_data=['pose', 'prev_pose', 'grip_pos', 'prev_grip_pos', 'joint_pos', 'raw_world_pose'],
)

ONEFINGER_DEFAULTS = dict(
    grip_in_action=False,
    state_data=['pose', 'prev_pose', 'joint_pos', 'raw_world_pose'],
)

CABINET_ONEFINGER_DEFAULTS = CABINET_DEFAULTS
CABINET_ONEFINGER_DEFAULTS.update(ONEFINGER_DEFAULTS)

CABINET_ONEFINGER_NOSTS_DEFAULTS = dict(
    **CABINET_ONEFINGER_DEFAULTS,
    sts_namespaces=[],
    sts_config_dirs=[],
    sts_data=[],
    sts_images=[]
)

NO_STS_SWITCH_DEFAULTS = dict(
    sts_no_switch_override=True,
    # sts_initial_mode='halfway'
    sts_initial_mode='tactile'
)

TACTILE_ONLY_DEFAULTS = dict(
    sts_no_switch_override=True,
    # sts_initial_mode='halfway'
    sts_initial_mode='tactile',
    sts_switch_in_action=False
)

VISUAL_ONLY_DEFAULTS = dict(
    sts_no_switch_override=True,
    # sts_initial_mode='halfway'
    sts_initial_mode='visual',
    sts_switch_in_action=False
)

ROS_FT_DEFAULTS = dict(
    polymetis_control=False,
    state_data=['pose', 'prev_pose', 'grip_pos', 'prev_grip_pos', 'joint_pos', 'raw_world_pose',
                'force_torque_internal']
)

# ------------------------------------------------------------------------------------------------------------
# 2DOF One Finger Envs
# ------------------------------------------------------------------------------------------------------------

CABINET_ONEFINGER_2DOF_DEFAULTS = dict(**CABINET_ONEFINGER_DEFAULTS, **panda_generic.XZ_DEFAULTS)
CABINET_ONEFINGER_2DOF_NOSTS_DEFAULTS = dict(**CABINET_ONEFINGER_NOSTS_DEFAULTS, **panda_generic.XZ_DEFAULTS)


class PandaCabinetOneFinger2DOF(PandaGeneric):
    def __init__(self, sts_config_dir, **kwargs):
        super().__init__(config_override_dict=
            dict(**CABINET_ONEFINGER_2DOF_DEFAULTS, sts_config_dirs=[sts_config_dir]), **kwargs)


class PandaCabinetOneFingerNoSTS2DOF(PandaGeneric):
    def __init__(self, **kwargs):
        super().__init__(config_override_dict=dict(**CABINET_ONEFINGER_2DOF_NOSTS_DEFAULTS), **kwargs)


# ------------------------------------------------------------------------------------------------------------
# 6DOF One Finger Envs
# ------------------------------------------------------------------------------------------------------------

CABINET_ONEFINGER_6DOF_DEFAULTS = dict(**CABINET_ONEFINGER_DEFAULTS, **panda_generic.SIXDOF_DEFAULTS)
CABINET_ONEFINGER_6DOF_NOSTS_DEFAULTS = dict(**CABINET_ONEFINGER_NOSTS_DEFAULTS, **panda_generic.SIXDOF_DEFAULTS)

class PandaCabinetOneFinger6DOF(PandaGeneric):
    def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
        super_config_override_dict = dict(**CABINET_ONEFINGER_6DOF_DEFAULTS, sts_config_dirs=[sts_config_dir])
        super_config_override_dict.update(config_override_dict)
        # super().__init__(config_override_dict=
        #     dict(**CABINET_ONEFINGER_6DOF_DEFAULTS, sts_config_dirs=[sts_config_dir], **config_override_dict), **kwargs)
        super().__init__(config_override_dict=super_config_override_dict, **kwargs)

class PandaCabinetOneFinger6DOFROS(PandaGeneric):
    def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
        super_config_override_dict = dict(**CABINET_ONEFINGER_6DOF_DEFAULTS, sts_config_dirs=[sts_config_dir])
        super_config_override_dict.update(ROS_FT_DEFAULTS)
        super_config_override_dict.update(config_override_dict)
        super().__init__(config_override_dict=super_config_override_dict, **kwargs)

class PandaCabinetOneFingerNoSTS6DOF(PandaGeneric):
    def __init__(self, config_override_dict={}, **kwargs):
        super().__init__(config_override_dict=dict(**CABINET_ONEFINGER_6DOF_NOSTS_DEFAULTS, **config_override_dict), **kwargs)

# ------------------------------------------------------------------------------------------------------------
# Real robot test (reach) envs
# ------------------------------------------------------------------------------------------------------------

class PandaTestOneFingerNoSTS6DOFRealRandomInit(PandaCabinetOneFingerNoSTS6DOF):
    def __init__(self, **kwargs):
        super().__init__(
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "real-env-tests.yaml"),
            **kwargs)


class PandaTestOneFingerNoSTS6DOFRealNoRandomInit(PandaTestOneFingerNoSTS6DOFRealRandomInit):
    def __init__(self, config_override_dict={}, **kwargs):
        super().__init__(
            config_override_dict=dict(**config_override_dict, init_gripper_random_lim=[0, 0, 0, 0, 0, 0]), **kwargs)


class PandaTestOneFinger6DOFRealRandomInit(PandaCabinetOneFinger6DOF):
    def __init__(self, sts_config_dir, **kwargs):
        super().__init__(
            sts_config_dir=sts_config_dir,
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "real-env-tests.yaml"),
            **kwargs)

# ------------------------------------------------------------------------------------------------------------
# Real robot cabinet envs
# ------------------------------------------------------------------------------------------------------------

class PandaBlackTopOneFinger6DOFRealRandomInit(PandaCabinetOneFinger6DOF):
    def __init__(self, sts_config_dir, **kwargs):
        super().__init__(
            sts_config_dir=sts_config_dir,
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-ct-black-knob.yaml"),
            **kwargs)

# class PandaTopGlassOrb(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, **kwargs):
#         super().__init__(
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-glass-orb.yaml"),
#             **kwargs)

# class PandaTopGlassOrbClose(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, **kwargs):
#         super().__init__(
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-glass-orb-close.yaml"),
#             **kwargs)

class PandaTopGlassOrbOneFinger6DOFRealNoRandomInit(PandaCabinetOneFinger6DOF):
    def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
        super().__init__(
            config_override_dict=dict(**config_override_dict, init_gripper_random_lim=[0, 0, 0, 0, 0, 0]),
            sts_config_dir=sts_config_dir,
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-glass-orb.yaml"),
            **kwargs)

class PandaTopGlassOrbOneFingerNoSTS6DOFRealNoRandomInit(PandaCabinetOneFingerNoSTS6DOF):
    def __init__(self, config_override_dict={}, **kwargs):
        super().__init__(
            config_override_dict=dict(**config_override_dict, init_gripper_random_lim=[0, 0, 0, 0, 0, 0]),
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-glass-orb.yaml"),
            **kwargs)

class PandaTopGlassOrbNoSTSSwitch(PandaCabinetOneFinger6DOF):
    def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
        super().__init__(
            config_override_dict=dict(**config_override_dict, **NO_STS_SWITCH_DEFAULTS),
            sts_config_dir=sts_config_dir,
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-glass-orb.yaml"),
            **kwargs)

class PandaTopWipe(PandaCabinetOneFinger6DOF):
    def __init__(self, sts_config_dir, **kwargs):
        super().__init__(
            sts_config_dir=sts_config_dir,
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-door-wipe.yaml"),
            **kwargs)

class PandaTopShelfWipe(PandaCabinetOneFinger6DOF):
    def __init__(self, sts_config_dir, **kwargs):
        super().__init__(
            sts_config_dir=sts_config_dir,
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-shelf-wipe.yaml"),
            **kwargs)

# ------------------------------------------------------------------------------------------------------------
# Silver handle bottom
# ------------------------------------------------------------------------------------------------------------

class PandaBottomSilverHandle(PandaCabinetOneFinger6DOF):
    def __init__(self, sts_config_dir, **kwargs):
        super().__init__(
            sts_config_dir=sts_config_dir,
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "bottom-silver-handle.yaml"),
            **kwargs)

class PandaBottomSilverHandleNoSTSSwitch(PandaCabinetOneFinger6DOF):
    def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
        super().__init__(
            config_override_dict=dict(**config_override_dict, **NO_STS_SWITCH_DEFAULTS),
            sts_config_dir=sts_config_dir,
            robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "bottom-silver-handle.yaml"),
            **kwargs)

# ------------------------------------------------------------------------------------------------------------
# Class generator for new classes (PandaTopBlackFlatHandle, PandaBottomSilverCylinder)
# ------------------------------------------------------------------------------------------------------------

# based on this https://stackoverflow.com/questions/15247075/how-can-i-dynamically-create-derived-classes-from-a-base-class
def open_close_factory(base_name, oc_type, sensor_type, def_config_filename):
    BaseClass = PandaCabinetOneFinger6DOF
    name = base_name
    config_override_dict_def = {}

    if sensor_type.startswith('ros'):
        config_override_dict_def.update(ROS_FT_DEFAULTS)
        name += "ROS"

    if oc_type == 'open':
        robot_config_file = os.path.join(os.path.dirname(__file__), "configs", def_config_filename + ".yaml")
    elif oc_type == 'close':
        robot_config_file = os.path.join(os.path.dirname(__file__), "configs", def_config_filename + "-close.yaml")
        name += "Close"
    else:
        raise NotImplementedError("Only implemented for oc_type open and close")
    if 'visual' in sensor_type:
        config_override_dict_def.update(VISUAL_ONLY_DEFAULTS)
        name += "VisualOnly"
    elif 'tactile' in sensor_type:
        config_override_dict_def.update(TACTILE_ONLY_DEFAULTS)
        name += "TactileOnly"
    elif sensor_type == 'switching':
        pass
    else:
        raise NotImplementedError("Only implemented for sensor_type visual, tactile, or switching")

    # if sensor_type == "ros_tactile":
    #     import ipdb; ipdb.set_trace()

    def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
        super_config_override_dict = config_override_dict_def
        super_config_override_dict.update(config_override_dict)
        BaseClass.__init__(self,
            config_override_dict=super_config_override_dict,
            sts_config_dir=sts_config_dir,
            robot_config_file=robot_config_file,
            **kwargs)

    new_class = type(name, (BaseClass,), {"__init__": __init__})
    return new_class

def add_all_classes_to_scope(base_name, def_config_filename):
    configs = (
        {'oc_type': 'open', 'sensor_type': 'switching'},
        {'oc_type': 'open', 'sensor_type': 'tactile'},
        {'oc_type': 'open', 'sensor_type': 'visual'},
        {'oc_type': 'open', 'sensor_type': 'ros_tactile'},
        {'oc_type': 'close', 'sensor_type': 'switching'},
        {'oc_type': 'close', 'sensor_type': 'tactile'},
        {'oc_type': 'close', 'sensor_type': 'visual'},
        {'oc_type': 'close', 'sensor_type': 'ros_tactile'},
    )

    for conf in configs:
        new_class = open_close_factory(base_name, def_config_filename=def_config_filename, **conf)
        globals()[new_class.__name__] = new_class


add_all_classes_to_scope("PandaTopBlackFlatHandle", 'top-flat-black-handle')
add_all_classes_to_scope("PandaTopBlackFlatHandleNewPos", 'top-flat-black-handle-new-pos')
add_all_classes_to_scope("PandaBottomSilverCylinder", 'bottom-silver-cylinder')
add_all_classes_to_scope("PandaTopGlassOrb", 'top-glass-orb')
add_all_classes_to_scope("PandaTopGlassOrbNewPos", 'top-glass-orb-new-pos')
add_all_classes_to_scope("PandaBottomGlassOrb", 'bottom-glass-orb')

# class PandaTopBlackFlatHandle(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, **kwargs):
#         super().__init__(
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-flat-black-handle.yaml"),
#             **kwargs)

# class PandaTopBlackFlatHandleTactileOnly(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
#         super().__init__(
#             config_override_dict=dict(**config_override_dict, **TACTILE_ONLY_DEFAULTS),
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-flat-black-handle.yaml"),
#             **kwargs)

# class PandaTopBlackFlatHandleVisualOnly(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
#         super().__init__(
#             config_override_dict=dict(**config_override_dict, **VISUAL_ONLY_DEFAULTS),
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-flat-black-handle.yaml"),
#             **kwargs)

# class PandaTopBlackFlatHandleClose(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, **kwargs):
#         super().__init__(
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-flat-black-handle-close.yaml"),
#             **kwargs)

# class PandaTopBlackFlatHandleCloseTactileOnly(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
#         super().__init__(
#             config_override_dict=dict(**config_override_dict, **TACTILE_ONLY_DEFAULTS),
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-flat-black-handle-close.yaml"),
#             **kwargs)

# class PandaTopBlackFlatHandleCloseVisualOnly(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
#         super().__init__(
#             config_override_dict=dict(**config_override_dict, **VISUAL_ONLY_DEFAULTS),
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-flat-black-handle-close.yaml"),
#             **kwargs)

# class PandaTopBlackFlatHandleROSTactileOnly(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
#         super().__init__(
#             config_override_dict=dict(**config_override_dict, **ROS_FT_DEFAULTS),
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-flat-black-handle.yaml"),
#             **kwargs)

# class PandaTopBlackFlatHandleROSCloseTactileOnly(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, config_override_dict={}, **kwargs):
#         super().__init__(
#             config_override_dict=dict(**config_override_dict, **ROS_FT_DEFAULTS),
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "top-flat-black-handle-close.yaml"),
#             **kwargs)

# ------------------------------------------------------------------------------------------------------------
# Silver Cylinder Bottom
# ------------------------------------------------------------------------------------------------------------

# class PandaBottomSilverCylinder(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, **kwargs):
#         super().__init__(
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "bottom-silver-cylinder.yaml"),
#             **kwargs)

# class PandaBottomSilverCylinderClose(PandaCabinetOneFinger6DOF):
#     def __init__(self, sts_config_dir, **kwargs):
#         super().__init__(
#             sts_config_dir=sts_config_dir,
#             robot_config_file=os.path.join(os.path.dirname(__file__), "configs", "bottom-silver-cylinder-close.yaml"),
#             **kwargs)