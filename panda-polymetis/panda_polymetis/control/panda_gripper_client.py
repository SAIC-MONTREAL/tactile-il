import time
import numpy as np
from polymetis import GripperInterface


class PandaGripperClient:
    def __init__(self, server_ip='localhost', open_width=0.08, grasp_force=10, grip_speed=0.05,
                 close_force=60, pinch_width=.04, not_open_means_close=True):
        print("Starting Panda gripper client.")

        self.server_ip = server_ip
        self.open_width = open_width
        self.default_grasp_force = grasp_force
        self.default_speed = grip_speed
        self.default_close_force = close_force
        self.default_pinch_width = pinch_width

        # self.gripper = GripperInterface(ip_address=self.server_ip, enforce_version=False)
        self.gripper = GripperInterface(ip_address=self.server_ip)

        self._state = "none"  # in case robot turns on with gripper not open

        if not_open_means_close:
            # check to see if already grasping -- assume grasping if not fully open
            self.get_and_update_state()
            if self._pos < (self._max_width - .001):  # a bit of tolerance
                self._state = "close"

        # this doesn't actually work
        # state_init = self.get_and_update_state()
        # time.sleep(.1)
        # state_after = self.get_and_update_state()
        # if state_after["is_moving"] and np.linalg.norm(state_init["pos"] - state_after["pos"]) < .001:
        #     self._state = "close"

        # internal state tracking
        self._pos = None
        self._error_code = 0

    def get_and_update_state(self):
        # google.protobuf.Timestamp timestamp = 1;
        # float width = 2;`
        # bool is_grasped = 3;
        # bool is_moving = 4;
        # bool prev_command_successful = 6;
        # int32 error_code = 5;`
        state = self.gripper.get_state()
        self._pos = state.width
        # self._error_code = state.error_code
        self._max_width = state.max_width

        return {
            "pos": state.width,
            "max_width": state.max_width,
            "is_moving": state.is_moving
        }

    def send_move_goal(self, width, speed=None, force=None, blocking=False):
        if speed is None: speed = self.default_speed
        if force is None: force = self.default_close_force
        self.gripper.goto(width=width, speed=speed, force=force, blocking=blocking)

        return True  # backwards compatibility

    def open(self, speed=None, blocking=False):
        if speed is None: speed = self.default_speed
        if self._state != "open":
            self._state = "open"
            self.send_move_goal(width=self.open_width, speed=speed, blocking=blocking)

        return True  # backwards compatibility

    def close(self, speed=None, force=None, blocking=False):
        if speed is None: speed = self.default_speed
        if force is None: force = self.default_grasp_force
        if self._state != "close":
            self._state = "close"
            self.grasp(force=force, width=0.0, speed=speed, blocking=blocking)

        return True  # backwards compatibility

    def grasp(self, force, width=0, ep_inner=-1, ep_outer=-1, speed=None, blocking=False):
        if speed is None: speed = self.default_speed

        # self.gripper.grasp(speed=speed, force=force, grasp_width=width, epsilon_inner=ep_inner,
        #                    epsilon_outer=ep_outer, blocking=blocking)
        self.gripper.grasp(speed=speed, force=force, blocking=blocking)

        return True  # backwards compatibility

    def pinch(self, width=None, speed=None, blocking=False):
        if width is None: width = self.default_pinch_width
        if speed is None: speed = self.default_speed
        if self._state != "pinch":
            self._state = "pinch"
            self.send_move_goal(width=width, speed=speed, blocking=blocking)

        return True  # backwards compatibility


class FakePandaGripperClient:
    def __init__(self, server_ip='localhost', open_width=0.08, grasp_force=10, grip_speed=0.1,
                 close_force=10, pinch_width=.04):
        print("Starting fake panda gripper client.")

        self.server_ip = server_ip
        self.open_width = open_width
        self.default_grasp_force = grasp_force
        self.default_speed = grip_speed
        self.default_close_force = close_force
        self.default_pinch_width = pinch_width

        self._state = "none"  # in case robot turns on with gripper not open

        # internal state tracking
        self._pos = None
        self._error_code = 0

    def get_and_update_state(self):
        if self._state == 'open':
            pos = self.open_width
        elif self._state == 'close':
            pos = 0.0
        elif self._state == 'pinch':
            pos = self.default_pinch_width
        else:
            pos = .024

        self._pos = pos
        self._error_code = 0

        return {
            "pos": pos,
            "error_code": 0,
        }

    def send_move_goal(self, width, speed=None, force=None, blocking=False):
        return True  # backwards compatibility

    def open(self, speed=None, blocking=False):
        if self._state != "open":
            self._state = "open"
        return True  # backwards compatibility

    def close(self, speed=None, force=None, blocking=False):
        if self._state != "close":
            self._state = "close"
        return True  # backwards compatibility

    def grasp(self, force, width=0, ep_inner=-1, ep_outer=-1, speed=None, blocking=False):
        return True  # backwards compatibility

    def pinch(self, width=None, speed=None, blocking=False):
        if self._state != "pinch":
            self._state = "pinch"
        return True