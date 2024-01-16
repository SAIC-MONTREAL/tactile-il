import termios
import tty
import pickle
import os
import sys
import argparse

import panda_polymetis
from panda_polymetis.control.panda_client import PandaClient
from panda_polymetis.control.panda_gripper_client import PandaGripperClient



INFO_MSG = """
Reading from the keyboard
---------------------------
7: toggle base frame vs. tool frame translation
8: toggle base frame vs. tool frame rotation
u: +x
j: -x
i: +y
k: -y
o: +z
l: -z
4: +rotx
1: -rotx
5: +roty
2: -roty
6: +rotz
3: -rotz

space: Activate or deactivate free drive
n: Open gripper
m: Close gripper
b: Pinch gripper
f: Toggle debug
e: Clear errors  (CURRENTLY NON-FUNCTIONAL)
c: Print current list of commands
r: Write commands to save_path
t: Empty list of commands (start logging a new set of commands)

q: Save joints
w: Save pose
a: Save "open gripper" command
s: Save "close gripper" command
d: Save "pinch mode for gripper" command
z: Save "activate full compliance" command
v: Save "activate rotational compliance" command
x: Save "deactivate compliance" command

CTRL-C to quit
"""

class KeyboardInterface:
    """
    Keyboard interface to store and replay trajectories.
    Assumes methods defined in panda_client.py and srj_gripper_client.py
    """
    def __init__(
        self,
        save_path,
        arm_client,
        gripper_client,
        base_frame=True,
        rot_base_frame=False,
        debug=False
    ):
        self.arm_client = arm_client
        self.gripper_client = gripper_client
        self.key_bindings = {
            'q',
            'w',
            'a',
            's',
            'd',
            'z',
            'v',
            'x',
            ' ',
            'n',
            'm',
            'b',
            'f',
            'e',
            'c',
            'r',
            't',
            'p'
        }
        self.move_key_bindings = 'uiojkl12345678'
        print(INFO_MSG)
        print(f"Saving commands to {save_path}")
        self.settings = self.save_terminal_settings()
        self.counter = 0
        self.save_path = save_path
        self.debug = debug
        self.command_list = []
        self.base_frame = base_frame
        self.rot_base_frame = rot_base_frame
        self.t_mv = .003
        self.r_mv = .03
        self.run()

    def run(self):
        freedrive_activated = False
        try:
            while(1):
                key = self.get_key(self.settings)

                if key in self.move_key_bindings:
                    bf_args = dict(base_frame=self.base_frame, rot_base_frame=self.rot_base_frame)
                    k = key
                    if k == 'u':
                        self.arm_client.shift_EE_by([self.t_mv, 0, 0], **bf_args)
                    elif k == 'j':
                        self.arm_client.shift_EE_by([-self.t_mv, 0, 0], **bf_args)
                    elif k == 'i':
                        self.arm_client.shift_EE_by([0.0, self.t_mv, 0], **bf_args)
                    elif k == 'k':
                        self.arm_client.shift_EE_by([0.0, -self.t_mv, 0], **bf_args)
                    elif k == 'o':
                        self.arm_client.shift_EE_by([0.0, 0, self.t_mv], **bf_args)
                    elif k == 'l':
                        self.arm_client.shift_EE_by([0.0, 0, -self.t_mv], **bf_args)
                    elif k == '4':
                        self.arm_client.shift_EE_by([0.0, 0, 0], [self.r_mv, 0, 0], **bf_args)
                    elif k == '1':
                        self.arm_client.shift_EE_by([0.0, 0, 0], [-self.r_mv, 0, 0], **bf_args)
                    elif k == '5':
                        self.arm_client.shift_EE_by([0.0, 0, 0], [0, self.r_mv, 0], **bf_args)
                    elif k == '2':
                        self.arm_client.shift_EE_by([0.0, 0, 0], [0, -self.r_mv, 0], **bf_args)
                    elif k == '6':
                        self.arm_client.shift_EE_by([0.0, 0, 0], [0, 0, self.r_mv], **bf_args)
                    elif k == '3':
                        self.arm_client.shift_EE_by([0.0, 0, 0], [0, 0, -self.r_mv], **bf_args)
                    elif k == '7':
                        self.base_frame = not self.base_frame
                        print(f"Base frame for motions set to {self.base_frame}")
                    elif k == '8':
                        self.rot_base_frame = not self.rot_base_frame
                        print(f"Rot base frame for motions set to {self.rot_base_frame}")

                elif key in self.key_bindings:
                    if key == "q":
                        self.arm_client.get_and_update_state()
                        if not self.debug:
                            print(
                                f"Storing joint waypoint {self.arm_client.joint_position}"
                            )
                        self.store_command((key, self.arm_client.joint_position))
                    elif key == "w":
                        self.arm_client.get_and_update_state()
                        if not self.debug:
                            print(
                                f"Storing pose waypoint {self.arm_client.EE_pose.get_array_quat()}"
                            )
                        self.store_command((key, self.arm_client.EE_pose))
                    elif key == "a":
                        if not self.debug: print("Storing open gripper command")
                        self.store_command((key, None))
                    elif key == "s":
                        if not self.debug: print("Storing close gripper command")
                        self.store_command((key, None))
                    elif key == "d":
                        if not self.debug: print("Storing go to pinch configuration command")
                        self.store_command((key, None))
                    elif key == "z":
                        if not self.debug: print("Storing activate full compliance mode command")
                        self.store_command((key, None))
                    elif key == "v":
                        if not self.debug: print("Storing activate rotational compliance mode command")
                        self.store_command((key, None))
                    elif key == "x":
                        if not self.debug: print("Storing deactivate compliance mode command")
                        self.store_command((key, None))
                    elif key == " ":
                        if freedrive_activated:
                            print("Deactivating freedrive")
                            self.arm_client.deactivate_freedrive()
                            freedrive_activated = False
                        else:
                            print("Activating freedrive")
                            self.arm_client.activate_freedrive()
                            freedrive_activated = True
                    elif key == "b":
                        print("Pinching gripper")
                        self.gripper_client.pinch()
                    elif key == "n":
                        print("Opening gripper")
                        self.gripper_client.open()
                    elif key == "m":
                        print("Closing gripper")
                        self.gripper_client.close()
                    elif key == "f":
                        print(f"Toggling debug to {not self.debug}")
                        self.toggle_debug()
                    elif key == "e":
                        print("Sending error recovery request")
                        self.arm_client.recover_from_errors()
                    elif key == "c":
                        print(f"Current commands: {[d[0] for d in self.command_list]}")
                    elif key == "r":
                        print(f"Writing commands to file: {self.save_path}.pkl")
                        self.save_commands()
                    elif key == "t":
                        print("Clearing stored commands")
                        self.clear_commands()
                    elif key == "p":
                        # debugging
                        print(f"ROBOT MODE: {self.arm_client.get_robot_mode()}")

                else:
                    if (key == '\x03'):
                        break
        except Exception as e:
            print(e)
        finally:
            self.restore_terminal_settings(self.settings)

    def toggle_debug(self):
        self.debug = not self.debug

    def store_command(self, command):
        if not self.debug:
            self.command_list.append(command)

    def save_commands(self):
        if not self.debug:
            with open(os.path.join(f"{self.save_path}.pkl"), "wb") as handle:
                pickle.dump(self.command_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def clear_commands(self):
        self.command_list = []

    def get_key(self, settings):
        if sys.platform == 'win32':
            # getwch() returns a string on Windows
            key = msvcrt.getwch()
        else:
            tty.setraw(sys.stdin.fileno())
            # sys.stdin.read() returns a string on Linux
            key = sys.stdin.read(1)
            # print("key", key)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def save_terminal_settings(self):
        if sys.platform == 'win32':
            return None
        return termios.tcgetattr(sys.stdin)

    def restore_terminal_settings(self, old_settings):
        if sys.platform == 'win32':
            return
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def __del__(self):
        self.restore_terminal_settings(self.settings)


def parse_args():
    format_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=format_class,
                    description='Default info')
    parser.add_argument('--server_ip', type=str, default='localhost',
                help='ip address of server, on real robot, nuc ip')
    parser.add_argument('--filename', type=str, default=None,
                help='Name of file where stored commands will be saved')
    parser.add_argument('--no_gripper', action='store_true', help='run without gripper')
    parser.add_argument('--gripper_ip', type=str, default='localhost', help='run without gripper')
    parser.add_argument('--debug_mode', action='store_true',
                help='Run in debug mode and save no trajectories')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    # Get directory to store data
    main_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if args.filename is not None:
        save_path = os.path.join(main_dir, "data", args.filename)
    else:
        save_path = os.path.join(main_dir, "data", "stored_commands")

    # Start clients
    pp_dir = os.path.dirname(panda_polymetis.__file__)
    json_file = os.path.join(pp_dir, 'conf/franka-desk/franka-gripper-and-blue-realsense.json')
    panda_client = PandaClient(
        server_ip=args.server_ip,
        delta_pos_limit=.02,
        delta_rot_limit=.2,
        ee_config_json=json_file)

    gripper_client = None
    if not args.no_gripper:
        gripper_client = PandaGripperClient(
            # server_ip='localhost',  # should be run NOT on NUC!
            server_ip=args.gripper_ip,
        )

    # Start keyboard interface
    keyboard_interface = KeyboardInterface(
        save_path=save_path,
        arm_client=panda_client,
        gripper_client=gripper_client,
        debug=args.debug_mode
    )