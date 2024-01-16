import copy
import time
from threading import Thread, Lock
import queue
import signal
import sys

from contact_il.imitation.devices import Keyboard


class Button:
    def __init__(self, hold_time_length):
        self.state = False
        self.last_state = False
        self.hold_state = False
        self.hold_time_start = time.time()
        self.last_hold_state = False
        self.hold_time_length = hold_time_length
        self.stored_state = dict(re=False, fe=False, rhe=False, fhe=False)

    def get_update(self, raw_state, cur_time):
        """
        Update the button state and hold state and return the rising and falling edges.
        :param raw_state: The raw state of the button from its source.
        :return: Whether there is a rising edge, falling edge, rising edge of being held, and falling edge of
                 being held.
        """
        self.last_hold_state = self.hold_state
        self.last_state = self.state
        self.state = raw_state
        if self.state and not self.last_state:
            self.hold_time_start = cur_time
            rising_edge = True
        else:
            rising_edge = False
        if not self.state and self.last_state:
            falling_edge = True
        else:
            falling_edge = False

        # hold state stuff
        if cur_time - self.hold_time_start > self.hold_time_length and self.state:
            self.hold_state = True
        else:
            self.hold_state = False
        if self.hold_state and not self.last_hold_state:
            hold_rising_edge = True
        else:
            hold_rising_edge = False
        if not self.hold_state and self.last_hold_state:
            hold_falling_edge = True
        else:
            hold_falling_edge = False

        return rising_edge, falling_edge, hold_rising_edge, hold_falling_edge

    def get_and_store_update(self, raw_state, cur_time):
        """ Only allows changing False to True, stores between calls to reset_state.
            Useful if we're updating in a thread and need to detect these events in a slower updating caller. """
        re, fe, hre, hfe = self.get_update(raw_state, cur_time)
        self.stored_state['re'] = re or self.stored_state['re']
        self.stored_state['fe'] = fe or self.stored_state['fe']
        self.stored_state['rhe'] = hre or self.stored_state['rhe']
        self.stored_state['fhe'] = hfe or self.stored_state['fhe']

    def reset_state(self):
        for k in self.stored_state:
            self.stored_state[k] = False


class Device:
    BUTTONS = {'reset', 'delete', 'start_stop', 'save', 'gripper', 'success', 'failure', 'error_recovery', 'sts_switch'}
    def __init__(
        self,
        device_type='keyboard',
        device_thread=True,
        thread_freq=100,
        hold_time=1
    ):
        """Update the button states of a device, optionally with a thread, allowing user to query at any time."""
        if device_type == 'keyboard':
            action_map_vals = Keyboard.ACTION_MAP.values()
            self.dev = Keyboard()
        else:
            raise NotImplementedError("Only implemented for keyboard for now")

        for b in Device.BUTTONS:
            if b not in action_map_vals:
                raise KeyError(f"Missing key for device_type {device_type}, Device class needs {Device.BUTTONS},"
                               f"but {device_type} only has {action_map_vals}")

        self.buttons = dict.fromkeys(Device.BUTTONS)
        for b in self.buttons:
            self.buttons[b] = Button(hold_time_length=hold_time)

        self._thread_lock = Lock()

        if device_thread:
            self._thread_sleep_time = 1/thread_freq
            self._q = queue.Queue()
            self._thread = Thread(target=self._worker, args=(self._q,))
            self._thread.daemon = True  # background thread
            self._thread.start()
            signal.signal(signal.SIGINT, self._signal_handler)
        else:
            self._thread = None

    def update_buttons(self):
        """ Update all button states. Should be called once per iteration, or is called automatically
            in a thread. """
        self._thread_lock.acquire()
        self.dev.process_events()
        cur_time = time.time()

        for b in self.dev.btn_state:
            self.buttons[b].get_and_store_update(self.dev.btn_state[b], cur_time)

        self._thread_lock.release()

    def get_latest_button_data(self):
        """ Called by user when they want current states of buttons """
        ret_dict = dict()
        self._thread_lock.acquire()
        for k in self.buttons:
            ret_dict[k] = copy.deepcopy(self.buttons[k].stored_state)
            # if any(ret_dict[k][edge] for edge in ret_dict[k].keys()):
            self.buttons[k].reset_state()  # always reset state when user checks
        self._thread_lock.release()
        return ret_dict

    def reset_all_button_states(self):
        for k in self.buttons:
            self.buttons[k].reset_state()

    def _worker(self, q: queue.Queue):
        while(True):
            try:
                q_data = q.get_nowait()
                if q_data == 'shutdown':
                    print('shutting down device thread')
                    break
            except queue.Empty:
                pass
            self.update_buttons()
            time.sleep(self._thread_sleep_time)

    def _signal_handler(self, sig, frame):
        self._shutdown_buttons_worker()
        sys.exit()

    def _shutdown_buttons_worker(self):
        self._q.put('shutdown')
        self._thread.join()
        print('device update thread shut down')


class CollectDevice(Device):
    def __init__(
        self,
        device_type='keyboard',
        initial_gripper_state=-1.0,
        initial_sts_state=-1.0,
    ):
        """ Wraps Device but allows definition of specific properties based on edge states of buttons in device."""
        super().__init__(device_type=device_type)
        self._initial_gripper_state = initial_gripper_state
        self._gripper_state = initial_gripper_state
        self._initial_sts_state = initial_sts_state
        self._sts_state = initial_sts_state
        self.update()

    def update(self):
        """ To be called once per environment timestep. """
        self.but_edge_dict = self.get_latest_button_data()

        if self.but_edge_dict['gripper']['fe']:
            if self._gripper_state == -1.0:
                self._gripper_state = 1.0
            else:
                self._gripper_state = -1.0

        if self.but_edge_dict['sts_switch']['fe']:
            if self._sts_state == -1.0:
                self._sts_state = 1.0
            else:
                self._sts_state = -1.0

    def reset_states(self):
        """ Called before each episode, reset state variables. """
        self._gripper_state = self._initial_gripper_state
        self._sts_state = self._initial_sts_state

    def return_on_press(self, get_att_func, loop_time=.03):
        self.reset_all_button_states()  # so we don't detect button presses that happened earlier
        while not get_att_func():
            self.update()
            time.sleep(loop_time)
        return

    def return_on_any_press(self, get_att_func_list, loop_time=.03):
        self.reset_all_button_states()  # so we don't detect button presses that happened earlier
        return_list = [False] * len(get_att_func_list)
        while not any(return_list):
            self.update()
            for i, func in enumerate(get_att_func_list):
                return_list[i] = func()
            time.sleep(loop_time)
        return return_list

    def get_error_recovery(self):
        return self.error_recovery

    def get_start_stop(self):
        return self.start_stop

    def get_reset(self):
        return self.reset

    def get_delete(self):
        return self.delete

    def get_success(self):
        return self.success

    def get_failure(self):
        return self.failure

    @property
    def error_recovery(self):
        return self.but_edge_dict['error_recovery']['fe']

    @property
    def start_stop(self):
        return self.but_edge_dict['start_stop']['fe']

    @property
    def success(self):
        return self.but_edge_dict['success']['fe']

    @property
    def failure(self):
        return self.but_edge_dict['failure']['fe']

    @property
    def reset(self):
        return self.but_edge_dict['reset']['fe']

    @property
    def delete(self):
        return self.but_edge_dict['delete']['rhe']

    @property
    def save(self):
        return self.but_edge_dict['save']['fe']

    @property
    def gripper(self):
        return self._gripper_state

    @property
    def sts_switch(self):
        return self._sts_state


if __name__ == "__main__":
    # dev = Device()

    # for i in range(20):
    #     ret_dict = dev.get_latest_button_data()
    #     print(ret_dict['save'])
    #     # print(dev.buttons['start_stop'].hold_state)
    #     # print(dev.buttons['start_stop'].hold_time_start)
    #     time.sleep(.2)


    dev = CollectDevice()

    for i in range(20):
        dev.update()
        print(f"start/stop? {dev.start_stop}")
        print(f"reset? {dev.reset}")
        print(f"delete? {dev.delete}")
        print(f"save? {dev.save}")

        time.sleep(.2)
