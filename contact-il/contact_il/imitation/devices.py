import time

import inputs  # forked version with non-blocking gamepad


# note: state of 0 is not pressed, 1 is pressed, 2 is held

class Keyboard:
    ACTION_MAP = {
        'KEY_R': 'reset',
        'KEY_BACKSPACE': 'delete',
        'KEY_SPACE': 'start_stop',
        'KEY_RIGHTSHIFT': 'save',
        'KEY_G': 'gripper',
        'KEY_T': 'sts_switch',
        'KEY_S': 'success',
        'KEY_F': 'failure',
        'KEY_E': 'error_recovery',
    }
    def __init__(self):
        self.kb = self._get_keyboard()
        self.btn_state = dict.fromkeys(Keyboard.ACTION_MAP.values())
        self.old_btn_state = dict.fromkeys(Keyboard.ACTION_MAP.values())

    def _get_keyboard(self):
        """Get a keyboard object."""
        try:
            return inputs.devices.keyboards[0]
        except IndexError:
            raise inputs.UnpluggedError("No keyboard found.")

    def _process_event(self, event):
        """Process the event into a state."""
        if event.ev_type == 'Sync':
            return
        if event.ev_type == 'Misc':
            return
        if event.code in Keyboard.ACTION_MAP:
            self.btn_state[Keyboard.ACTION_MAP[event.code]] = bool(event.state)  # so no diff between held (2) and pushed (1)

    def process_events(self):
        """Process available events. Call this one."""
        self._set_old_button_states()
        try:
            events = self.kb.read()
        except EOFError:
            events = []
        for event in events:
            self._process_event(event)

    def _set_old_button_states(self):
        for key in self.btn_state.keys():
            self.old_btn_state[key] = self.btn_state[key]


if __name__ == "__main__":
    kb = Keyboard()

    for i in range(20):
        kb.process_events()
        print(kb.btn_state)
        time.sleep(.1)
