import sys
import termios
import tty


KEY_MOTION_ACT_MAP = {
    'u': [1, 0, 0, 0, 0, 0, 0],
    'j': [-1, 0, 0, 0, 0, 0, 0],
    'i': [0, 1, 0, 0, 0, 0, 0],
    'k': [0, -1, 0, 0, 0, 0, 0],
    'o': [0, 0, 1, 0, 0, 0, 0],
    'l': [0, 0, -1, 0, 0, 0, 0],
    '7': [0, 0, 0, 1, 0, 0, 0],
    '4': [0, 0, 0, -1, 0, 0, 0],
    '8': [0, 0, 0, 0, 1, 0, 0],
    '5': [0, 0, 0, 0, -1, 0, 0],
    '9': [0, 0, 0, 0, 0, 1, 0],
    '6': [0, 0, 0, 0, 0, -1, 0],
    'n': [0, 0, 0, 0, 0, 0, 1],
    'm': [0, 0, 0, 0, 0, 0, -1],
    '0': [0, 0, 0, 0, 0, 0, 0],
}


class BlockingKeyboard:
    def __init__(self) -> None:
        self.settings = self.save_terminal_settings()

    def get_key(self):
        if sys.platform == 'win32':
            key = msvcrt.getwch()
        else:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            if (key == '\x03'):  # for handling ctrl-c
                sys.exit(0)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def wait_for_key(self, key):
        pressed_key = None
        while pressed_key != key:
            pressed_key = self.get_key()
        return

    def save_terminal_settings(self):
        if sys.platform == 'win32':
            return None
        return termios.tcgetattr(sys.stdin)

    def restore_terminal_settings(self):
        if sys.platform == 'win32':
            return
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def __del__(self):
        self.restore_terminal_settings()