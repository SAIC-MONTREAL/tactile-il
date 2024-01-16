import json
import sys

from pysts.led_utils import ColorRGBA, fill_strip


COLOR_MAP = {
    "black": [0, 0, 0, 255],
    "off": [0, 0, 0, 255],
    "white": [255, 255, 255, 255],
    "half": [128, 128, 128, 255],
    "quarter": [64, 64, 64, 255],
    "red": [255, 0, 0, 255],
    "green": [0, 255, 0, 255],
    "blue": [0, 0, 255, 255],
}

def get_pattern_strip(config_dir, pattern='white'):
    with open(f"{config_dir}/tactile.json") as fd:
        config = json.load(fd)
        strip_length = config['strip_len']

    print(f'Static: pattern {pattern} length {strip_length}')

    if pattern in COLOR_MAP:
        strip = [ColorRGBA(*COLOR_MAP[pattern])] * strip_length
    elif pattern == "3colors":
        length = int(strip_length/3)
        strip = [ColorRGBA(0, 0, 80, 0)] * length + \
                [ColorRGBA(0, 80, 0, 0)] * length + \
                [ColorRGBA(80, 0, 0, 0)] * (strip_length - 2 * length)
    elif isinstance(pattern, int):
        pos = int(pattern)
        if (pos >= 0) and (pos < strip_length):
            s = [ColorRGBA(0, 0, 0, 255)] * strip_length
            s[pos] =  ColorRGBA(255, 255, 255, 255)
            strip = s
        else:
            print(f"Static: int pattern {pattern} invalid")
            sys.exit(0)
    else:
        print(f"Static: don't know pattern {pattern}")
        sys.exit(0)

    return strip