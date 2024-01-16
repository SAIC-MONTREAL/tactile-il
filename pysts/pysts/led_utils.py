


class ColorRGBA:
    def __init__(self, r=0., g=0., b=0., a=0.):
        self.r = r
        self.g = g
        self.b = b
        self.a = a


def fill_strip(vals):
    n = len(vals)
    led_strip = [ColorRGBA(0,0,0,255)] * n
    for i in range(len(vals)):
        led_strip[i] = vals[i]
    return led_strip