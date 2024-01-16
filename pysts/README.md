# pysts

*Now working on MacOS directly!*

pysts is a pure python way of running an STS sensor, as an alternative to ROS.

Specifically, if you have a usb camera and an arduino for controlling the LEDs, you can use this package to run your STS directly in python!
If you have a raspberry pi camera and want to be able to control it from an attached machine, that is not yet supported (though probably wouldn't be too hard to implement with socket.)

This is all you need to do to start using an STS:
```python
from pysts.sts import PySTS

sts = PySTS(config_dir=/path/to/config)
sts.set_mode('tactile')
while True:
    img = sts.get_image()
    # do processing here...
```

You can also do the same with a video file of sts data:
```python
from pysts.sts import SimPySTS

sts = SimPySTS(source_vid=/path/to/source)
while True:
    img = sts.get_image()
    # do processing here...
```

## Installation
```bash
git clone git@github.sec.samsung.net:SAIC-Montreal/sts-cam-ros2.git && cd sts-cam-ros2/src/sts
pip install -e .  # allows importing sts and all non-ros functions
cd /my/project/folder  # make sure not to clone this in sts-cam-ros2/src/sts!!
git clone git@github.sec.samsung.net:t-ablett/pysts.git && cd pysts
pip install -e .
```

## Usage
To run the same calibration that exists in `sts-cam-ros2`:
```bash
python -m pysts.calibrate /path/to/sts/config
```

See examples directory for more examples.

## Features
- Calibration node
- Mode switching
- Recording and playing back data
- All regular STS processing (through the installation of the `sts` package from `sts-cam-ros2`)

### Not yet supported
- Running STS on different machine (e.g. raspberry pi) from host
