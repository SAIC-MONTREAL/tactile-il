import os
from sts.scripts.helper import read_json

class ContactDetectionCreator:
    """The Factory Class"""
    def __init__(self, config_dir):
        self.config_dir = config_dir

    def create_object(self,):
        tmp = os.path.join(self.config_dir, "tactile.json")
        tactile_config = read_json(tmp)

        """A static method to get a concrete product"""
        if tactile_config['contact_detection'] == "back_sub":
            from sts.scripts.contact_detection.contact_detection_back_sub import ContactDetectionBackSub
            return ContactDetectionBackSub(self.config_dir)
        if tactile_config['contact_detection'] == "depth":
            from sts.scripts.contact_detection.contact_detection_depth import ContactDetectionDepth
            return ContactDetectionDepth(self.config_dir)


class ContactDetection(object):
    """Process optical flow on the STS"""

    def __init__(self, ):
        """Config is a dictionary of configuration parameters"""
        pass

    def detect(self, img):
        """The actual detection process defined by sub-classes"""
        raise NotImplementedError()



