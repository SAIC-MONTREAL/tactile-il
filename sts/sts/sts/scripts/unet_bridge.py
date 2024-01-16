_PYTHON_UNET_PATH='/home/jenkin/ros2_ws/sts-data/Pytorch-Unet-STS/pytorch_unet'
print(_PYTHON_UNET_PATH)
import sys
sys.path.append(_PYTHON_UNET_PATH)

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

from unet.unet_model import UNet

#
# This has been extracted from JF's code
#
class UNetInference:
    def __init__(self, n_channels=3, n_classes=2, model=None, scale_factor=1):
        print(f"Unetinference called n_channels={n_channels} n_classes={n_classes} scale_factor={scale_factor} model={model}")
        self.model = model
        self.net = UNet(n_channels=n_channels, n_classes=n_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load(self.model, map_location=self.device))
        self.net.eval()
        self.scale_factor = scale_factor
        print("Unetinference init complete")

    def preprocess(self, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"
        pil_img = pil_img.resize( (newW, newH), resample=Image.NEAREST )
        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255

        return img_ndarray


    def predict(self, full_img):
        print("Unet predicting")
        # 1. Convert img to PIL
        full_img = Image.fromarray(full_img.astype("uint8"), "RGB")
        img = torch.from_numpy(
             self.preprocess(full_img, self.scale_factor)
#            STSDataset.preprocess(full_img, self.scale_factor, is_mask=False)
        )
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img)

            if self.net.n_classes > 1:
                probs = F.softmax(output, dim=1)[0]
            else:
                probs = torch.sigmoid(output)[0]

            if self.scale_factor != 1:
                tf = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((full_img.size[1], full_img.size[0])),
                        transforms.ToTensor(),
                    ]
                )
            else:
                tf = lambda x: x

            full_mask = tf(probs.cpu()).squeeze()

        if self.net.n_classes == 1:
            return (full_mask > 0.5).numpy()
        else:
            return (
                F.one_hot(full_mask.argmax(dim=0), self.net.n_classes)
                .permute(2, 0, 1)
                .numpy()
            )


