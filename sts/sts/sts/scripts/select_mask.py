# importing the module
import cv2
import argparse
import numpy as np
from PIL import Image
from sts.scripts.helper import write_json, warp_image

class MaskSelector(object):
    # function to display the coordinates of
    # of the points clicked on the image
    def __init__(self):
        self.coord_list = []

    def click_event(self, event, x, y, flags, params):
        img = self.img
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
     
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            self.coord_list.append([x,y])
     
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (1, 0, 0), 2)
            cv2.imshow('image', self.img)
     
class RectangularMask(MaskSelector):
    def __init__(self):
        super().__init__()

    def create_mask(self, img):
        h = img.shape[0]
        w = img.shape[1]

        import pylab as plt
        import numpy as np
        from matplotlib.path import Path

        polygon = self.coord_list
        poly_path=Path(polygon)

        x, y = np.mgrid[:h, :w]
        coors=np.hstack((y.reshape(-1, 1), x.reshape(-1,1))) # coors.shape is (4000000,2)

        mask = poly_path.contains_points(coors).reshape(h, w)


        mask = np.where(mask == True, 255, 0).astype('float32')
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha = 0.0

        input_pts = np.float32(self.coord_list)
        output_pts = np.float32([[0, 0], [0, h-1], [w- 1, h -1], [w-1, 0]])
        self.M = cv2.getPerspectiveTransform(input_pts, output_pts)
        cv2.imshow('dst', warp_image(img, self.M))
        cv2.waitKey(0)
        return mask

    def select_mask(self, ref_img, output_path, img_pre=""):
        self.img = ref_img
        cv2.imshow('image', ref_img)
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)

        mask = self.create_mask(ref_img)
        cv2.imwrite(output_path + "/" + img_pre + "mask.png", mask)
        write_json(output_path + "/warp_matrix.json", {'M':self.M.tolist()})
        cv2.destroyAllWindows()


class CircularMask(MaskSelector):
    def __init__(self):
        super().__init__()

    def create_mask(self, img):
        center = np.array(self.coord_list[0])
        radius = np.linalg.norm(np.array(self.coord_list[1]) - np.array(self.coord_list[0]) )
        h = img.shape[0]
        w = img.shape[1]

        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        mask = np.where(mask == True, 255, 0).astype('float32')
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha = .5
        dst = cv2.addWeighted(mask_rgb.astype('uint8'), alpha, img.astype('uint8'), 1-alpha, 0.0)
        cv2.imshow('dst', dst)
        cv2.waitKey(0)
        return mask

    def select_mask(self, ref_img, output_path, img_pre=""):
        self.img = ref_img
        cv2.imshow('image', ref_img)
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)

        mask = self.create_mask(ref_img)
        cv2.imwrite(output_path + "/" + img_pre + "mask.png", mask)
        cv2.destroyAllWindows()

if __name__=="__main__":
 
    parser = argparse.ArgumentParser(description='Parameters to create sts mask of sensorized area.')
    parser.add_argument('--type', type=str, default='rectangular',
                        help='calibrate internal circle area')
    parser.add_argument('--reference', default='/home/ros2/ros2_ws/configs/sts_usb/reference.png', help='source file name')
    parser.add_argument('--output', default='./sts/mask.png', help='output file name')

    vargs = vars(parser.parse_args())
    import matplotlib.pyplot as plt
    img = plt.imread(vargs['reference'])
    if vargs['type']=='rectangular':
        mask_selector = RectangularMask()
        mask_selector.select_mask(img, vargs['output'])
    elif vargs['type']=='circle':
        mask_selector = CircularMask()
        mask_selector.select_mask(img, vargs['output'])
