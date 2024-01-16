from __future__ import print_function
import cv2 as cv
import argparse
import sys
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame, learningRate=0)


    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))


    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = None
    while keyboard not in {'c', 'q', 27}:
        # keyboard = cv.waitKey(30)
        keyboard = cv.waitKey(0)
        # print(f"KEYBOARD: {keyboard}")
        if keyboard == 113 or keyboard == 27:
            sys.exit(0)
        if keyboard == 99:
            break
