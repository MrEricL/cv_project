# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import asyncio

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# Color threshold in  HSV color space
# Currently set to green-yellow
colorLower = (29, 86, 6)
colorUpper = (64, 255, 255)

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)


while True:

    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #morphological transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    pre_morph_HSV = HSV_mask
    HSV_mask = cv2.erode(HSV_mask,kernel)
    HSV_mask = cv2.dilate(HSV_mask,kernel)



    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    # YCrCb_mask = cv2.inRange(img_YCrCb, (60, 100, 135), (250,125,170)) 

    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #morphological transformations
    pre_morph_YCrCb = YCrCb_mask
    YCrCb_mask = cv2.erode(YCrCb_mask,kernel)
    YCrCb_mask = cv2.dilate(YCrCb_mask,kernel,iterations = 1)


    #merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask = cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)

   #2 merge skin detection (YCbCr and hsv)
    global_mask2 = cv2.bitwise_and(pre_morph_HSV, pre_morph_YCrCb)
    global_mask2 = cv2.medianBlur(global_mask2,3)
    global_mask2 = cv2.morphologyEx(global_mask2, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    HSV_result2 = cv2.bitwise_not(pre_morph_HSV)
    YCrCb_result2 = cv2.bitwise_not(pre_morph_YCrCb)
    global_result2 =cv2.bitwise_not(global_mask2)

    #image concatenation 
    numpy_horiz_concat = np.concatenate((img_YCrCb, img_HSV), axis=1)
    numpy_horiz_concat2 = np.concatenate((YCrCb_result, HSV_result), axis=1)
    numpy_horiz_concat3 = np.concatenate((cv2.cvtColor(global_result2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(global_result, cv2.COLOR_GRAY2BGR) ), axis=1)
    numpy_horiz_concat4 = np.concatenate((pre_morph_YCrCb, pre_morph_HSV), axis=1)

    numpy_horiz_concat2 = cv2.cvtColor(numpy_horiz_concat2, cv2.COLOR_GRAY2BGR)
    numpy_horiz_concat4 = cv2.cvtColor(numpy_horiz_concat4, cv2.COLOR_GRAY2BGR)


    # result = np.concatenate((numpy_horiz_concat, numpy_horiz_concat2), axis=0)
    numpy_horiz_concat = cv2.resize(numpy_horiz_concat, (0, 0), None, .25, .25)
    numpy_horiz_concat2 = cv2.resize(numpy_horiz_concat2, (0, 0), None, .25, .25)
    numpy_horiz_concat3 = cv2.resize(numpy_horiz_concat3, (0, 0), None, .25, .25)
    numpy_horiz_concat4 = cv2.resize(numpy_horiz_concat4, (0, 0), None, .25, .25)



    result = np.concatenate((numpy_horiz_concat, numpy_horiz_concat2), axis=0)
    result = np.concatenate((result, numpy_horiz_concat4), axis=0)
    result = np.concatenate((result, numpy_horiz_concat3), axis=0)


    # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

    # show the frame to our screen
    cv2.imshow("Frame", result)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()


