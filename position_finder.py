# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import play_audio
import asyncio


# Flags that check wether or not to play a sound.
flagA = True
flagB = True
flagC = True
flagD = True
flagE = True
flagF = True

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
backgroundSubtractionThreshold = 50
backgroundSubtractionModelLearningRate = 0

# variables
isBgCaptured = False   # bool, whether the background captured
triggerSwitch = False  # if true, Ouput Fingercount
skinRecognitionMode = False


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
#colorLower = (29, 86, 6)
colorLower = (40, 40, 6)
#colorUpper = (64, 255, 255)
colorUpper = (100, 255, 255)

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

    # GESTURE REC
    # MAKE A Copy of the for bg subs and skin rec
    # smoothing filter
    frame2 = cv2.bilateralFilter(frame, 5, 50, 100)  
    frame2 = cv2.flip(frame2, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0), (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    frame2 = cv2.resize(frame2, (0, 0), None, .45, .45)
    fingerCount = gestureRec(frame2)
    if triggerSwitch:
        print(fingerCount)

    
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color threshold, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the elipsise 
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        #Uses moments to find the center of object
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #print(center)

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            #cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        
    # Draws rectangles
    cv2.rectangle(frame, (frame.shape[1]-100, frame.shape[0]-100), (frame.shape[1]-0, frame.shape[0]-2), (0,0,255), 2)
    cv2.rectangle(frame, (frame.shape[1]-102, frame.shape[0]-2), (frame.shape[1]-200, frame.shape[0]-100), (0,225,0), 2)
    cv2.rectangle(frame, (frame.shape[1]-202, frame.shape[0]-2), (frame.shape[1]-300, frame.shape[0]-100), (225,0,0), 2)
    cv2.rectangle(frame, (frame.shape[1]-302, frame.shape[0]-2), (frame.shape[1]-400, frame.shape[0]-100), (225,225,0), 2)
    cv2.rectangle(frame, (frame.shape[1]-402, frame.shape[0]-2), (frame.shape[1]-500, frame.shape[0]-100), (0,225,225), 2)
    cv2.rectangle(frame, (frame.shape[1]-502, frame.shape[0]-2), (frame.shape[1]-600, frame.shape[0]-100), (225,0,225), 2)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    #Sound code


    if center is not None:
        if center[0] <= frame.shape[1]-0 and center[0] >= frame.shape[1]-100 and center[1] <= frame.shape[0]-2 and center[1] >= frame.shape[0]-100:
            if flagA == True:
                flagA = False
                play_audio.playA()
        else:
            flagA = True
        
        if center[0] <= frame.shape[1]-102 and center[0] >= frame.shape[1]-200 and center[1] <= frame.shape[0]-2 and center[1] >= frame.shape[0]-100:
            if flagB == True:
                flagB = False
                play_audio.playB()
        else:
            flagB = True

        if center[0] <= frame.shape[1]-202 and center[0] >= frame.shape[1]-300 and center[1] <= frame.shape[0]-2 and center[1] >= frame.shape[0]-100:
            if flagC == True:
                flagC = False
                play_audio.playC()
        else:
            flagC = True

        if center[0] <= frame.shape[1]-302 and center[0] >= frame.shape[1]-400 and center[1] <= frame.shape[0]-2 and center[1] >= frame.shape[0]-100:
            if flagD == True:
                flagD = False
                play_audio.playD()
        else:
            flagD = True
        
        if center[0] <= frame.shape[1]-402 and center[0] >= frame.shape[1]-500 and center[1] <= frame.shape[0]-2 and center[1] >= frame.shape[0]-100:
            if flagE == True:
                flagE = False
                play_audio.playE()
        else:
            flagE = True

        if center[0] <= frame.shape[1]-502 and center[0] >= frame.shape[1]-600 and center[1] <= frame.shape[0]-2 and center[1] >= frame.shape[0]-100:
            if flagF == True:
                flagF = False
                play_audio.playF()
        else:
            flagF = True
                

    #counter += 1

    # if the 'q' or ESC key is pressed, stop the loop
    if key == ord("q") or key == 23:
        break
        k = cv2.waitKey(10)
    elif key == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, backgroundSubtractionThreshold)
        isBgCaptured = True
        print( 'Background Captured')
    elif key == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = False
        print ('Reset BackGround')
    elif key == ord('t'):
        triggerSwitch = True
        print ('Trigger On')
    elif key == ord('s'):
        skinRecognitionMode = True
        print ('Skin Rec')

    if center is not None:    
        cv2.putText(frame, "x: {}, y: {}".format(center[0], center[1]),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)

    # show the frame to our screen and increment the frame counter
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()




def ConvexHull(thresh):
    thresh1 = copy.deepcopy(thresh)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        cv2.imshow('output', drawing)
        isFinishCal,cnt = calculateFingers(res,drawing)
        if triggerSwitch and isFinishCal:
            return cnt


def SkinRec(frame):
    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    # YCrCb_mask = cv2.inRange(img_YCrCb, (60, 100, 135), (250,125,170)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    #merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask = cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    res = cv2.bitwise_and(frame, frame, mask=global_mask)
    return res



def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=backgroundSubtractionModelLearningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):
            cnt = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if angle <= math.pi / 2:  # angle less than 90 degree are fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

def gestureRec(frame):
    #  Main operation
    if isBgCaptured:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]), int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)
        cnt = ConvexHull(thresh)
        return cnt
    elif skinRecognitionMode:
        img = SkinRec(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]), int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = gray
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)
        cnt = ConvexHull(thresh)
        return cnt

