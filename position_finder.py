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
colorLower = (29, 40, 6)
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
        if radius > 5:
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

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

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


