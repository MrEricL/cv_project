# USAGE
# python track.py --video video/sample.mov

# import the necessary packages
import numpy as np
import argparse
import cv2

# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
roiPts = []
inputMode = False

def center(points):
    x = np.float32((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0)
    y = np.float32((points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)


def selectROI(event, x, y, flags, param):
	# grab the reference to the current frame, list of ROI
	# points and whether or not it is ROI selection mode
	global frame, roiPts, inputMode

	# select the ROI if less than four points
	if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
		roiPts.append((x, y))
		cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
		cv2.imshow("frame", frame)

def main():
	#init Kalman
	kalman = cv2.KalmanFilter(4,2)

	dt = 1 #step interval
	kalman.measurementMatrix = np.array([[1,0,0,0],
	                                     [0,1,0,0]],np.float32)

	kalman.transitionMatrix = np.array([[1,0,dt,0],
	                                    [0,1,0,dt],
	                                    [0,0,1,0],
	                                    [0,0,0,1]],np.float32)

	kalman.processNoiseCov = np.array([[1,0,0,0],
	                                   [0,1,0,0],
	                                   [0,0,1,0],
	                                   [0,0,0,1]],np.float32) * 0.01 #smoothing

	f = 0.1845 #from paper
	kalman.measurementNoiseCov = np.array([[f,f/40],
	                                       [f/40,f/4]],np.float32)

	prediction = np.zeros((4,1), np.float32)


	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help = "path to the (optional) video file")
	args = vars(ap.parse_args())

	# grab the reference to the current frame, list of ROI
	# points and whether or not it is ROI selection mode
	global frame, roiPts, inputMode

	# if the video path was not supplied, grab the reference to the
	# camera
	if not args.get("video", False):
		camera = cv2.VideoCapture(0)

	# otherwise, load the video
	else:
		camera = cv2.VideoCapture(args["video"])

	# setup the mouse callback
	cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
	cv2.setMouseCallback("frame", selectROI)

	# initialize the termination criteria for cam shift, indicating
	# a maximum of ten iterations or movement by a least one pixel
	# along with the bounding box of the ROI
	termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
	roiBox = None

	# keep looping over the frames
	while True:
		# grab the current frame
		(grabbed, frame) = camera.read()

		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break

		# if the see if the ROI has been computed
		if roiBox is not None:
			# convert the current frame to the HSV color space
			# and perform mean shift
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

			# apply cam shift to the back projection, convert the
			# points to a bounding box, and then draw them
			(r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
			pts = np.int0(cv2.boxPoints(r))
			#cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

			print(r)
			print(roiBox)
			print(type(pts))
			print(type([pts]))


			# draw observation on image - in BLUE
			x,y,w,h = roiBox
			frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

			# extract center of this observation as points

			pts = cv2.boxPoints(r)
			pts = np.int0(pts)
			# (cx, cy), radius = cv2.minEnclosingCircle(pts)

			# use to correct kalman filter
			kalman.correct(center(pts))

			# get new kalman filter prediction
			# print(prediction, "\n")

			prediction = kalman.predict()

			#cv2.polylines(frame, list(prediction[0]-(0.5*w),prediction[1]-(0.5*h)), (prediction[0]+(0.5*w),prediction[1]+(0.5*h)), True, (255, 0, 0), 2)
			# draw predicton on image - in GREEN
			frame = cv2.rectangle(frame, (prediction[0]-(0.5*w),prediction[1]-(0.5*h)), (prediction[0]+(0.5*w),prediction[1]+(0.5*h)), (255,0,0),2)


		# show the frame and record if the user presses a key
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# handle if the 'i' key is pressed, then go into ROI
		# selection mode
		if key == ord("i") and len(roiPts) < 4:
			# indicate that we are in input mode and clone the
			# frame
			inputMode = True
			orig = frame.copy()

			# keep looping until 4 reference ROI points have
			# been selected; press any key to exit ROI selction
			# mode once 4 points have been selected
			while len(roiPts) < 4:
				cv2.imshow("frame", frame)
				cv2.waitKey(0)

			# determine the top-left and bottom-right points
			roiPts = np.array(roiPts)
			s = roiPts.sum(axis = 1)
			tl = roiPts[np.argmin(s)]
			br = roiPts[np.argmax(s)]

			# grab the ROI for the bounding box and convert it
			# to the HSV color space
			roi = orig[tl[1]:br[1], tl[0]:br[0]]
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

			# compute a HSV histogram for the ROI and store the
			# bounding box
			roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
			roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
			roiBox = (tl[0], tl[1], br[0], br[1])

		# if the 'q' key is pressed, stop the loop
		elif key == ord("q"):
			break

	# cleanup the camera and close any open windows
	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()