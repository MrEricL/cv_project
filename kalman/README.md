# Kalman and One Euro Filter

The Kalman and One Euro Filter are two methods of attempting to track the location and trajectory of an object.

The Kalman filter does so through analyzing the previous and current values to make an estimation. It uses the joint probability distributions to calculate and is slower. 

The One Euro Filter is a low pass filter more used for noisy signals in real time. It is much faster but at the cost of long-term consistency. 


## Description
This file allows you to compare four different types of tracking:
1. OpenCV's camshift [GREEN]
2. OpenCV's camshift using dynamic polylines [YELLOW]
3. Kalman filter [BLUE]
4. One Euro filter [BLACK]

## Usage

To use the webcam run `python track.py`

To use a video run `python track.py -v [VIDEONAME.EXT]`

1. Press `I` and select the four corners of an object
2. Press any key and watch the results
3. Press `Q` to quit

In the first few lines of code, you can choose which boxes you want to see, and the parameters for the filters.

## Files

`track.py` the main driver file with everything
`OneEuro.py` the One Euro filter code
`test.mov` a sample video to test on (likely compressed/sped up due to Github)


## Sources

The Camshift algorithm is sourced from Adrian Roseback's [blog](pyimagesearch.com). 

The Kalman filter was based off this paper from the [Navy](https://www.nrl.navy.mil/itd/imda/sites/www.nrl.navy.mil.itd.imda/files/pdfs/cp_GESTURE03.pdf) and this university research [paper](https://www.researchgate.net/publication/221230574_Hand_gesture_tracking_system_using_Adaptive_Kalman_Filter).

