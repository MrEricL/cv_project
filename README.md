# cv_project

Air drums with created with HSV thresholding.

## Proposal

We are attempting to link gesture recognition with music. Depending on region/location and other factors, the project should ideally mirror playing air drums or air piano with notes modulated by the users gestures. The first portion involved combining several algorithms to properly detect the hand and gesture. 

We have also experimented with different levels of optimization such as a Kalman filter. This second portion explored using a key filter as well as exploring a novel proposed filter for high lag scenarios.

## Objectives

* Detect object motion
* Detect a hand (multiple ways)
* Detect specific gestures based
* Add alternative estimators to optimize
* Play some music

## Usage

To run with a webcam simply do `python3 ./position_finder.py` and please allow a few seconds to start up

To run with a video do `python3 ./position_finder.py --video [video_path]`

When running you may press the `Q` key to quit.

Put hand in the upper right box and make gestures when playing the air drums to modulate the sound made.


To explore the Kalman filter and One Eurofilter `cd kalman` and read the instructions. The code is completely modular but was not included to prevent it from being too messy.
