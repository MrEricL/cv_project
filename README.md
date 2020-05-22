# cv_project

Air drums with created with HSV thresholding.

## Proposal

We are attempting to link gesture recognition with music. Depending on region/location and other factors, the project should ideally mirror playing air drums or air piano with notes modulated by the users gestures. We have also experimented with different levels of optimization from neural nets to aKalmanfilter.

## Usage

To run with a webcam simply do `python3 ./position_finder.py` and please allow a few seconds to start up

To run with a video do `python3 ./position_finder.py --video [video_path]`

When running you may press the `Q` key to quit.

Put hand in the upper right box and make gestures when playing the air drums to modulate the sound made.