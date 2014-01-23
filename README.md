Capsulized videos!
==================

Main program is opticalflowstitching and a helper graph-making-program
is shotanalysis.

Capsulation is divided to two programs: shotAnalysis and opticalFlowStitching.
Shot analyse separates different shots of a video for the stitcher-program.
Stitcher makes the final images.

Compile:
+ make shotanalysis
+ make stitching

To compile you need newest version of opencv!
( For Ubuntu check https://help.ubuntu.com/community/OpenCV )

Example case
------------
An example video is viewable at http://youtu.be/0Kjki0VJf1E

Shotanalysis graphical output for example video
![Motion detected from example video](http://granite.dy.fi/jafna/kandipics/graafimotion.png "Motion detected from example video")
Motion spikes on the graph shows us the shot boundaries.

Stitcher then makes representative image for each shot detected.
![Because camera moves in first shot, resulting image is panorama](http://granite.dy.fi/jafna/kandipics/shot1.png)

![Detected moving objects are blended to the final image](http://granite.dy.fi/jafna/kandipics/shot2.png)

![In last shot movement was too small so only 1 frame was selected from shot](http://granite.dy.fi/jafna/kandipics/shot3.png)

