# Car Speed Detection - carspeed.py (version 4.0) Forked from version3.0

## Blog URL

Version 2.0 Full program details at:   https://gregtinkers.wordpress.com/2016/03/25/car-speed-detector/
This version https://www.hackster.io/hodgestk/traffic-camera-9d3739

## Description

This program for the Raspberry Pi determines the speed of cars moving through the Picamera's field of view. An image of the car and labeled with its speed can be saved.

## Requirements

* Raspberry Pi 3 Model B or newer
* Picamera v 2
* Opencv4
* Raspberry OS Bullseye or Bookworm (32bit or 64bit)
  
## Usage

Install OpenCV 4 and Python 3 on the Pi. 

Copy carspeed.py into the same directory as your home directory. 

Point the Picamera at the road. Before you run carspeed.py, modify the constant DISTANCE to the distance from the front of the Picamera lens to the middle of the road. You may also need to adjust the vflip and hflip to match you camera's orientation.

Before you run carspeed.py, modify the constants L2R_DISTANCE and R2L_DISTANCE to the distances from the front of the Pi camera lens to the middle of the Left to Right lane and Right to Left lane.

Modify the MIN_SPEED_IMAGE to the minimum speed in mph to capture an image. I usually set this quite high as I don't want to fill up the Pi's storage with images of cars that I'll never look at.

Modify the MIN_SPEED_SAVE to the minimum speed in mph to capture the data in the CSV file and pass to the MQTT broker. I usually set this to about 10 mph so I don't record pedestrians.

Modify MAX_SPEED_SAVE to the max speed to record. Sometimes when two vehicles cross it can lead to crazy speed numbers.

You may also need to adjust the vflip and hflip to match your camera’s orientation depending on the orientation of your camera sensor.

Run from a terminal with: python3 carspeed.py -ulx 64 -uly 321 -lrx 960 -lry 444.  
The above passes in the coordinates (in pixels) of the upper left and lower right of the monitoring area.  You can adjust them to suit your own requirements.


As cars pass through the monitored area, an image will be written to disk with the speed if it is travelling above MIN_SPEED_IMAGE. If the speed is between MIN_SPEED_SAVE and MAX_SPEED_SAVE then data will be recorded to a CSV file.

Exit the program with a press of the 'q' key.

## License:

(The MIT License)

Copyright (c) 2020 Tim Hodges

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

 	You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
