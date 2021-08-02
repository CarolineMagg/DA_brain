#!/bin/bash
xhost +

docker run --gpus all -d -t --rm --name pycharm \
	    -v /home/caroline:/home/caroline \
	    -v /tmp/.X11-unix:/tmp/.X11-unix \
	    -e DISPLAY=$DISPLAY \
	    --user=caroline \
	    --entrypoint=/pch/bin/pycharm.sh \
	    python:1.00
