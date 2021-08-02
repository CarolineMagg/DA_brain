docker run -it --gpus all --name jupyter_notebook --rm \
        	-v /home/caroline/Documents/DiplomaThesis:/tf/workdir \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
	       	-p 8888:8888 \
		-p 6006:6006 \
		-e DISPLAY=$DISPLAY \
	       	python:1.00
