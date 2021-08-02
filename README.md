# Domain Adaptation for Brain VS segmentation

## Dataset
TBD

## Docker & Requirements
You can use [Docker](https://www.docker.com/) to setup your environment. For installation guide see [Install Docker](https://docs.docker.com/get-docker/). <br> 

The docker image contains (minimal requirements):
* Python 3.6
* Tensorflow 2.4.1 
* Jupyter notebooks
* Pycharm (installer.tgz)

### How to build the docker image:
1. clone the github repository 
2. go to Dockerfile: ``` cd dockerfile/ ```
3. change Dockerfile to use your user name instead of *caroline* 
4. either download pycharm and store in installer.tgz or remove corresponding part in Dockerfile
5. Build docker image: ``` docker build --tag python:1.00 .``` 

### How to run the docker container:
(Note: Change *user* to your home folder name.)
* Run jupyter docker container: <br>
``` docker run -it --gpus all --name jupyter_notebook --rm -v /home/user/:/tf/workdir -p 8888:8888 python-tf-docker:1.00 ``` <br>

Recommended: modify sh files to fit your settings
