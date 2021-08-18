# Domain Adaptation for Brain VS segmentation



## Dataset
The dataset used for this work is publicly available in The Cancer Imaging Archive (TCIA):

Shapey, J., Kujawa, A., Dorent, R., Wang, G., Bisdas, S., Dimitriadis, A., Grishchuck, D., Paddick, I., Kitchen, N., Bradford, R., Saeed, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and Baseline Algorithm [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.9YTJ-5Q73

### How to extract the data from TCIA and convert it to format used in this repo

Extracting data:
1. download data from [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053): <br>
Images and Radiation Therapy Structures, registration matrices, contours
2. extract data with NBIA Data Retriever with 'Descriptive Directory format'.
3. follow [instructions](https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing) to:
* convert the original folder structure to a more convenient folder structure and file names
* convert DICOM images and segmentations into NIFTI files

This structure is required since the `DataContainer` and the different `DataSet2D` are tailored to deal with it.

### Folder structure

The data folder should be called `data/VS_segm/VS_registered` and should have the following hierarchy:
```
.
+-- tf
|   +-- workdir
|       +-- DA_brain
|           this repository
|       +-- data/VS_segm/VS_registered
|           +-- training
|           +-- test
|           +-- validation       
```

Some run scripts contain the relative paths as outlined above.

### Data split

The split training/test split was 80:20 and the training data is split again into 80:20. 

| Dataset    | # samples | numbers (excl.)          |
| ---------- |:---------:| ------------------------:|
| train      | 154       | 1 - 157 (39,97,130)      |
| validation | 38        | 158 - 197 (160,168)      |
| test       | 50        | 198 - 250 (208,219,227)  |


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
