# Domain Adaptation for Brain VS segmentation

This work was done as part of a master thesis.

Two domain adaptation frameworks that utilize image alignment with CycleGAN [1] and supervised VS segmentation networks with X-Net [2], which can be interpretated as two U-Nets with skip connections in series. Both approaches are enhanced with a classification-guided module [3] that trains classification and segmentation together. In addition, an image and feature alginment framework based on SIFA [4] with classification-guided module enhancement was designed.

![alt text](https://github.com/CarolineMagg/DA_brain/blob/main/domain_adaptation.png)

[1] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation
using cycle-consistent adversarial networks. In 2017 IEEE International Conference
on Computer Vision (ICCV), pages 2242–2251, 2017.

[2] J. Bullock, C. Cuesta-Lázaro, and A. Quera-Bofarull. XNet: a convolutional neural
network (CNN) implementation for medical X-Ray image segmentation suitable
for small datasets. In Medical Imaging 2019: Biomedical Applications in Molecular,
Structural, and Functional Imaging, volume 10953, pages 453 – 463. International
Society for Optics and Photonics, SPIE, 2019.

[3] H. Huang, L. Lin, R. Tong, H. Hu, Q. Zhang, Y. Iwamoto, X. Han, Y.-W. Chen, and
J. Wu. UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.
In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pages 1055–1059, 2020.

[4] C. Chen, Q. Dou, H. Chen, J. Qin, and P. A. Heng. Unsupervised Bidirectional
Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for
Medical Image Segmentation. IEEE Transactions on Medical Imaging, 39(7):2494–
2505, 2020.

## Dataset
The dataset used for this work is publicly available in The Cancer Imaging Archive (TCIA):

Shapey, J., Kujawa, A., Dorent, R., Wang, G., Bisdas, S., Dimitriadis, A., Grishchuck, D., Paddick, I., Kitchen, N., Bradford, R., Saeed, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and Baseline Algorithm [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.9YTJ-5Q73

### How to extract the data from TCIA and convert it to format used in this repo

Extracting data:
1. download data from [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053): <br>
Images and Radiation Therapy Structures, registration matrices, contours
2. extract data with NBIA Data Retriever with 'Descriptive Directory format'.
3. follow [instructions](https://github.com/KCL-BMEIS/VS_Seg/tree/master/preprocessing) in order to:
* convert the original folder structure to a more convenient folder structure and file names
* convert DICOM images and segmentations into NIFTI files
* register T2 images to T1 images

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

Some run scripts and notebooks contain the relative/absolute paths as outlined above.

### Data split

The split training/test split was 80:20 and the training data is split again into 80:20. The split can be performed by calling `data_utils/GenerateSplit.py`.

| Dataset    | # samples | numbers (excl.)          |
| ---------- |:---------:| ------------------------:|
| train      | 155       | 1 - 158 (39,97,130)      |
| validation | 39        | 159 - 199 (160,168)      |
| test       | 48        | 200 - 250 (208,219,227)  |

### Remove empty slices

There are empty (all-zero) slices in T1 and T2 scans. In order to remove them, call `data_utils/RemoveEmptySlices.py`. This will first create a .json file storing the first and last non-empty slice index and then, remove the empty slices in the entire volume (T1, T2, VS segmentation, Cochlea segmentation if available).

### Data Preprocessing

The data preprocessing performed in `DataSet2D` and `DataSet2DMixed` are based on data statistics and informations.
After the data has the structure discribed above, the script `data_utils/GenerateStatistics.py` generates a json file in each patient folder with the infromation necessary for preprocessing. In order to apply the script, the dataset split and the empty slice removal need to be done first.

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
