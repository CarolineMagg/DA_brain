########################################################################################################################
# Function for loading the data
########################################################################################################################

import nibabel as nib
import numpy as np
import os

__author__ = "c.magg"


def load_data_from_folder(data_folder, subfiles=None):
    """
    Load data from data folder
    :param data_folder: path to data folder
    :param subfiles: subfiles in data folder
    :return: list with images as np.arrays
    """

    if subfiles is None:
        subfiles = ["vs_gk_t1_refT1.nii", "vs_gk_t2_refT1.nii", "vs_gk_seg_refT1.nii"]

    path = [os.path.join(data_folder, f) for f in subfiles]
    images = [nib.load(p) for p in path]
    return [img.get_fdata() for img in images]


def load_data_from_nii_file(file_name):
    """
    Load data from nii file
    :param file_name: file name for nii file
    :return: image as np.array
    """
    image = nib.load(file_name)
    return image.get_fdata()


def get_non_zero_slices_segmentation(segmentation):
    """
    Extract all slices of segmentation that are non-zero
    :param segmentation: np.array segmentation mask
    :return: list with non-zero slice indices
    """
    non_zero_index = []
    for idx in range(0, segmentation.shape[2]):
        if np.sum(segmentation[:, :, idx]) != 0:
            non_zero_index.append(idx)
    return non_zero_index
