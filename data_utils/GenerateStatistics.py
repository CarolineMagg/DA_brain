########################################################################################################################
# Script to generate additional information about T1 slices that are empty
# will generate a txt file in the dataset folder called "vs_gk_t1_info.txt" including the first non-empty slice index
########################################################################################################################

import logging
import os
import cv2
import numpy as np

from data_utils.DataContainer import DataContainer
from data_utils.DataSet2D import DataSet2D

########################################################################################################################
# DataContainer to load nifti files in patient data folder
########################################################################################################################

import numpy as np
import os.path
import nibabel as nib
import json

__author__ = "c.magg"


class DataContainer:
    """
    DataContainer is a container for a nifti folders with different information like:
    * T1
    * T2
    * segmentation of VS/cochlea
    """

    def __init__(self, dir_path):
        """
        Create a new DataContainer object.
        :param dir_path: path to nifti directory with t1, t2, vs and cochlea segmentation file
        """
        self._path_dir = dir_path
        files = os.listdir(dir_path)
        self._path_t1 = os.path.join(dir_path, "vs_gk_t1_refT2.nii")
        self._path_t2 = os.path.join(dir_path, "vs_gk_t2_refT2.nii")
        self._path_vs = os.path.join(dir_path, [f for f in files if "vs_gk_struc1" in f][0])
        self._path_cochlea = None
        if len([f for f in files if "vs_gk_struc2" in f]) != 0:
            self._path_cochlea = os.path.join(dir_path, [f for f in files if "vs_gk_struc2" in f][0])
        self._path_data_info = os.path.join(dir_path, "vs_gk_t1_info.txt")
        self._data_t1 = None
        self._data_t2 = None
        self._data_vs = None
        self._data_cochlea = None
        self.load()

    def load(self):
        """
        (Re)Load the data from nifti paths.
        """
        self._data_t1 = nib.load(self._path_t1)
        self._data_t2 = nib.load(self._path_t2)
        self._data_vs = nib.load(self._path_vs)
        self._data_cochlea = nib.load(self._path_cochlea) if self._path_cochlea is not None else None

    def uncache(self):
        """
        Uncache the nifti container.
        """
        self._data_t1.uncache()
        self._data_t2.uncache()
        self._data_vs.uncache()
        self._data_cochlea.uncache()

    @property
    def t1_scan(self):
        return np.asarray(self._data_t1.dataobj, dtype=np.float32)

    @property
    def t2_scan(self):
        return np.asarray(self._data_t2.dataobj, dtype=np.float32)

    @property
    def segm_vs(self):
        return np.asarray(self._data_vs.dataobj, dtype=np.int16)

    @property
    def segm_cochlea(self):
        return np.asarray(self._data_cochlea.dataobj,
                          dtype=np.int16) if self._data_cochlea is not None else np.zeros_like(
            self._data_t1.dataobj, dtype=np.int16)

    @property
    def segm_vs_class(self):
        return [0 if np.sum(self.segm_vs_slice(idx)) == 0 else 1 for idx in range(len(self))]

    @property
    def segm_cochlea_class(self):
        return [0 if np.sum(self.segm_cochlea_slice(idx)) == 0 else 1 for idx in range(len(self))]

    def t1_scan_slice(self, index=None):
        return np.asarray(self._data_t1.dataobj[..., index], dtype=np.float32)

    def t2_scan_slice(self, index=None):
        return np.asarray(self._data_t2.dataobj[..., index], dtype=np.float32)

    def segm_vs_slice(self, index=None):
        return np.asarray(self._data_vs.dataobj[..., index], dtype=np.int16)

    def segm_vs_class_slice(self, index=None):
        return 0 if np.sum(self.segm_vs_slice(index)) == 0 else 1

    def segm_cochlea_slice(self, index=None):
        return np.asarray(self._data_cochlea.dataobj[..., index],
                          dtype=np.int16) if self._data_cochlea is not None else np.zeros_like(
            self._data_t1.dataobj[..., index])

    def segm_cochlea_class_slice(self, index=None):
        return 0 if np.sum(self.segm_cochlea_slice(index)) == 0 else 1


if __name__ == "__main__":

    paths = ["/tf/workdir/data/VS_segm/VS_registered/training/",
             "/tf/workdir/data/VS_segm/VS_registered/validation/",
             "/tf/workdir/data/VS_segm/VS_registered/test/"]
    batch_sizes = [10450 // 50, 2960 // 16, 3960 // 30]

    # extract statistics
    for path in paths:
        folders = [os.path.join(path, p) for p in os.listdir(path)]
        for folder in folders:
            statistics = {}
            fn = os.path.join(folder, "vs_gk_statistics.json")
            container = DataContainer(folder)
            ## t1 ##
            sample = container.t1_scan
            # remove empty slices
            sums = [np.sum(sample[:, :, idx]) for idx in range(np.shape(sample)[-1])]
            empty_slices = [idx for idx, s in enumerate(sums) if s == 0]
            first_nonemtpy_slice = empty_slices[-1] + 1 if len(empty_slices) != 0 else 0
            sample = sample[:, :, first_nonemtpy_slice:]
            # percentile
            percentile1 = np.percentile(sample, 1)
            percentile99 = np.percentile(sample, 99)
            sample = np.clip(sample, percentile1, percentile99)
            # z score normalization
            mean = np.mean(sample)
            std = np.std(sample)
            sample = (sample - mean) / std
            # min-max normalization
            min = sample.min()
            max = sample.max()
            sample = (sample - min) / (max - min)
            statistics["t1"] = {"first_nonempty_slice": str(first_nonemtpy_slice),
                                "1st_percentile": str(percentile1),
                                "99th_percentile": str(percentile99),
                                "mean": str(mean),
                                "std": str(std),
                                "min": str(min),
                                "max": str(max)}

            ## t2 ##
            sample = container.t1_scan
            # remove empty slices
            sums = [np.sum(sample[:, :, idx]) for idx in range(np.shape(sample)[-1])]
            empty_slices = np.where(sums == 0)[0]
            first_nonemtpy_slice = empty_slices[-1] + 1 if len(empty_slices) != 0 else 0
            sample = sample[:, :, first_nonemtpy_slice:]
            # percentile
            percentile1 = np.percentile(sample, 1)
            percentile99 = np.percentile(sample, 99)
            sample = np.clip(sample, percentile1, percentile99)
            # z score normalization
            mean = np.mean(sample)
            std = np.std(sample)
            sample = (sample - mean) / std
            # min-max normalization
            min = sample.min()
            max = sample.max()
            sample = (sample - min) / (max - min)
            statistics["t2"] = {"first_nonempty_slice": str(first_nonemtpy_slice),
                                "1st_percentile": str(percentile1),
                                "99th_percentile": str(percentile99),
                                "mean": str(mean),
                                "std": str(std),
                                "min": str(min),
                                "max": str(max)}
            with open(fn, 'w') as outfile:
                json.dump(statistics, outfile)
