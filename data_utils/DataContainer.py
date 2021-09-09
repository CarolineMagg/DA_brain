########################################################################################################################
# DataContainer to load nifti files in patient data folder
########################################################################################################################
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
import nibabel as nib

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
        self._path_t1 = os.path.join(dir_path, "vs_gk_t1_refT1.nii")
        self._path_t2 = os.path.join(dir_path, "vs_gk_t2_refT1.nii")
        self._path_vs = os.path.join(dir_path, [f for f in files if "vs_gk_struc1" in f][0])
        self._path_cochlea = None
        if len([f for f in files if "vs_gk_struc2" in f]) != 0:
            self._path_cochlea = os.path.join(dir_path, [f for f in files if "vs_gk_struc2" in f][0])
        self._path_data_info = os.path.join(dir_path, "vs_gk_statistics.json")
        self._data_t1 = None
        self._data_t2 = None
        self._data_vs = None
        self._data_cochlea = None
        self._statistics = None
        self.load()

    def __len__(self):
        return self._data_t1.shape[2]

    @property
    def data(self):
        """
        Data dictionary with modality as key and data arrays as values.
        """
        return {'t1': self.t1_scan,
                't2': self.t2_scan,
                'vs': self.segm_vs,
                'cochlea': self.segm_cochlea}

    @property
    def shape(self):
        return self._data_t1.shape

    def load(self):
        """
        (Re)Load the data from nifti paths.
        """
        self._data_t1 = nib.load(self._path_t1)
        self._data_t2 = nib.load(self._path_t2)
        self._data_vs = nib.load(self._path_vs)
        self._data_cochlea = nib.load(self._path_cochlea) if self._path_cochlea is not None else None
        with open(self._path_data_info) as json_file:
            self._statistics = json.load(json_file)

    def uncache(self):
        """
        Uncache the nifti container.
        """
        self._data_t1.uncache()
        self._data_t2.uncache()
        self._data_vs.uncache()
        self._data_cochlea.uncache() if self._data_cochlea is not None else None

    @property
    def t1_scan(self):
        return self._load_item(np.asarray(self._data_t1.dataobj, dtype=np.float32), "t1")

    @property
    def t2_scan(self):
        return self._load_item(np.asarray(self._data_t2.dataobj, dtype=np.float32), "t2")

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
        return [1 if np.sum(self.segm_vs[:, :, idx]) != 0 else 0 for idx in range(0, self.segm_vs.shape[2])]

    @property
    def segm_cochlea_class(self):
        return [1 if np.sum(self.segm_cochlea[:, :, idx]) != 0 else 0 for idx in range(0, self.segm_cochlea.shape[2])]

    def t1_scan_slice(self, index=None):
        return self._load_item(np.asarray(self._data_t1.dataobj[..., index], dtype=np.float32), "t1")

    def t2_scan_slice(self, index=None):
        return self._load_item(np.asarray(self._data_t2.dataobj[..., index], dtype=np.float32), "t2")

    def segm_vs_slice(self, index=None):
        return np.asarray(self._data_vs.dataobj[..., index], dtype=np.int16)

    def segm_cochlea_slice(self, index=None):
        return np.asarray(self._data_cochlea.dataobj[..., index],
                          dtype=np.int16) if self._data_cochlea is not None else np.zeros_like(
            self._data_t1.dataobj[..., index])

    def segm_vs_class_slice(self, index=None):
        return self.segm_vs_class[index]

    def segm_cochlea_class_slice(self, index=None):
        return self.segm_cochlea_class[index]

    def _load_item(self, array, data_type):
        # clip extrem values
        sample = np.clip(array, float(self._statistics[data_type]["1st_percentile"]),
                         float(self._statistics[data_type]["99th_percentile"]))
        # z score normalization
        sample = (sample - float(self._statistics[data_type]["mean"])) / float(self._statistics[data_type]["std"])
        # range [0, 1]
        return (sample - float(self._statistics[data_type]["min"])) / (
                float(self._statistics[data_type]["max"]) - float(self._statistics[data_type]["min"]))

    def get_non_zero_slices_segmentation(self):
        """
        Extract all slices of segmentation VS that are non-zero.
        :return: list with non-zero slice indices for VS and cochlea
        """
        img_vs = self.segm_vs
        non_zero_vs = self._get_non_zero_slices_segmentation(img_vs)
        return non_zero_vs

    def get_zero_slices_segmentation(self):
        """
        Extract all slices of segmentation VS that are non-zero.
        :return: list with non-zero slice indices for VS
        """
        img_vs = self.segm_vs
        non_zero_vs = self._get_zero_slices_segmentation(img_vs)
        return non_zero_vs

    def get_val_slices_segmentation(self, val, dsize=None):
        """
        Extract all slices of segmentation VS that are larger than val.
        :return: list with slice indices for VS
        """
        img_vs = self.segm_vs
        non_zero_vs = self._get_slices_segmentation(img_vs, val, dsize)
        return non_zero_vs

    @staticmethod
    def _get_non_zero_slices_segmentation(segmentation):
        return [idx for idx in range(0, segmentation.shape[2]) if np.sum(segmentation[:, :, idx]) != 0]

    @staticmethod
    def _get_zero_slices_segmentation(segmentation):
        return [idx for idx in range(0, segmentation.shape[2]) if np.sum(segmentation[:, :, idx]) == 0]

    @staticmethod
    def _get_slices_segmentation(segmentation, val, dsize=None):
        if dsize is None:
            return [idx for idx in range(0, segmentation.shape[2]) if np.sum(segmentation[:, :, idx]) > val]
        else:
            return [idx for idx in range(0, segmentation.shape[2]) if
                    np.sum(cv2.resize(segmentation[:, :, idx], dsize=dsize, interpolation=cv2.INTER_CUBIC)) > val]

    def reduce_to_nonzero_segm(self, structure=None):
        """
        Reduce the data to nonzero segmentation slices - should keep only relevant information
        :param structure: identifier for structure to use for filtering - default: both structures are kept
        """
        idx_vs = self.get_non_zero_slices_segmentation()
        if structure == "vs":
            idx_slice = idx_vs
        else:
            logging.warning("DataContainer: other structures are not supported.")
        if len(idx_slice) == 0:
            logging.warning(
                "DataContainer: dataset {0} does not contain structure {1}. Full data volume is kept.".format(
                    self._path_dir,
                    structure))
        else:
            self._data_t1 = self._data_t1.slicer[..., idx_slice[0]:idx_slice[-1]]
            self._data_t2 = self._data_t2.slicer[..., idx_slice[0]:idx_slice[-1]]
            self._data_vs = self._data_vs.slicer[..., idx_slice[0]:idx_slice[-1]]
            self._data_cochlea = self._data_cochlea.slicer[...,
                                 idx_slice[0]:idx_slice[-1]] if self._data_cochlea is not None else None

    def get_slice_to_vis(self):
        """
        Get median slice where VS (and cochlea) segmentations exist.
        :return:
        """
        slice_vs = self.get_non_zero_slices_segmentation()
        return int(np.median(slice_vs))

    def plot_slice(self, slice_to_vis=None, figsize=(15, 15), cmap="gray"):
        """
        Tile plot of one timepoint with T1, T2 and VS segmentation mask.
        :param slice_to_vis: slice to be visualized; if None  - the median slice of non-zero slices will be taken
        :param figsize: figure size
        :param cmap: color map
        """
        # determine median slice that has segmentation mask
        if slice_to_vis is None:
            slice_to_vis = self.get_slice_to_vis()

        # plot
        fig, ax = plt.subplots(1, 4, figsize=figsize)
        fig.tight_layout()
        ax[0].imshow(self.t1_scan_slice(slice_to_vis), cmap=cmap)
        ax[0].set_title("T1")
        ax[1].imshow(self.t2_scan_slice(slice_to_vis), cmap=cmap)
        ax[1].set_title("T2")
        ax[2].imshow(self.segm_vs_slice(slice_to_vis), cmap=cmap)
        ax[2].set_title("VS")
        ax[3].imshow(self.segm_cochlea_slice(slice_to_vis), cmap=cmap)
        ax[3].set_title("Cochlea")
        fig.suptitle("Data slice {}".format(slice_to_vis), fontsize=16)
        plt.show()

    def plot_grid(self, nrows=5, ncols=6, skip=2, cmap="gray", modality="t1"):
        """
        Tile plot with multiple slides in a grid order.
        :param nrows: number of rows in the grid
        :param ncols: number of columns in the grid
        :param skip: number of slices to be skipped (default: every second slice is shown)
        :param cmap: color map
        :return:
        """
        # load NIFTI images
        data = self.data[modality]

        if nrows * ncols > len(self):
            raise ValueError("nrows*ncols larger than {}".format(len(self)))

        # plot
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        counter = 0
        for row in range(nrows):
            for col in range(ncols):
                axes[row, col].imshow(data[:, :, counter], cmap=cmap)
                axes[row, col].set_title(counter)
                axes[row, col].axis("off")
                counter += skip
        plt.show()

    def plot_overlap(self, slice_to_vis=None, figsize=(15, 15), cmap="gray"):
        """
        Plot segmentation mask on top of either T1 and T2 image for one time point.
        :param slice_to_vis: slice to be visualized; if None  - the median slice of non-zero slices will be taken
        :param figsize: figure size
        :param cmap: color map
        """

        # determine median slice that has segmentation mask
        if slice_to_vis is None:
            slice_to_vis = self.get_slice_to_vis()

        # process data
        tmp = self.t1_scan_slice(slice_to_vis).copy()
        tmp2 = self.t2_scan_slice(slice_to_vis).copy()
        toshow = np.zeros_like(tmp)
        toshow2 = np.zeros_like(tmp2)
        toshow = cv2.normalize(tmp, toshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        toshow2 = cv2.normalize(tmp2, toshow2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        toshow3 = self.segm_vs_slice(slice_to_vis)
        toshow4 = self.segm_cochlea_slice(slice_to_vis)

        # plot
        if cmap == "gray":
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            ax[0].imshow(toshow, cmap="gray")
            ax[0].imshow(toshow3 * 200, alpha=0.3)
            ax[0].imshow(toshow4 * 300, alpha=0.3)
            ax[0].set_title("T1")
            ax[0].axis("off")
            ax[1].imshow(toshow2, cmap="gray")
            ax[1].imshow(toshow3 * 200, alpha=0.3)
            ax[1].imshow(toshow4 * 300, alpha=0.3)
            ax[1].set_title("T2")
            ax[1].axis("off")
        else:
            img = cv2.addWeighted(toshow / 255., 1.0, toshow3 + toshow4 * 2, 0.5, 0.0)
            img2 = cv2.addWeighted(toshow2 / 255., 1.0, toshow3 + toshow4 * 2, 0.5, 0.0)
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            ax[0].imshow(img)
            ax[0].set_title("T1")
            ax[0].axis("off")
            ax[1].imshow(img2)
            ax[1].set_title("T2")
            ax[1].axis("off")
        fig.suptitle("Data slice {}".format(slice_to_vis), fontsize=16)
        plt.show()
