########################################################################################################################
# Script to generate additional information about T1, T2 slices that are non-empty
# will generate a txt file in the dataset folder containing statistical values.
########################################################################################################################

import os
import shutil

import numpy as np
import os.path
import nibabel as nib
import json

__author__ = "c.magg"

from natsort import natsorted


class DataContainer:
    """
    DataContainer is a container for a nifti folders with different information like:
    * T1
    * T2
    * segmentation of VS/cochlea
    """

    def __init__(self, dir_path, correct=False):
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
        self._path_data_info = os.path.join(dir_path, "vs_gk_slices.json")
        self._data_t1 = None
        self._data_t2 = None
        self._data_vs = None
        self._data_cochlea = None
        self.load(correct)

    def load(self, correct):
        """
        (Re)Load the data from nifti paths.
        """
        self._data_t1 = nib.load(self._path_t1)
        self._data_t2 = nib.load(self._path_t2)
        self._data_vs = nib.load(self._path_vs)
        self._data_cochlea = nib.load(self._path_cochlea) if self._path_cochlea is not None else None
        if os.path.isfile(self._path_data_info) and correct:
            with open(self._path_data_info) as json_file:
                self._statistics = json.load(json_file)
            # only use non-empty slices
            idx_slice0 = max(int(self._statistics["t1"]["first_nonempty_slice"]),
                             int(self._statistics["t2"]["first_nonempty_slice"]))
            idx_slice1 = min(int(self._statistics["t1"]["last_nonempty_slice"]),
                             int(self._statistics["t2"]["last_nonempty_slice"]))
            previous_size = self._data_t1.shape[2]
            if idx_slice0 is not 0 or idx_slice1 is not self._data_t1.shape[2]:
                self._data_t1 = self._data_t1.slicer[..., idx_slice0:idx_slice1]
                self._data_t2 = self._data_t2.slicer[..., idx_slice0:idx_slice1]
                self._data_vs = self._data_vs.slicer[..., idx_slice0:idx_slice1]
                self._data_cochlea = self._data_cochlea.slicer[...,
                                     idx_slice0:idx_slice1] if self._data_cochlea is not None else None
                print(f"DataContainer {self._path_dir}: Removed empty slices; {previous_size} to {self._data_t1.shape[2]}")

    def uncache(self):
        """
        Uncache the nifti container.
        """
        self._data_t1.uncache()
        self._data_t2.uncache()
        self._data_vs.uncache()
        self._data_cochlea.uncache()

    def save(self):
        """
        Save new nifti images.
        """
        nib.save(self._data_t1, self._path_t1)
        nib.save(self._data_t2, self._path_t2)
        nib.save(self._data_vs, self._path_vs)
        if self._data_cochlea is not None:
            nib.save(self._data_cochlea, self._path_cochlea)

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

    # extract info about empty slices
    print("Extract info about empty slices.")
    for path in paths:
        folders = natsorted([os.path.join(path, p) for p in os.listdir(path)])
        for folder in folders:
            print(folder)
            statistics = {}
            fn = os.path.join(folder, "vs_gk_slices.json")
            container = DataContainer(folder)
            ## t1 ##
            sample = container.t1_scan
            # remove empty slices
            sums = [np.sum(sample[:, :, idx]) for idx in range(np.shape(sample)[-1])]
            empty_slices = [idx for idx, s in enumerate(sums) if s == 0]
            if len(empty_slices) != 0:
                tmp = np.where(np.diff(empty_slices[:]) > 1)[0]
                if len(tmp) != 0:
                    first_nonemtpy_slice = empty_slices[tmp[0]] + 1
                    last_nonempty_slice = empty_slices[tmp[0] + 1] - 1
                else:
                    if empty_slices[0] == 0:
                        first_nonemtpy_slice = empty_slices[-1] + 1
                        last_nonempty_slice = len(sums)
                    elif empty_slices[-1] == len(sums) - 1:
                        first_nonemtpy_slice = 0
                        last_nonempty_slice = empty_slices[0] - 1
            else:
                first_nonemtpy_slice = 0
                last_nonempty_slice = len(sums)
            sample = sample[:, :, first_nonemtpy_slice:last_nonempty_slice]
            statistics["t1"] = {"first_nonempty_slice": str(first_nonemtpy_slice),
                                "last_nonempty_slice": str(last_nonempty_slice)}

            ## t2 ##
            sample = container.t2_scan
            # remove empty slices
            sums = [np.sum(sample[:, :, idx]) for idx in range(np.shape(sample)[-1])]
            empty_slices = [idx for idx, s in enumerate(sums) if s == 0]
            if len(empty_slices) != 0:
                tmp = np.where(np.diff(empty_slices[:]) > 1)[0]
                if len(tmp) != 0:
                    first_nonemtpy_slice = empty_slices[tmp[0]] + 1
                    last_nonempty_slice = empty_slices[tmp[0] + 1] - 1
                else:
                    if empty_slices[0] == 0:
                        first_nonemtpy_slice = empty_slices[-1] + 1
                        last_nonempty_slice = len(sums)
                    elif empty_slices[-1] == len(sums) - 1:
                        first_nonemtpy_slice = 0
                        last_nonempty_slice = empty_slices[0] - 1
            else:
                first_nonemtpy_slice = 0
                last_nonempty_slice = len(sums)
            sample = sample[:, :, first_nonemtpy_slice:last_nonempty_slice]
            statistics["t2"] = {"first_nonempty_slice": str(first_nonemtpy_slice),
                                "last_nonempty_slice": str(last_nonempty_slice)}
            # write file
            with open(fn, 'w') as outfile:
                json.dump(statistics, outfile)

    # remove empty slices
    print("Remove non-empty slices.")
    for path in paths:
        folders = natsorted([os.path.join(path, p) for p in os.listdir(path)])
        for folder in folders:
            print(folder)
            container = DataContainer(folder, correct=True)
            container.save()

    for path in paths:
        folders = natsorted([os.path.join(path, p) for p in os.listdir(path)])
        for folder in folders:
            container = DataContainer(folder)
            sample = container.t1_scan
            sums = [np.sum(sample[:, :, idx]) for idx in range(np.shape(sample)[-1])]
            empty_slices = [idx for idx, s in enumerate(sums) if s == 0]
            if len(empty_slices) != 0:
                print("something went wrong.")
            sample = container.t2_scan
            sums = [np.sum(sample[:, :, idx]) for idx in range(np.shape(sample)[-1])]
            empty_slices = [idx for idx, s in enumerate(sums) if s == 0]
            if len(empty_slices) != 0:
                print("something went wrong.")

    print("Finished.")
