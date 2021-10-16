########################################################################################################################
# DataSet2D base class for Tensorflow trainings pipeline
########################################################################################################################
import logging

import numpy as np
import os.path
import random
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from random import randrange
import albumentations as A
from natsort import natsorted

from data_utils.DataContainer import DataContainer

__author__ = "c.magg"


class DataSet2D(tf.keras.utils.Sequence):
    """
    DataSet2D is a container for a 2D dataset used in training.
    This base class only contains an input data dictionary.
    """

    def __init__(self, dataset_folder, batch_size=4,
                 input_data="t1", input_name="image",
                 shuffle=True, p_augm=0.0, use_filter=None, use_balance=False, segm_size=None,
                 dsize=(256, 256), alpha=0, beta=1, seed=13375):
        """
        Create a new DataSet2D object.
        """
        # set random seed
        np.random.seed(seed)

        # dataset folder
        self._path_dataset = dataset_folder
        self._folders = natsorted(os.listdir(self._path_dataset))

        # load data
        self._data = []
        self._load_data()

        # set indices
        self.list_index = []
        self._number_index = int(np.sum([len(d) for d in self._data])) if len(self.list_index) == 0 else len(
            self.list_index)
        self.index_pairwise = []
        self._shuffle = shuffle
        random.seed(13371984)
        self.reset()

        # other parameters
        self._batch_size = batch_size
        self._input_name = input_name if type(input_name) == list else [input_name]
        self._input_data = input_data if type(input_data) == list else [input_data]
        self._mapping_data_name = {k: v for k, v in zip(self._input_data, self._input_name)}
        assert len(self._input_data) == len(self._input_name), "Number of input data and input names is not matching."

        self._alpha = alpha
        self._beta = beta
        self._dsize = dsize if dsize is not None else (448, 448)
        self._p_augm = p_augm
        self._augm_methods = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                     A.MedianBlur(p=0.5, blur_limit=5),
                     A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
        ]
        self._filter = use_filter
        if self._filter is not None and segm_size is not None:
            logging.warning("DataSet2D: It is inefficient to use 'filter' and 'segm_size' at the same time. \
            And the behavior might not lead to the same results.")
        if self._filter is not None:
            self.reduce_to_nonzero_segm(self._filter)
        self._segm_size = segm_size
        self._balance = use_balance
        if self._segm_size is not None and not self._balance:
            self.reduce_greater_than_val_segm()
        if self._balance:
            self.balance_dataset()

    @staticmethod
    def lookup_data_call():
        """
        Generate the mapping between data naming and data loading method in DataContainer.
        :return lookup dictionary with data name as key and method name as value
        """
        return {'t1': 't1_scan_slice',
                't2': 't2_scan_slice',
                'vs': 'segm_vs_slice',
                'cochlea': 'segm_cochlea_slice'}

    def lookup_data_augmentation(self):
        """
        Generate the mapping between data naming and augmentation role.
        Input names -> image
        Output name -> mask
        :return: lookup dictionary with data name as key and augmentation role as value (as expected by albumentations)
        """
        lookup = {}
        for input_name in self._input_name:
            lookup[input_name] = "image"
        return lookup

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, to):
        logging.debug(f"DataSet2D: set 'batch_size' to {to}")
        self._batch_size = to

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, to):
        logging.debug(f"DataSet2D: set 'shuffle' to {to}")
        self._shuffle = to

    @property
    def dsize(self):
        return self._dsize

    @dsize.setter
    def dsize(self, to):
        logging.debug(f"DataSet2D: set 'dsize' to {to}")
        self._dsize = to

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, to):
        logging.debug(f"DataSet2D: set 'alpha' to {to}")
        self._alpha = to

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, to):
        logging.debug(f"DataSet2D: set 'beta' to {to}")
        self._beta = to

    @property
    def p_augm(self):
        return self._p_augm

    @p_augm.setter
    def p_augm(self, to):
        logging.debug(f"DataSet2D: set 'p_augm' to {to}")
        self._p_augm = to

    @property
    def augm_methods(self):
        return self._augm_methods

    @augm_methods.setter
    def augm_methods(self, to):
        assert type(to) == list, "'augm_methods' needs to be a list of albumentations methods"
        self._augm_methods = to

    @property
    def segm_size(self):
        return self._segm_size

    @segm_size.setter
    def segm_size(self, to):
        self._segm_size = to
        self.reduce_greater_than_val_segm()
        if self._balance:
            self.balance_dataset()

    def __len__(self):
        return self._number_index // self.batch_size

    def __getitem__(self, item):
        return self._next(item)

    def _next(self, item):
        """
        Generate the next batch of data.
        :param item: number of batch
        :return: dictionary for inputs and outputs
        """
        indices = self.index_pairwise[
                  self.batch_size * item:self.batch_size * item + self.batch_size]
        if len(indices) == 0:
            raise ValueError(
                "DataSet2D: Indices length is 0 with item {0} of {1}".format(item,
                                                                             self._number_index // self.batch_size))
        if len(indices) < self.batch_size:
            raise ValueError("DataSet2D: Batch size is too large, not enough data samples available.")
        data = [dict()] * self.batch_size
        for idx, ind in enumerate(indices):
            data[idx] = self._next_data_item(ind)

        inputs = {}
        for input_name in self._input_name:
            inputs[input_name] = np.stack([data[idx][input_name] for idx in range(len(data))]).astype(np.float32)
        return inputs

    def generator(self):
        for idx in range(self._number_index // self.batch_size):
            yield self._next(idx)

    def _next_data_item(self, indices):
        """
        Generate the next data item.
        :param indices: tuple of indices for (patient ID, slice)
        :return: tuple of input and output
        """
        ds, item = indices
        data = self._load_data_item(ds, item)
        data = self._augmentation(data)
        data = self._final_prep(data)
        return data

    def _load_data_item(self, ds, item):
        """
        Load data item, normalize images, resize image and segm mask.
        :param ds: index for patient data
        :param item: index for slice
        :return: data dictionary with input(s)
        """
        data = {}
        for input_name, input_data in zip(self._input_name, self._input_data):
            data[input_name] = self._load_data_sample(ds, input_data, item)
        return data

    def _load_data_sample(self, ds, data_type, item):
        sample = getattr(self._data[ds], self.lookup_data_call()[data_type])(item)
        if data_type in ["t1", "t2", "vs", "cochlea"]:
            return cv2.resize(sample, dsize=self._dsize, interpolation=cv2.INTER_CUBIC)
        elif data_type in ["vs_class", "cochlea_class"]:
            return sample
        else:
            raise ValueError(f"DataSet2D: datatype {data_type} not supported.")

    def _augmentation(self, data):
        """
        Apply augmentation to image and segm mask.
        :param data: data dictionary
        :return: transformed data dictionary
        """
        transformed = data
        if self._p_augm != 0 and self._p_augm > random.uniform(0, 1):
            transform = A.Compose(self._augm_methods,
                                  additional_targets=self.lookup_data_augmentation())
            transformed = transform(**data)
        return transformed

    def _final_prep(self, data):
        """
        Apply final processing methods, eg. normalize for images
        :param data: data dictionary
        :return: transformed data dictionary
        """
        image_keys = [k for k, v in self.lookup_data_augmentation().items() if v == 'image']
        for k in image_keys:
            if np.max(data[k]) - np.min(data[k]) == 0:
                data[k] = data[k]
            else:
                data[k] = ((data[k] - np.min(data[k])) / (np.max(data[k]) - np.min(data[k]))) * (
                        self.beta - self.alpha) + self.alpha
        return data

    def _load_data(self):
        """
        Load data from folders with DataContainer.
        """
        logging.info("DataSet2D: Load data ...")
        for f in self._folders:
            folder = os.path.join(self._path_dataset, f)
            self._data.append(DataContainer(folder))

    def reset(self):
        """
        Reset indices and shuffle randomly if necessary.
        """
        self.index_pairwise = [(idx, n) for idx, d in enumerate(self._data) for n in range(len(d))] if len(
            self.list_index) == 0 else self.list_index
        if self._shuffle:
            self.index_pairwise = np.random.permutation(self.index_pairwise)

    def reduce_to_nonzero_segm(self, structure=None):
        """
        Reduce each data container to nonzero segmentation slices - should keep only relevant information.
        :param structure: identifier for structure to use for filtering - default: both structures are kept
        """
        len_before = self._number_index
        logging.info(f"DataSet2D: filtering {len_before} data points for {structure}...")
        for idx, d in enumerate(self._data):
            idx_vs_nonzero = d.get_non_zero_slices_segmentation()
            self.list_index += [(idx, ix) for ix in sorted(idx_vs_nonzero)]
            # d.reduce_to_nonzero_segm(structure)
        self._number_index = int(np.sum([len(d) for d in self._data])) if len(self.list_index) == 0 else len(
            self.list_index)
        len_after = self._number_index
        self.reset()
        logging.info(
            f"DataSet2D: filtering removed {len_before - len_after} data points. remaining {len_after}.")

    def balance_dataset(self):
        """
        Balance each data container to have equal number of slices with/without the structure.
        """
        self.list_index = []
        len_before = self._number_index
        logging.info(f"DataSet2D: balancing {len_before} data points for vs_class...")
        if self._segm_size is None:
            segm_size = 0
        else:
            segm_size = self._segm_size
        for idx, d in enumerate(self._data):
            idx_vs_nonzero = d.get_val_slices_segmentation(segm_size, self._dsize)
            idx_vs_zero = d.get_zero_slices_segmentation()
            blub = random.sample(idx_vs_zero, k=len(idx_vs_nonzero))
            self.list_index += [(idx, ix) for ix in sorted(blub + idx_vs_nonzero)]
        self._number_index = int(np.sum([len(d) for d in self._data])) if len(self.list_index) == 0 else len(
            self.list_index)
        len_after = self._number_index
        self.reset()
        logging.info(
            f"DataSet2D: balancing removed {len_before - len_after} data points. remaining {len_after}.")

    def reduce_greater_than_val_segm(self):
        """
        Reduce each data container to segmentation slices that has at least val non-zero px
        - should keep only relevant information.
        """
        self.list_index = []
        len_before = self._number_index
        logging.info(f"DataSet2D: filtering {len_before} data points for segm size...")
        for idx, d in enumerate(self._data):
            self.list_index += [(idx, ix) for ix in d.get_val_slices_segmentation(self._segm_size, self._dsize)]
        self._number_index = int(np.sum([len(d) for d in self._data])) if len(self.list_index) == 0 else len(
            self.list_index)
        len_after = self._number_index
        self.reset()
        logging.info(
            f"DataSet2D: filtering removed {len_before - len_after} data points. remaining {len_after}.")

    def plot_random_images(self, nrows=2, ncols=2):
        """
        Plot random examples of dataset.
        :param nrows: number of rows
        :param ncols: number of columns
        """
        if nrows * ncols > self.batch_size:
            raise ValueError(
                "DataSet2D: More images requested than available in a batch: {0} vs {1}.".format(nrows * ncols,
                                                                                                 self.batch_size))
        data = self._next(randrange(self.__len__()))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        counter = 0
        for row in range(nrows):
            for col in range(ncols):
                random_data = randrange(len(self._input_name))
                axes[row, col].imshow(data[self._input_name[random_data]][counter], cmap="gray")
                axes[row, col].set_title(counter)
                axes[row, col].axis("off")
                counter += 1
        fig.suptitle("{0} random examples of {1} images.".format(nrows * ncols, self._number_index), fontsize=16)
        plt.show()
