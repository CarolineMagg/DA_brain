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

from data_utils.DataContainer import DataContainer

__author__ = "c.magg"


class DataSet2D(tf.keras.utils.Sequence):
    """
    DataSet2D is a container for a 2D dataset used in training.
    This base class only contains an input data dictionary.
    """

    def __init__(self, dataset_folder, batch_size=4,
                 input_data="t1", input_name="image",
                 shuffle=True, p_augm=0.0, use_filter=None,
                 dsize=(256, 256), alpha=0, beta=255):
        """
        Create a new DataSet2D object.
        :param dataset_folder: path to dataset folder
        :param batch_size: batch size for loading data
        :param input_data: input data identifiers, eg. t1, t2
        :param shuffle: boolean for shuffle the indices
        :param use_filter: use structure for filtering
        :param dsize: image size
        :param alpha: alpha values of images (lower boundary of pixel intensity range)
        :param beta: beta values of images (upper boundary of pixel intensity range)
        """

        # dataset folder
        self._path_dataset = dataset_folder
        self._folders = os.listdir(self._path_dataset)

        # load data
        self._data = []
        self._load_data()

        # set indices
        self._number_index = int(np.sum([len(d) for d in self._data]))
        self.index_pairwise = []
        self._shuffle = shuffle
        random.seed(13371984)
        self.reset()

        # other parameters
        self._batch_size = batch_size
        self._input_name = input_name if type(input_name) == list else [input_name]
        self._input_data = input_data if type(input_data) == list else [input_data]
        self._mapping_data_name = {k: v for k, v in zip(self._input_data, self._input_name)}
        assert len(self._input_data) == len(self._input_name)

        self._alpha = alpha
        self._beta = beta
        self._dsize = dsize
        self._p_augm = p_augm
        self._augm_methods = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                     A.MedianBlur(p=0.5, blur_limit=5),
                     A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
        ]
        self._filter = use_filter
        if self._filter is not None:
            self.reduce_to_nonzero_segm(self._filter)

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
        logging.info(f"DataSet2D: set 'batch_size' to {to}")
        self._batch_size = to

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, to):
        logging.info(f"DataSet2D: set 'shuffle' to {to}")
        self._shuffle = to

    @property
    def dsize(self):
        return self._dsize

    @dsize.setter
    def dsize(self, to):
        logging.info(f"DataSet2D: set 'dsize' to {to}")
        self._dsize = to

    @property
    def p_augm(self):
        return self._p_augm

    @p_augm.setter
    def p_augm(self, to):
        logging.info(f"DataSet2D: set 'p_augm' to {to}")
        self._p_augm = to

    @property
    def augm_methods(self):
        return self._augm_methods

    @augm_methods.setter
    def augm_methods(self, to):
        assert type(to) == list, "'augm_methods' needs to be a list of albumentations methods"
        self._augm_methods = to

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
                "DataSet: Indices length is 0 with item {0} of {1}".format(item, self._number_index // self.batch_size))
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
            img = getattr(self._data[ds], self.lookup_data_call()[input_data])(item)
            img = (img - np.mean(img)) / np.std(img)  # z score normalization
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            img = cv2.resize(img, dsize=self._dsize, interpolation=cv2.INTER_CUBIC)
            data[input_name] = img
        return data

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
            data[k] = cv2.normalize(data[k], None, alpha=self._alpha, beta=self._beta, norm_type=cv2.NORM_MINMAX)
        return data

    def _load_data(self):
        """
        Load data from folders with DataContainer.
        """
        for f in self._folders:
            folder = os.path.join(self._path_dataset, f)
            self._data.append(DataContainer(folder))

    def reset(self):
        """
        Reset indices and shuffle randomly if necessary.
        """
        self.index_pairwise = [(idx, n) for idx, d in enumerate(self._data) for n in range(len(d))]
        if self._shuffle:
            self.index_pairwise = np.random.permutation(self.index_pairwise)

    def reduce_to_nonzero_segm(self, structure=None):
        """
        Reduce each data container to nonzero segmentation slices - should keep only relevant information.
        :param structure: identifier for structure to use for filtering - default: both structures are kept
        """
        len_before = self._number_index
        logging.info(f"DataSetGenerator: filtering {len_before} data points for {structure}...")
        for d in self._data:
            d.reduce_to_nonzero_segm(structure)
        self._number_index = int(np.sum([len(d) for d in self._data]))
        len_after = self._number_index
        self.reset()
        logging.info(
            f"DataSetGenerator: filtering removed {len_before - len_after} data points. remaining {len_after}.")

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
