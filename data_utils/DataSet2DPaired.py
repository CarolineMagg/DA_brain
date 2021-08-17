########################################################################################################################
# DataSet for Tensorflow trainings pipeline
########################################################################################################################
import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randrange

from data_utils.DataSet2D import DataSet2D

__author__ = "c.magg"


class DataSet2DPaired(DataSet2D):
    """
    DataSet2DPaired is a container for a 2D dataset used in supervised training.
    It contains inputs (image) and outputs (same as input or segm mask) in a paired manner (ie same patient and slice).
    """

    def __init__(self, dataset_folder, batch_size=4,
                 input_data="t1", input_name="image",
                 output_data="vs", output_name="mask",
                 shuffle=True, p_augm=0.0, use_filter=None,
                 dsize=(256, 256), alpha=0, beta=255):
        """
        Create a new DataSet2D object.
        :param dataset_folder: path to dataset folder
        :param batch_size: batch size for loading data
        :param input_data: input data identifiers, eg. t1, t2
        :param output_data: output data identifiers, eg t2, t1, vs
        :param input_name: name of the input tensors, eg. image (Note: at least one image input is mandatory)
        :param output_name: name of the output tensors, eg. mask
        :param shuffle: boolean for shuffle the indices
        :param use_filter: use structure for filtering
        :param dsize: image size
        :param alpha: alpha values of images (lower boundary of pixel intensity range)
        :param beta: beta values of images (upper boundary of pixel intensity range)

        Examples:
        # Dataset for supervised segmentation network with T1 as input and VS segm as output
        >> dataset = DataSet2DPaired(dataset_folder="../data/VS_segm/VS_registered/training/",
                                    input_data="t1", input_name="image",
                                    output_data="vs", output_data="segm",
                                    batch_size = 4, shuffle=True, p_augm=0.0)

        # Dataset for supervised autoencoder training with same (!) input and output
        # will be mapped to input_name and output_{image_name} dict
        >> dataset = DataSet2DPaired(dataset_folder="../data/VS_segm/VS_registered/training/",
                                    input_data="t1", input_name="image",
                                    output_data=None, output_data="output_image",
                                    batch_size = 4, shuffle=True, p_augm=0.0)
        """
        super(DataSet2DPaired, self).__init__(dataset_folder, batch_size=batch_size,
                                              input_data=input_data, input_name=input_name,
                                              shuffle=shuffle, p_augm=p_augm, use_filter=use_filter,
                                              dsize=dsize, alpha=alpha, beta=beta)

        # output data
        self._output_name = output_name if type(output_name) == list else [output_name]
        self._output_data = output_data if type(output_data) == list else [output_data]
        self._mapping_data_name.update({k: v for k, v in zip(self._output_data, self._output_name)})
        assert len(self._output_data) == len(self._output_name)
        if output_data is None:
            self._output_name = None
            self._output_data = None

        # same output as input
        self._same_output_as_input = False if output_data is not None else True

    @property
    def has_same_output_as_input(self):
        return self._same_output_as_input

    @has_same_output_as_input.setter
    def has_same_output_as_input(self, to):
        self._same_output_as_input = to

    def lookup_data_augmentation(self):
        """
        Generate the mapping between data naming and augmentation role.
        Input names -> image
        Output name -> mask
        :return: lookup dictionary with data name as key and augmentation role as value (as expected by albumentations)
        """
        lookup = super(DataSet2DPaired, self).lookup_data_augmentation()
        if self._output_name is not None:
            for output_name in self._output_name:
                lookup[output_name] = "mask"
        return lookup

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
        outputs = {}
        for input_name in self._input_name:
            inputs[input_name] = np.stack([data[idx][input_name] for idx in range(len(data))]).astype(np.float32)
        if self._same_output_as_input:
            logging.info("DataSetPaired2D: 'output_name' will be ignored and replaced by 'output_{input_name}'.")
            return inputs, {"output_{0}".format(k): v for k, v in inputs.items()}
        for output_name in self._output_name:
            outputs[output_name] = np.stack([data[idx][output_name] for idx in range(len(data))]).astype(np.float32)
        return inputs, outputs

    def _load_data_item(self, ds, item):
        """
        Load data item, normalize images, resize image and segm mask.
        :param ds: index for patient data
        :param item: index for slice
        :return: data dictionary
        """
        data = super(DataSet2DPaired, self)._load_data_item(ds, item)
        if self._output_name is not None and self._output_data is not None:
            for output_name, output_data in zip(self._output_name, self._output_data):
                segm = getattr(self._data[ds], self.lookup_data_call()[output_data])(item)
                if output_data in ["t1", "t2"]:
                    segm = (segm - np.mean(segm)) / np.std(segm)  # z score normalization
                    segm = cv2.normalize(segm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                segm = cv2.resize(segm, dsize=self._dsize, interpolation=cv2.INTER_CUBIC)
                data[output_name] = segm
        return data

    def plot_random_images(self, nrows=2, ncols=2):
        """
        Plot random examples of dataset.
        :param nrows: number of rows
        :param ncols: number of columns
        """
        if nrows * ncols > self.batch_size:
            raise ValueError(
                "DataSet2D: More images {0} requested than available in a batch with size {1}.".format(nrows * ncols,
                                                                                                       self.batch_size))
        data = self._next(randrange(self.__len__()))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        counter = 0
        for row in range(nrows):
            for col in range(ncols):
                random_data = randrange(len(self._input_name))
                axes[row, col].imshow(data[0][self._input_name[random_data]][counter], cmap="gray")
                if not self._same_output_as_input:
                    for output_name in self._output_name:
                        axes[row, col].imshow(data[1][output_name][counter], alpha=0.3)
                axes[row, col].set_title("{0}, mod: {1}".format(counter, self._input_data[random_data]))
                axes[row, col].axis("off")
                counter += 1
        fig.suptitle("{0} random examples of {1} images.".format(nrows * ncols, self._number_index), fontsize=16)
        plt.show()

    def plot_random_images_separately(self, nrows=2, ncols=2):
        """
        Plot random examples of dataset.
        :param nrows: number of rows
        :param ncols: number of columns
        """
        if nrows * ncols > self.batch_size:
            raise ValueError(
                "DataSet2D: More images {0} requested than available in a batch with size {1}.".format(nrows * ncols,
                                                                                                       self.batch_size))
        data = self._next(randrange(self.__len__()))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, figsize=(15, 15))
        counter = 0
        for row in range(nrows):
            for col in range(0, ncols * 2, 2):
                random_data = randrange(len(self._input_name))
                axes[row, col].imshow(data[0][self._input_name[random_data]][counter], cmap="gray")
                for output_name in data[1].keys():
                    if np.sum(data[1][output_name][counter]) != 0:
                        axes[row, col + 1].imshow(data[1][output_name][counter], cmap="gray")
                axes[row, col].set_title("{0}, mod: {1}".format(counter, self._input_data[random_data]))
                axes[row, col + 1].set_title("{0}, mod: {1}".format(counter, list(data[1].keys())))
                axes[row, col].axis("off")
                axes[row, col + 1].axis("off")
                counter += 1
        fig.suptitle("{0} random examples of {1} images.".format(nrows * ncols, self._number_index), fontsize=16)
        plt.show()
