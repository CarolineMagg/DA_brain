########################################################################################################################
# DataSet Mixed for Tensorflow trainings pipeline
########################################################################################################################
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randrange
import albumentations as A

from data_utils.DataSet2D import DataSet2D

__author__ = "c.magg"

from models.utils import check_gpu


class DataSet2DSynthetic(DataSet2D):
    """
    DataSet2DSynthetic is a container for a 2D dataset used in training.
    It contains inputs (images) and outputs (segmentation) in a paired manner:
     * same patient and slice for image to segm (paired)
     * the input is the synthetic version of the input_data -> if input_data = "t1", then the image looks like "t2"
    Segm mask will always have pixel values [0,1], whereas images will have [alpha, beta] (default: [0,1]).

    Note: very slow in training (!)
    """

    def __init__(self, dataset_folder, cycle_gan, batch_size=4,
                 input_data="t1", input_name="image",
                 output_data=["vs", "vs_class"], output_name=["vs_output"],
                 shuffle=True, p_augm=0.0, use_filter=None, use_balance=False, segm_size=None,
                 dsize=(256, 256), alpha=0, beta=1, seed=13375):
        """
        Create a new DataSet2D object.
        """

        self.index_pairwise_output = []
        super(DataSet2DSynthetic, self).__init__(dataset_folder, batch_size=batch_size,
                                                 input_data=input_data, input_name=input_name,
                                                 shuffle=shuffle, p_augm=p_augm, use_filter=use_filter,
                                                 segm_size=segm_size,
                                                 use_balance=use_balance, dsize=dsize, alpha=alpha, beta=beta,
                                                 seed=seed)

        # output data
        self._output_name = output_name if type(output_name) == list else [output_name]
        self._output_data = output_data if type(output_data) == list else [output_data]
        self._mapping_data_name.update({k: v for k, v in zip(self._output_data, self._output_name)})
        assert len(self._output_data) == len(
            self._output_name), "Number of output data and output names is not matching."

        self._augm_methods = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                     A.MedianBlur(p=0.5, blur_limit=5),
                     A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
        ]

        # cycle gan
        check_gpu()
        self._generator = tf.keras.models.load_model(os.path.join("/tf/workdir/DA_brain/saved_models", cycle_gan))

    @staticmethod
    def lookup_data_call():
        """
        Generate the mapping between data naming and data loading method in DataContainer.
        :return lookup dictionary with data name as key and method name as value
        """
        return {'t1': 't1_scan_slice',
                't2': 't2_scan_slice',
                'vs': 'segm_vs_slice',
                'vs_class': 'segm_vs_class_slice',
                'cochlea': 'segm_cochlea_slice',
                'cochlea_class': 'segm_cochlea_class_slice'}

    def lookup_data_augmentation(self):
        """
        Generate the mapping between data naming and augmentation role.
        Input names -> image
        Output name -> mask
        :return: lookup dictionary with data name as key and augmentation role as value (as expected by albumentations)
        """
        lookup = super(DataSet2DSynthetic, self).lookup_data_augmentation()
        for output_data, output_name in zip(self._output_data, self._output_name):
            if output_data in ["vs", "cochlea"]:
                lookup[output_name] = "mask"
            elif output_data in ["vs_class", "cochlea_class"]:
                lookup[output_name] = "classification"
            else:
                raise ValueError(f"DataSet2DSynthetic: {output_name} not valid.")
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
                "DataSet2DSynthetic: Indices length is 0 with item {0} of {1}".format(item,
                                                                                      self._number_index // self.batch_size))
        if len(indices) < self.batch_size:
            raise ValueError("DataSet2DSynthetic: Batch size is too large, not enough data samples available.")
        data = [dict()] * self.batch_size
        for idx, ind in enumerate(indices):
            data[idx] = self._next_data_item(ind)

        inputs = {}
        for input_name in self._input_name:
            inputs[input_name] = np.stack([data[idx][input_name] for idx in range(len(data))]).astype(np.float32)
        outputs = {}
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
        data = {}
        # synthetic images
        for input_name, input_data in zip(self._input_name, self._input_data):
            img = tf.expand_dims(self._load_data_sample(ds, input_data, item), 0)
            img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * (-1) + 1  # px range [-1,1]
            img = self._generator(img)  # px range [-1,1]
            img = (img + 1) / 2  # px range [0,1]
            data[input_name] = img.numpy()[0, :, :, 0]
        # paired segm mask
        for output_name, output_data in zip(self._output_name, self._output_data):
            if output_data in ["vs", "cochlea"]:
                data[output_name] = self._load_data_sample(ds, output_data, item)
            if output_data in ["vs_class", "cochlea_class"]:
                data[output_name] = self._load_data_sample(ds, output_data, item)
        return data

    def plot_random_images(self, nrows=4, ncols=2):
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
                axes[row, col].imshow(data[0][self._input_name[0]][counter], cmap="gray")
                output_name_segm = [v for k, v in self._mapping_data_name.items() if k in ["vs", "cochlea"]]
                output_name_class = [v for k, v in self._mapping_data_name.items() if
                                     k in ["vs_class", "cochlea_class"]]
                if len(output_name_segm) != 0:
                    axes[row, col].imshow(data[1][output_name_segm[0]][counter], alpha=0.5)
                if len(output_name_class) != 0:
                    axes[row, col].set_title(
                        "{0}, mod: {1}, class: {2}".format(counter, self._input_data[0],
                                                           data[1][output_name_class[0]][counter]))
                else:
                    axes[row, col].set_title(
                        "{0}, mod: {1}".format(counter, self._input_data[0]))

                if len(output_name_segm) != 0:
                    axes[row, col + 1].imshow(data[1][self._output_name[0]][counter], cmap="gray")
                output_name_segm2 = [v for k, v in self._mapping_data_name.items() if k in ["vs", "cochlea"]]
                output_name_class2 = [v for k, v in self._mapping_data_name.items() if
                                      k in ["vs_class", "cochlea_class"]]
                if len(output_name_segm2) != 0:
                    axes[row, col + 1].imshow(data[1][output_name_segm2[0]][counter], alpha=0.5)
                if len(output_name_class2) != 0:
                    axes[row, col + 1].set_title(
                        "{0}, mod: {1}, class: {2}".format(counter, self._output_data[0],
                                                           data[1][output_name_class2[0]][counter]))
                else:
                    axes[row, col + 1].set_title(
                        "{0}, mod: {1}".format(counter, self._input_data[0]))
                axes[row, col].axis("off")
                axes[row, col + 1].axis("off")
                counter += 1
        fig.suptitle("{0} random examples of {1} images.".format(nrows * ncols, self._number_index), fontsize=16)
        plt.show()
