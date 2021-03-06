########################################################################################################################
# DataSet Mixed for Tensorflow trainings pipeline
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randrange
import albumentations as A

from data_utils.DataSet2D import DataSet2D

__author__ = "c.magg"


class DataSet2DMixed(DataSet2D):
    """
    DataSet2DMixed is a container for a 2D dataset used in training.
    It contains inputs (image) and outputs (images, segmentation) in an unpaired/paired manner:
     * not same patient and slice for image to image (unpaired)
     * same patient and slice for image to segm (paired)
    Segm mask will always have pixel values [0,1], whereas images will have [alpha, beta] (default: [0,1]).
    """

    def __init__(self, dataset_folder, batch_size=4,
                 input_data="t1", input_name="image",
                 output_data=["t2", "vs", "vs_class"], output_name=["image_output", "vs_output", "vs_class"],
                 shuffle=True, p_augm=0.0, use_filter=None, use_balance=False, segm_size=None, paired=False,
                 dsize=(256, 256), alpha=0, beta=1, seed=13375):
        """
        Create a new DataSet2D object.

        Examples:
        # Dataset for adversarial learning with input t1 and output t2
        >> dataset = DataSet2DMixed(dataset_folder="../data/VS_segm/VS_registered/training/",
                                    input_data="t1", input_name="image",
                                    output_data="t2", output_data="image_output",
                                    batch_size = 4, shuffle=True, p_augm=0.0)
        """

        self.index_pairwise_output = []
        self._paired = paired
        super(DataSet2DMixed, self).__init__(dataset_folder, batch_size=batch_size,
                                             input_data=input_data, input_name=input_name,
                                             shuffle=shuffle, p_augm=p_augm, use_filter=use_filter, segm_size=segm_size,
                                             use_balance=use_balance, dsize=dsize, alpha=alpha, beta=beta, seed=seed)

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
        lookup = super(DataSet2DMixed, self).lookup_data_augmentation()
        for output_data, output_name in zip(self._output_data, self._output_name):
            if output_data in ["vs", "cochlea"]:
                lookup[output_name] = "mask"
            elif output_data in ["vs_class", "cochlea_class"]:
                lookup[output_name] = "classification"
            else:
                lookup[output_name] = "image"
        return lookup

    def _next(self, item):
        """
        Generate the next batch of data.
        :param item: number of batch
        :return: dictionary for inputs and outputs
        """
        indices = self.index_pairwise[
                  self.batch_size * item:self.batch_size * item + self.batch_size]
        indices_output = self.index_pairwise_output[
                         self.batch_size * item:self.batch_size * item + self.batch_size]
        assert len(indices) == len(indices_output)
        if len(indices) == 0:
            raise ValueError(
                "DataSet2D: Indices length is 0 with item {0} of {1}".format(item,
                                                                             self._number_index // self.batch_size))
        if len(indices) < self.batch_size:
            raise ValueError("DataSet2D: Batch size is too large, not enough data samples available.")
        data = [dict()] * self.batch_size
        for inputs, outputs in zip(enumerate(indices), enumerate(indices_output)):
            idx = inputs[0]
            ind = [[inputs[1][0], outputs[1][0]], [inputs[1][1], outputs[1][1]]]
            data[idx] = self._next_data_item(ind)

        inputs = {}
        outputs = {}
        for input_name in self._input_name:
            inputs[input_name] = np.stack([data[idx][input_name] for idx in range(len(data))]).astype(np.float32)
        for output_name in self._output_name:
            if output_name in [v for k, v in self._mapping_data_name.items() if k in ["vs_class", "cochlea_class",
                                                                                      "vs_class_2", "cochlea_class_2"]]:
                outputs[output_name] = np.stack([data[idx][output_name] for idx in range(len(data))]).astype(np.int8)
            else:
                outputs[output_name] = np.stack([data[idx][output_name] for idx in range(len(data))]).astype(np.float32)
        return inputs, outputs

    def _load_data_item(self, ds, item):
        """
        Load data item, normalize images, resize image and segm mask.
        :param ds: index for patient data
        :param item: index for slice
        :return: data dictionary
        """
        data = super(DataSet2DMixed, self)._load_data_item(ds[0], item[0])
        # paired segm mask
        for output_name, output_data in zip(self._output_name, self._output_data):
            if output_data in ["vs", "cochlea"]:
                data[output_name] = self._load_data_sample(ds[0], output_data, item[0])
            if output_data in ["vs_class", "cochlea_class"]:
                data[output_name] = self._load_data_sample(ds[0], output_data, item[0])
        # unpaired image and corresponding segm mask (for validation)
        if "t1" in self._output_data or "t2" in self._output_data:
            tmp = []
            tmp_class = []
            tmp2 = []
            tmp2_class = []
            for output_name, output_data in zip(self._output_name, self._output_data):
                if output_data in ["t1", "t2"]:
                    data[output_name] = self._load_data_sample(ds[1], output_data, item[1])
                elif output_data in ["vs", "cochlea"]:
                    output_name = output_name + "_2"
                    tmp.append(output_name)
                    tmp2.append(output_data + "_2")
                    data[output_name] = self._load_data_sample(ds[1], output_data, item[1])
                elif output_data in ["vs_class", "cochlea_class"]:
                    output_name = output_name + "_2"
                    tmp_class.append(output_name)
                    tmp2_class.append(output_data + "_2")
                    data[output_name] = self._load_data_sample(ds[1], output_data, item[1])
            if len(tmp) != 0 and tmp[0] not in self._output_name:
                self._output_name.append(*tmp)
                for k, v in zip(tmp2, tmp):
                    self._mapping_data_name[k] = v
            if len(tmp_class) != 0 and tmp_class[0] not in self._output_name:
                self._output_name.append(*tmp_class)
                for k, v in zip(tmp2_class, tmp_class):
                    self._mapping_data_name[k] = v
        return data

    def reset(self):
        """
        Reset indices and shuffle randomly if necessary.
        """
        self.index_pairwise = [(idx, n) for idx, d in enumerate(self._data) for n in range(len(d))] if len(
            self.list_index) == 0 else self.list_index
        self.index_pairwise_output = self.index_pairwise
        if not self._paired:
            self.index_pairwise_output = np.random.permutation(self.index_pairwise_output)
        if self._shuffle:
            self.index_pairwise = np.random.permutation(self.index_pairwise)
            if self._paired:
                self.index_pairwise_output = self.index_pairwise

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
