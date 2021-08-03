########################################################################################################################
# DataSet for Tensorflow trainings pipeline with tf.data.Datasets
########################################################################################################################
import logging
import tensorflow as tf
import os
import albumentations as A
import cv2
import numpy as np

__author__ = "c.magg"

from data_utils.DataContainer import DataContainer


class DataSetGenerator:

    def __init__(self, dataset_folder,
                 input_data="t1", output_data=None,
                 batch_size=4, p_augm=0.0, shuffle=True,
                 n_shuffle=None, dsize=(256, 256), use_filter=None,
                 use_unpaired=False):
        # dataset folder
        self._path_dataset = dataset_folder
        self._folders = os.listdir(self._path_dataset)

        # other parameters
        self._batch_size = batch_size
        self._dsize = dsize
        self._p_augm = p_augm
        self._shuffle = shuffle
        self._shuffle_buffer_size = max(batch_size * 128, 1000) if n_shuffle is None else n_shuffle
        self._filter = use_filter
        self._use_unpaired = use_unpaired
        self._augm_methods = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                     A.MedianBlur(p=0.5, blur_limit=5),
                     A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
        ]
        self._augm_lookup = {}

        # data
        self._input_data = input_data if type(input_data) == list else [input_data]
        self._output_data = output_data if type(output_data) == list else [output_data]
        if output_data is None:
            self._output_data = None
        self._data = []
        self._load_data()

    @staticmethod
    def lookup_data_call():
        """
        Generate the mapping between data naming and data loading method in Dataset.
        :return lookup dictionary with data name as key and method name as value
        """
        return {'t1': 't1_scan_slice',
                't2': 't2_scan_slice',
                'vs': 'segm_vs_slice',
                'cochlea': 'segm_cochlea_slice'}

    @staticmethod
    def lookup_data_augm():
        """
        Generate the mapping between data naming and augm role in Dataset
        :return lookup dictionary with data name as key and albumentation role as value
        """
        return {'t1': 'image',
                't2': 'image',
                'vs': 'mask',
                'cochlea': 'mask'}

    def _load_data(self):
        """
        Load data from folders with DataContainer.
        """
        for f in self._folders:
            folder = os.path.join(self._path_dataset, f)
            self._data.append(DataContainer(folder))

    def generator_data(self, data_types):
        """
        Data generator
        :param data_types: data type that should be generator, see lookup_data_call() keys
        """
        for idx in range(len(self._data)):
            for item in range(len(self._data[idx])):
                tensors = {}
                for dt in data_types:
                    dt = dt.decode() if type(dt) == bytes else dt
                    tensor = getattr(self._data[idx], self.lookup_data_call()[dt])(item)
                    tensor = cv2.normalize(tensor, None, alpha=0, beta=255,
                                           norm_type=cv2.NORM_MINMAX)
                    tensor = cv2.resize(tensor, dsize=self._dsize,
                                        interpolation=cv2.INTER_CUBIC)
                    tensors[dt] = np.expand_dims(tensor, -1)
                yield tensors

    def generate_dataset(self):
        """
        Generate tf.data.Dataset with given settings
        """
        # 1.) generate dataset with inputs (and outputs)
        output_sign_input = ({dt: tf.TensorSpec(shape=(*self._dsize, 1), dtype=tf.float32) for dt in self._input_data})
        output_sign_output = ({dt: tf.TensorSpec(shape=(*self._dsize, 1), dtype=tf.float32) for dt in
                               self._output_data}) if self._output_data is not None else None
        dataset = tf.data.Dataset.from_generator(self.generator_data, args=[self._input_data],
                                                 output_signature=output_sign_input)
        if self._output_data is not None:
            output_dataset = tf.data.Dataset.from_generator(self.generator_data, args=[self._output_data],
                                                            output_signature=output_sign_output)
            if self._use_unpaired:
                logging.info("DataSetGenerator: target dataset is shuffled ...")
                output_dataset = output_dataset.shuffle(buffer_size=1000,
                                                        reshuffle_each_iteration=False)
            dataset = tf.data.Dataset.zip((dataset, output_dataset))

        # 2.) Filter dataset if needed
        if self._filter is not None:
            len_before = len(list(dataset.as_numpy_iterator()))
            logging.info(f"DataSetGenerator: filtering {len_before} data points for {self._filter}...")
            dataset = dataset.filter(self.filter_fn)
            len_after = len(list(dataset.as_numpy_iterator()))
            logging.info(
                f"DataSetGenerator: filtering removed {len_before - len_after} data points. remaining {len_after}.")

        # 3.) shuffle entire dataset
        if self._shuffle:
            logging.info(f"DataSetGenerator: shuffling with buffer size {self._shuffle_buffer_size}...")
            dataset = dataset.shuffle(self._shuffle_buffer_size)

        # 4.) process single (input, target) tuple
        if self._p_augm != 0:
            if self._output_data:
                logging.info("DataSetGenerator: processing with targets ...")
                dataset = dataset.map(self._processing_with_targets, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                logging.info("DataSetGenerator: processing without targets ...")
                dataset = dataset.map(self._processing_without_targets, num_parallel_calls=tf.data.AUTOTUNE)

        # 5.) batch + prefetch
        logging.info("DataSetGenerator: batching and prefetching ...")
        dataset = dataset.batch(self._batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def _processing_with_targets(self, inputs, targets):
        # output_sign = [tf.float32 for dt in self._input_data]+[tf.float32 for dt in self._output_data]
        inp = [inputs[k] for k in self._input_data] + [targets[k] for k in self._output_data]
        tout = [tf.float32] * len(inp)
        res = tf.numpy_function(self._augm_fn, inp=inp, Tout=tout)
        res_dict = {k: b for k, b in zip(self._input_data + self._output_data, res)}
        res1 = {k: v for k, v in res_dict.items() if k in self._input_data}
        res2 = {k: v for k, v in res_dict.items() if k in self._output_data}
        return res1, res2

    def _processing_without_targets(self, inputs):
        inp = [inputs[k] for k in self._input_data]
        tout = [tf.float32] * len(inp)
        res = tf.numpy_function(self._augm_fn, inp=inp, Tout=tout)
        return {k: b for k, b in zip(self._input_data, res)}

    def _augm_fn(self, *elem):
        if self._output_data is not None:
            data = {k: v for k, v in zip(self._input_data + self._output_data, elem)}
            ordered_keys = ["image"] + self._input_data[1:] + self._output_data
        else:
            data = {k: v for k, v in zip(self._input_data, elem)}
            ordered_keys = ["image"] + self._input_data[1:]
        data["image"] = data.pop(self._input_data[0])
        data_augm = A.Compose(self._augm_methods,
                              additional_targets=self.lookup_data_augm())(**data)

        return [data_augm[k] for k in ordered_keys]

    @tf.function
    def filter_fn(self, *elem):
        """
        Filter dataset for non-zero segm masks according to filter_arg segm mask
        """
        res = True
        if len(elem) > 1:
            filter_arg = self._filter
            res = not tf.reduce_all(tf.math.equal(elem[1][filter_arg], 0))
        return res

    def _clear_data(self):
        """
        Clear data from memory
        """
        self._data = []

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, to):
        self._batch_size = to

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, to):
        self._shuffle = to

    @property
    def dsize(self):
        return self._dsize

    @dsize.setter
    def dsize(self, to):
        self._dsize = to

    @property
    def p_augm(self):
        return self._p_augm

    @p_augm.setter
    def p_augm(self, to):
        self._p_augm = to

    @property
    def augm_methods(self):
        return self._augm_methods

    @augm_methods.setter
    def augm_methods(self, to):
        assert type(to) == list, "'augm_methods' needs to be a list of albumentations methods"
        self._augm_methods = to
