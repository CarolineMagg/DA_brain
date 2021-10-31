########################################################################################################################
# Inference for Simple Segmentation Pipeline
########################################################################################################################
import cv2
import tensorflow as tf
from heapq import nlargest, nsmallest
import numpy as np
import medpy.metric as metric
from sklearn.metrics import confusion_matrix

from data_utils.data_visualization import plot_predictions_overlap
from inference.InferenceBase import InferenceBase
from losses.dice import DiceLoss, DiceCoefficient

__author__ = "c.magg"


class InferenceSimpleSegmentation(InferenceBase):
    """
    Inference of SimpleSegmentation network, eg. UNet or XNet
    """

    def __init__(self, saved_model_dir, data_gen=None):
        """
        Create Inference object.
        :param saved_model_dir: path to saved model
        :param data_gen: dataset generator
        """
        super(InferenceSimpleSegmentation, self).__init__(saved_model_dir, data_gen)
        self._multi_modal = False if len(self.data_gen._input_name) == 1 else True

    def load_models(self):
        try:
            self.model = tf.keras.models.load_model(self._saved_model_dir,
                                                    custom_objects={'DiceLoss': DiceLoss,
                                                                    'DiceCoefficient': DiceCoefficient})
        except:
            self.model = tf.keras.models.load_model(self._saved_model_dir)

    def infer(self, inputs):
        """
        Create prediction of inputs.
        """
        if not self._multi_modal:
            if type(inputs) == dict and "image" in inputs.keys():
                inputs = inputs["image"]
            if not isinstance(inputs, np.ndarray):
                inputs = np.stack(inputs)
            if len(np.shape(inputs)) == 2:
                inputs = tf.expand_dims(tf.expand_dims(inputs, 0), -1)
        return self.model.predict(inputs)
