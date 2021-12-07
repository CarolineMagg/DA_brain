########################################################################################################################
# Inference for Cycle GAN + Segmentation Pipeline
########################################################################################################################
import cv2
import tensorflow as tf
import numpy as np
from heapq import nlargest, nsmallest

from sklearn.metrics import confusion_matrix

from data_utils.data_visualization import plot_predictions_separate_overlap, plot_predictions_separate
from inference.InferenceBase import InferenceBase
from losses.dice import DiceLoss, DiceCoefficient
from losses.gan import generator_loss_lsgan
import medpy.metric as metric

__author__ = "c.magg"


class InferenceCGSIFA(InferenceBase):
    """
    Inference of SIFA.
    workflow: T -> Encoder -> Segm -> Y_T
    """

    def __init__(self, encoder_dir, segm_dir, data_gen):
        """
        Create Inference object
        :param encoder_dir: path to Encoder
        :param segm_dir: path to segmentation network
        :param data_gen: dataset generator
        """
        self.encoder = None
        self.segmentor = None
        super(InferenceCGSIFA, self).__init__([encoder_dir, segm_dir], data_gen)

    def load_models(self):
        self.encoder = tf.keras.models.load_model(self._saved_model_dir[0])
        self.segmentor = tf.keras.models.load_model(self._saved_model_dir[1],
                                                    custom_objects={'DiceLoss': DiceLoss,
                                                                    'DiceCoefficient': DiceCoefficient})

    def infer(self, inputs):
        """
        Create prediction of inputs and discriminator results if reference is given.
        """
        if type(inputs) == dict:
            inputs = inputs["image"]
        if not isinstance(inputs, np.ndarray):
            inputs = np.stack(inputs)
        if len(np.shape(inputs)) == 2:
            inputs = tf.expand_dims(tf.expand_dims(inputs, 0), -1)
        S_feature = self.encoder.predict(inputs)
        pred = self.segmentor.predict(S_feature)
        print(pred)
        return pred["vs"]
