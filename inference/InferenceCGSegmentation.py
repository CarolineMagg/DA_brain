########################################################################################################################
# Inference for CG Segmentation Pipeline
########################################################################################################################
import cv2
import tensorflow as tf
from heapq import nlargest, nsmallest
import numpy as np
from medpy import metric
from sklearn.metrics import confusion_matrix

from data_utils.data_visualization import plot_predictions_overlap

__author__ = "c.magg"

from inference.InferenceBase import InferenceBase

from losses.dice import DiceLoss, DiceCoefficient


class InferenceCGSegmentation(InferenceBase):
    """
    Inference of Classification-guided Segmentation network, eg. CGUNet
    """

    def __init__(self, saved_model_dir, data_gen):
        """
        Create Inference object.
        :param saved_model_dir: path to saved model
        :param data_gen: dataset generator
        """
        super(InferenceCGSegmentation, self).__init__(saved_model_dir, data_gen)

    def load_models(self):
        self.model = tf.keras.models.load_model(self._saved_model_dir,
                                                custom_objects={'DiceLoss': DiceLoss,
                                                                'DiceCoefficient': DiceCoefficient})

    def infer(self, inputs):
        """
        Create prediction of inputs.
        """
        inputs = inputs["image"]
        if not isinstance(inputs, np.ndarray):
            inputs = np.stack(inputs)
        if len(np.shape(inputs)) == 2:
            inputs = tf.expand_dims(tf.expand_dims(inputs, 0), -1)
        return self.model.predict(inputs)

    def _evaluate(self, opt_batch_size=1):
        """
        Evaluate batch-wise (optimized to validation dataset, due to memory resources):
        * calculate Dice Coefficient (DC) and Average Symmetric Surface Area
        * generate T2S, y_pred, discriminator results for T2S and S
        Note: use y_pred with threshold 0.5 -> y_pred is binary, like y_gt
        (this might result in other DC values as observed during training)
        """
        self.data_gen.batch_size = opt_batch_size
        segm_pred = []
        segm_gt = []
        class_gt = []
        images = []
        dc = []
        surface_distance = []
        for idx in range(len(self.data_gen)):
            inputs, outputs = self.data_gen[idx]
            y_pred, y_class = self.infer(inputs)
            for idx in range(len(y_pred)):
                dice, assd, y_pred_thres = self._process_segm_result(y_pred[idx][:, :, 0], outputs["vs"][idx])
                images.append(inputs["image"][idx])
                segm_gt.append(outputs["vs"][idx])
                class_gt.append(outputs["vs_class"][idx])
                segm_pred.append(y_pred_thres)
                dc.append(dice)
                surface_distance.append(assd)
        class_pred = [1 if np.sum(x) != 0 else 0 for x in segm_pred]
        tn, fp, fn, tp = confusion_matrix(class_gt, class_pred, labels=[0, 1]).ravel()
        self.data_gen.batch_size = self.data_gen._number_index
        return dc, surface_distance, segm_pred, images, segm_gt, tn, fp, fn, tp, class_pred, class_gt
