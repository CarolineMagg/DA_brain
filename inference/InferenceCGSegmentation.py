########################################################################################################################
# Inference for CG Segmentation Pipeline
########################################################################################################################
import cv2
import tensorflow as tf
from heapq import nlargest, nsmallest
import numpy as np
import medpy.metric as metric

from data_utils.data_visualization import plot_predictions_overlap

__author__ = "c.magg"

from losses.dice import DiceLoss, DiceCoefficient


class InferenceCGSegmentation:
    """
    Inference of Classification-guided Segmentation network, eg. CGUNet
    """

    def __init__(self, saved_model_dir, data_gen=None):
        """
        Create Inference object.
        :param saved_model_dir: path to saved model
        :param data_gen: dataset generator
        """
        # load model
        self._saved_model_dir = saved_model_dir
        self.model = tf.keras.models.load_model(saved_model_dir,
                                                custom_objects={'DiceLoss': DiceLoss,
                                                                'DiceCoefficient': DiceCoefficient})

        # generate data
        self.data_gen = data_gen
        self.data_gen.p_augm = 0.0
        self.data_gen._alpha = 0
        self.data_gen._beta = 1
        self.data_gen.shuffle = False
        self.data_gen.reset()

    def infer(self, inputs):
        """
        Create prediction of inputs.
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.stack(inputs)
        if len(np.shape(inputs)) == 2:
            inputs = tf.expand_dims(tf.expand_dims(inputs, 0), -1)
        return self.model.predict(inputs)

    def evaluate(self, opt_batch_size=5):
        """
        Evaluate inference pipeline
        """
        dice, assd, tp, tn, _, _, _, _, _ = self._evaluate(opt_batch_size)
        result = {'dice_coeff_mean': np.nanmean(dice),
                  'dice_coeff_std': np.nanstd(dice),
                  'assd_mean': np.nanmean(assd),
                  'assd_std': np.nanstd(assd),
                  'tp_mean': np.nanmean(tp),
                  'tp_std': np.nanstd(tp),
                  'tn_mean': np.nanmean(tn),
                  'tn_std': np.nanstd(tn),
                  }
        print(result)
        return result

    def _evaluate(self, opt_batch_size=5):
        """
        Evaluate batch-wise (optimized to validation dataset, due to memory resources):
        * calculate Dice Coefficient (DC) and Average Symmetric Surface Area
        * generate T2S, y_pred, discriminator results for T2S and S
        Note: use y_pred with threshold 0.5 -> y_pred is binary, like y_gt
        (this might result in other DC values as observed during training)
        """
        self.data_gen.batch_size = opt_batch_size
        segm_pred = []
        class_pred = []
        segm_gt = []
        class_gt = []
        images = []
        dc = []
        assd = []
        tp = []
        tn = []
        for idx in range(len(self.data_gen)):
            inputs, outputs = self.data_gen[idx]
            y_pred, y_class = self.infer(inputs["image"])
            for idx in range(len(y_pred)):
                thres, y_pred_thres = cv2.threshold(y_pred[idx][:, :, 0], 0.5, 1, cv2.THRESH_BINARY)
                images.append(inputs["image"][idx])
                segm_gt.append(outputs["vs"][idx])
                class_gt.append(outputs["vs_class"][idx])
                segm_pred.append(y_pred_thres)
                class_pred.append(y_class)
                dc.append(metric.binary.dc(y_pred_thres, outputs["vs"][idx]))
                if np.sum(y_pred_thres) != 0 and np.sum(outputs["vs"][idx]) != 0:
                    assd.append(metric.binary.assd(y_pred_thres, outputs["vs"][idx]))
                else:
                    assd.append(np.NAN)
                tp.append(metric.binary.true_positive_rate(y_class, outputs["vs_class"][idx]))
                tn.append(metric.binary.true_negative_rate(y_class, outputs["vs_class"][idx]))
        self.data_gen.batch_size = self.data_gen._number_index
        return dc, assd, tp, tn, segm_pred, class_pred, images, segm_gt, class_gt

    def get_k_results(self, k=4, plot=True):
        """
        Get k best/worst results and statistic with mean, median, std, max, min value of Dice Coefficient.
        Optionally: plot the results of best/worst k results.
        """
        dice, assd, tp, tn, segm_pred, class_pred, images, segm_gt, class_gt = self._evaluate()

        # top k dice results
        dice_top = nlargest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [images[k[0]] for k in dice_top]
        targets = [segm_gt[k[0]] for k in dice_top]
        pred = [segm_pred[k[0]] for k in dice_top]
        assd_top = [assd[k[0]] for k in dice_top]
        print(f"best {k} dice: {dice_top} \nwith assd: {assd_top}")
        if plot:
            plot_predictions_overlap(inputs, targets, pred)

        # bottom k dice results
        dice_bottom = nsmallest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [images[k[0]] for k in dice_bottom]
        targets = [segm_gt[k[0]] for k in dice_bottom]
        pred = [segm_pred[k[0]] for k in dice_bottom]
        assd_bottom = [assd[k[0]] for k in dice_bottom]
        print(f"worst {k} dice: {dice_bottom} \nwith assd: {assd_bottom}")
        if plot:
            plot_predictions_overlap(inputs, targets, pred)
