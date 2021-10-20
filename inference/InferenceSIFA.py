########################################################################################################################
# Inference for Cycle GAN + Segmentation Pipeline
########################################################################################################################
import cv2
import tensorflow as tf
import numpy as np
from heapq import nlargest, nsmallest

from sklearn.metrics import confusion_matrix

from data_utils.data_visualization import plot_predictions_separate_overlap, plot_predictions_separate
from losses.dice import DiceLoss, DiceCoefficient
from losses.gan import generator_loss_lsgan
import medpy.metric as metric

__author__ = "c.magg"


class InferenceSIFA:
    """
    Inference of GT2S and Segmentation in S domain.
    workflow:
    """

    def __init__(self, encoder_dir, segm_dir, data_gen):
        """
        Create Inference object
        :param G_T2S_dir: path to Generator T2S
        :param segm_dir: path to segmentation network
        :param D_S_dir: path to Discriminator S
        :param data_gen: dataset generator
        """
        # load models
        self._encoder_dir = encoder_dir
        self._segm_dir = segm_dir
        self.encoder = tf.keras.models.load_model(encoder_dir)
        self.segmentor = tf.keras.models.load_model(segm_dir,
                                                    custom_objects={'DiceLoss': DiceLoss,
                                                                    'DiceCoefficient': DiceCoefficient})
        # generate data
        self._data_gen = data_gen
        self._data_gen.p_augm = 0.0
        self._data_gen._alpha = -1
        self._data_gen._beta = 1
        self._data_gen.shuffle = False
        self._data_gen.reset()

    def infer(self, inputs):
        """
        Create prediction of inputs and discriminator results if reference is given.
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.stack(inputs)
        if len(np.shape(inputs)) == 2:
            inputs = tf.expand_dims(tf.expand_dims(inputs, 0), -1)
        S_feature = self.encoder.predict(inputs)
        segmentation = self.segmentor.predict(S_feature)
        return segmentation

    def evaluate(self, opt_batch_size=5, do_print=True):
        """
        Evaluate inference pipeline
        """
        dice, assd, segm_pred, T2_inputs, segm_gt, tn, fp, fn, tp = self._evaluate(opt_batch_size)

        result = {'dice_coeff_mean': np.nanmean(dice),
                  'dice_coeff_std': np.nanstd(dice),
                  'assd_mean': np.nanmean(assd),
                  'assd_std': np.nanstd(assd),
                  'tp': tp,
                  'fp': fp,
                  'fn': fn,
                  'tn': tn,
                  }
        if do_print:
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
        self._data_gen.batch_size = opt_batch_size
        segm_pred = []
        T2_inputs = []
        segm_gt = []
        dc = []
        assd = []
        for idx in range(len(self._data_gen)):
            inputs, outputs = self._data_gen[idx]
            y_pred = self.infer(inputs["image"])
            for idx in range(len(y_pred)):
                thres, y_pred_thres = cv2.threshold(y_pred[idx][:, :, 0], 0.5, 1, cv2.THRESH_BINARY)
                segm_pred.append(y_pred_thres)
                T2_inputs.append(inputs["image"][idx])
                segm_gt.append(outputs["vs"][idx])
                dc.append(metric.binary.dc(y_pred_thres, outputs["vs"][idx]))
                if np.sum(y_pred_thres) != 0:
                    assd.append(metric.binary.assd(y_pred_thres, outputs["vs"][idx]))
                else:
                    assd.append(np.NAN)
        class_gt = [1 if np.sum(x) != 0 else 0 for x in segm_gt]
        class_pred = [1 if np.sum(x) != 0 else 0 for x in segm_pred]
        tn, fp, fn, tp = confusion_matrix(class_gt, class_pred, labels=[0, 1]).ravel()
        self._data_gen.batch_size = self._data_gen._number_index
        return dc, assd, segm_pred, T2_inputs, segm_gt, tn, fp, fn, tp

    def get_k_results(self, k=4, opt_batch_size=5, do_plot=True):
        """
        Get k best/worst results wrt Dice Coefficient.
        Optionally: plot the results of best/worst k results.
        """
        # calculate dice coeff
        dice, assd, segm_pred, T2_inputs, segm_gt, _, _, _, _ = self._evaluate(opt_batch_size)

        # top k dice results
        dice_top = nlargest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [T2_inputs[k[0]] for k in dice_top]
        targets = [segm_gt[k[0]] for k in dice_top]
        pred = [segm_pred[k[0]] for k in dice_top]
        assd_top = [assd[k[0]] for k in dice_top]
        sz = [np.sum(s) for s in targets]
        sz_pred = [np.sum(s) for s in pred]
        print(f"best {k} dice: {dice_top} \nwith assd: {assd_top} \nwith original sz: {sz} \nwith pred sz: {sz_pred}")
        if do_plot:
            plot_predictions_separate(inputs, targets, pred)

        # bottom k dice results
        dice_bottom = nsmallest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [T2_inputs[k[0]] for k in dice_bottom]
        targets = [segm_gt[k[0]] for k in dice_bottom]
        pred = [segm_pred[k[0]] for k in dice_bottom]
        assd_bottom = [assd[k[0]] for k in dice_top]
        sz = [np.sum(s) for s in targets]
        sz_pred = [np.sum(s) for s in pred]
        print(f"worst {k} dice: {dice_bottom} \nwith assd: {assd_bottom} \nwith original sz: {sz} \nwith pred sz: {sz_pred}")
        if do_plot:
            plot_predictions_separate(inputs, targets, pred)
