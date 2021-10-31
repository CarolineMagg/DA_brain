########################################################################################################################
# InferenceBase for Segmentation Pipeline
########################################################################################################################
import cv2
import tensorflow as tf
from heapq import nlargest, nsmallest
import numpy as np
import medpy.metric as metric
from sklearn.metrics import confusion_matrix

from data_utils.data_visualization import plot_predictions_overlap
from losses.dice import DiceLoss, DiceCoefficient

__author__ = "c.magg"


class InferenceBase:
    """
    Inference of SimpleSegmentation network, eg. UNet or XNet
    """

    def __init__(self, saved_model_dir, data_gen):
        """
        Create InferenceBase object.
        :param saved_model_dir: path to saved model
        :param data_gen: dataset generator
        """
        # load model
        self._saved_model_dir = saved_model_dir
        self.model = None
        self.load_models()

        # generate data
        self.data_gen = data_gen
        self.data_gen.shuffle = False
        self.data_gen.reset()

    def load_models(self):
        raise NotImplementedError("Load Model is not implemented.")

    def infer(self, inputs):
        """
        Create prediction of inputs.
        """
        raise NotImplementedError("Infer not implemented.")

    def evaluate(self, opt_batch_size=1, do_print=True):
        """
        Evaluate inference pipeline
        """
        dice, assd, segm_pred, _, _, tn, fp, fn, tp, _, _ = self._evaluate(opt_batch_size)
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
        images = []
        dc = []
        surface_distance = []
        for idx in range(len(self.data_gen)):
            inputs, outputs = self.data_gen[idx]
            y_pred = self.infer(inputs)
            for idx in range(len(y_pred)):
                dice, assd, y_pred_thres = self._process_segm_result(y_pred[idx][:, :, 0], outputs["vs"][idx])
                images.append(inputs["image"][idx])
                segm_gt.append(outputs["vs"][idx])
                segm_pred.append(y_pred_thres)
                dc.append(dice)
                surface_distance.append(assd)
        class_gt = [1 if np.sum(x) != 0 else 0 for x in segm_gt]
        class_pred = [1 if np.sum(x) != 0 else 0 for x in segm_pred]
        tn, fp, fn, tp = confusion_matrix(class_gt, class_pred, labels=[0, 1]).ravel()
        self.data_gen.batch_size = self.data_gen._number_index
        return dc, surface_distance, segm_pred, images, segm_gt, tn, fp, fn, tp, class_pred, class_gt

    def _process_segm_result(self, y_pred, gt):
        thres, y_pred_thres = cv2.threshold(y_pred, 0.5, 1, cv2.THRESH_BINARY)
        dice = 0.0
        assd = 362.0
        if np.sum(gt) != 0 and np.sum(y_pred_thres) != 0:
            dice = metric.binary.dc(y_pred_thres, gt)
            assd = metric.binary.assd(y_pred_thres, gt)
        elif np.sum(gt) == 0 and np.sum(y_pred_thres) == 0:
            dice = 1.0
            assd = 0.0
        return dice, assd, y_pred_thres

    def get_k_results(self, k=4, do_plot=True):
        """
        Get k best/worst results and statistic with mean, median, std, max, min value of Dice Coefficient.
        Optionally: plot the results of best/worst k results.
        """
        dice, assd, segm_pred, images, segm_gt, _, _, _, _, _, _ = self._evaluate()

        # top k dice results
        dice_top = nlargest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [images[k[0]] for k in dice_top]
        targets = [segm_gt[k[0]] for k in dice_top]
        pred = [segm_pred[k[0]] for k in dice_top]
        sz = [np.sum(s) for s in targets]
        sz_pred = [np.sum(s) for s in pred]
        assd_top = [assd[k[0]] for k in dice_top]
        print(f"best {k} dice: {dice_top} \nwith assd: {assd_top} \nwith original sz: {sz} \nwith pred sz: {sz_pred}")
        if do_plot:
            plot_predictions_overlap(inputs, targets, pred)

        # bottom k dice results
        dice_bottom = nsmallest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [images[k[0]] for k in dice_bottom]
        targets = [segm_gt[k[0]] for k in dice_bottom]
        pred = [segm_pred[k[0]] for k in dice_bottom]
        sz = [np.sum(s) for s in targets]
        sz_pred = [np.sum(s) for s in pred]
        assd_bottom = [assd[k[0]] for k in dice_bottom]
        print(
            f"worst {k} dice: {dice_bottom} \nwith assd: {assd_bottom} \nwith original sz: {sz} \nwith pred sz: {sz_pred}")
        if do_plot:
            plot_predictions_overlap(inputs, targets, pred)
