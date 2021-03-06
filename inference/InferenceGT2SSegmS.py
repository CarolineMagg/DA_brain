########################################################################################################################
# Inference for Cycle GAN + Segmentation Pipeline
########################################################################################################################
import cv2
import tensorflow as tf
import numpy as np
from heapq import nlargest, nsmallest

from sklearn.metrics import confusion_matrix

from data_utils.data_visualization import plot_predictions_separate_overlap
from inference.InferenceBase import InferenceBase
from losses.dice import DiceLoss, DiceCoefficient
from losses.gan import generator_loss_lsgan
import medpy.metric as metric

__author__ = "c.magg"


class InferenceGT2SSegmS(InferenceBase):
    """
    Inference of GT2S and Segmentation in S domain.
    workflow: T -> GT2S -> S' -> Segm_S -> Y_T
    """

    def __init__(self, G_T2S_dir, segm_dir, D_S_dir, data_gen):
        """
        Create Inference object
        :param G_T2S_dir: path to Generator T2S
        :param segm_dir: path to segmentation network
        :param D_S_dir: path to Discriminator S
        :param data_gen: dataset generator
        """
        super(InferenceGT2SSegmS, self).__init__([G_T2S_dir, segm_dir, D_S_dir], data_gen)
        if self.data_gen._alpha != -1 and self.data_gen._beta != 1:
            raise ValueError("Dataset generator has wrong alpha, beta values.")

    def load_models(self):
        self.G_T2S = tf.keras.models.load_model(self._saved_model_dir[0])
        self.D_S = tf.keras.models.load_model(self._saved_model_dir[2])
        self.segmentor = tf.keras.models.load_model(self._saved_model_dir[1],
                                                    custom_objects={'DiceLoss': DiceLoss,
                                                                    'DiceCoefficient': DiceCoefficient})

    def infer(self, inputs, reference=False):
        """
        Create prediction of inputs and discriminator results if reference is given.
        """
        if reference:
            reference = inputs["image"]
            inputs = inputs["t2"]
        else:
            reference = None
            inputs = inputs["t2"]
        if not isinstance(inputs, np.ndarray):
            inputs = np.stack(inputs)
        if len(np.shape(inputs)) == 2:
            inputs = tf.expand_dims(tf.expand_dims(inputs, 0), -1)
        S_generated = self.G_T2S(inputs)
        S_d_gen = None
        S_d_gt = None
        if reference is not None:
            S_d_gen = self.D_S(S_generated)
            S_d_gt = self.D_S(reference)
        segmentation = self.segmentor.predict((S_generated + 1) / 2)
        if type(segmentation) == list and len(segmentation) == 2:
            segmentation = segmentation[0]
        return S_generated, segmentation, S_d_gen, S_d_gt

    def evaluate(self, opt_batch_size=1, do_print=True):
        """
        Evaluate inference pipeline
        """
        dice, assd, segm_pred, images, segm_gt, tn, fp, fn, tp, _, _, reference, S_pred, S_pred_d, S_gt_d = self._evaluate(
            opt_batch_size)
        mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(S_pred, tf.expand_dims(reference, -1)))
        d_loss = generator_loss_lsgan(S_pred_d) if S_gt_d is not None else None
        result = {'dice_coeff_mean': np.nanmean(dice),
                  'dice_coeff_std': np.nanstd(dice),
                  'assd_mean': np.nanmean(assd),
                  'assd_std': np.nanstd(assd),
                  'tp': tp,
                  'fp': fp,
                  'fn': fn,
                  'tn': tn,
                  'MSE_T2S': mse_loss.numpy(),
                  'G_loss': d_loss.numpy(),
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
        S_pred = []
        segm_pred = []
        S_pred_d = []
        S_gt_d = []
        reference = []
        images = []
        segm_gt = []
        dc = []
        surface_distance = []
        for idx in range(len(self.data_gen)):
            inputs, outputs = self.data_gen[idx]
            T2S, y_pred, T2S_d, S_d = self.infer(inputs, True)
            for idx in range(len(T2S)):
                dice, assd, y_pred_thres = self._process_segm_result(y_pred[idx][:, :, 0], outputs["vs"][idx])
                S_pred.append(T2S[idx])
                segm_pred.append(y_pred_thres)
                S_pred_d.append(T2S_d[idx])
                S_gt_d.append(S_d[idx])
                reference.append(inputs["image"][idx])
                images.append(inputs["t2"][idx])
                segm_gt.append(outputs["vs"][idx])
                dc.append(dice)
                surface_distance.append(assd)
        class_gt = [1 if np.sum(x) != 0 else 0 for x in segm_gt]
        class_pred = [1 if np.sum(x) != 0 else 0 for x in segm_pred]
        tn, fp, fn, tp = confusion_matrix(class_gt, class_pred, labels=[0, 1]).ravel()
        self.data_gen.batch_size = self.data_gen._number_index
        return dc, surface_distance, segm_pred, images, segm_gt, tn, fp, fn, tp, class_pred, class_gt, reference, S_pred, S_pred_d, S_gt_d

    def get_k_results(self, k=4, do_plot=True):
        """
        Get k best/worst results wrt Dice Coefficient.
        Optionally: plot the results of best/worst k results.
        """
        dice, assd, segm_pred, images, segm_gt, _, _, _, _, _, _, reference, S_pred, S_pred_d, S_gt_d =  self._evaluate()

        # top k dice results
        dice_top = nlargest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [images[k[0]] for k in dice_top]
        inputs_gen = [S_pred[k[0]] for k in dice_top]
        targets = [segm_gt[k[0]] for k in dice_top]
        targets_gen = [reference[k[0]] for k in dice_top]
        pred = [segm_pred[k[0]] for k in dice_top]
        assd_top = [assd[k[0]] for k in dice_top]
        print(f"best {k} dice: {dice_top} \nwith assd: {assd_top}")
        if do_plot:
            plot_predictions_separate_overlap(inputs, targets_gen, inputs_gen, targets, pred)

        # bottom k dice results
        dice_bottom = nsmallest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [images[k[0]] for k in dice_bottom]
        inputs_gen = [S_pred[k[0]] for k in dice_bottom]
        targets = [segm_gt[k[0]] for k in dice_bottom]
        targets_gen = [reference[k[0]] for k in dice_bottom]
        pred = [segm_pred[k[0]] for k in dice_bottom]
        assd_bottom = [assd[k[0]] for k in dice_top]
        print(f"worst {k} dice: {dice_bottom} \nwith assd: {assd_bottom}")
        if do_plot:
            plot_predictions_separate_overlap(inputs, targets_gen, inputs_gen, targets, pred)
