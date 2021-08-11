########################################################################################################################
# Inference for Cycle GAN + Segmentation Pipeline
########################################################################################################################
import cv2
import tensorflow as tf
import numpy as np
from heapq import nlargest, nsmallest

from data_utils.data_visualization import plot_predictions_separate_overlap
from losses.dice import DiceLoss, DiceCoefficient
from losses.gan import generator_loss_lsgan
import medpy.metric as metric

__author__ = "c.magg"


class InferenceGT2SSegmS:
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
        # load models
        self._generator_dir = G_T2S_dir
        self._segm_dir = segm_dir
        self._discriminator_dir = D_S_dir
        self.G_T2S = tf.keras.models.load_model(G_T2S_dir)
        self.D_S = tf.keras.models.load_model(D_S_dir)
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
        # self.source_data = tf.expand_dims(self._data_gen[0][0]["image"], -1)
        # self.target_data = self._data_gen[0][0]["generated_t2"]
        # self.segmentation = self._data_gen[0][1]["vs"]

    def infer(self, inputs, reference=None):
        """
        Create prediction of inputs and discriminator results if reference is given.
        """
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
        segmentation = self.segmentor.predict((S_generated + 1) / 2 * 255)
        return S_generated, segmentation, S_d_gen, S_d_gt

    def evaluate(self, opt_batch_size=5):
        """
        Evaluate inference pipeline
        """
        dice, assd, S_pred, segm_pred, S_pred_d, S_gt_d, S_inputs, _, _ = self._evaluate(opt_batch_size)
        mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(S_pred, tf.expand_dims(S_inputs, -1)))
        d_loss = generator_loss_lsgan(S_pred_d) if S_gt_d is not None else None

        result = {'MSE_T2S': mse_loss.numpy(),
                  'G_loss': d_loss.numpy(),
                  'dice_coeff_mean': np.nanmean(dice),
                  'dice_coeff_std': np.nanstd(dice),
                  'assd_mean': np.nanmean(assd),
                  'assd_std': np.nanstd(assd)}
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
        S_pred = []
        segm_pred = []
        S_pred_d = []
        S_gt_d = []
        S_inputs = []
        T_inputs = []
        segm_gt = []
        dc = []
        assd = []
        for idx in range(len(self._data_gen)):
            inputs, outputs = self._data_gen[idx]
            T2S, y_pred, T2S_d, S_d = self.infer(inputs["generated_t2"], inputs["image"])
            for idx in range(len(T2S)):
                thres, y_pred_thres = cv2.threshold(y_pred[idx][:, :, 0], 0.5, 1, cv2.THRESH_BINARY)
                S_pred.append(T2S[idx])
                segm_pred.append(y_pred_thres)
                S_pred_d.append(T2S_d[idx])
                S_gt_d.append(S_d[idx])
                S_inputs.append(inputs["image"][idx])
                T_inputs.append(inputs["generated_t2"][idx])
                segm_gt.append(outputs["vs"][idx])
                dc.append(metric.binary.dc(y_pred_thres, outputs["vs"][idx]))
                if np.sum(y_pred_thres) != 0:
                    assd.append(metric.binary.assd(y_pred_thres, outputs["vs"][idx]))
                else:
                    assd.append(np.NAN)
        self._data_gen.batch_size = self._data_gen._number_index
        return dc, assd, S_pred, segm_pred, S_pred_d, S_gt_d, S_inputs, T_inputs, segm_gt

    def get_k_results(self, k=4, opt_batch_size=5, plot=True):
        """
        Get k best/worst results wrt Dice Coefficient.
        Optionally: plot the results of best/worst k results.
        """
        # calculate dice coeff
        dice, assd, S_pred, segm_pred, S_pred_d, S_gt_d, S_inputs, T_inputs, segm_gt = self._evaluate(opt_batch_size)

        # top k dice results
        dice_top = nlargest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [T_inputs[k[0]] for k in dice_top]
        inputs_gen = [S_pred[k[0]] for k in dice_top]
        targets = [segm_gt[k[0]] for k in dice_top]
        targets_gen = [S_inputs[k[0]] for k in dice_top]
        pred = [segm_pred[k[0]] for k in dice_top]
        assd_top = [assd[k[0]] for k in dice_top]
        print(f"best {k} dice: {dice_top} \nwith assd: {assd_top}")
        if plot:
            plot_predictions_separate_overlap(inputs, targets_gen, inputs_gen, targets, pred)

        # bottom k dice results
        dice_bottom = nsmallest(k, enumerate([d for d in dice]), key=lambda x: x[1])
        inputs = [T_inputs[k[0]] for k in dice_bottom]
        inputs_gen = [S_pred[k[0]] for k in dice_bottom]
        targets = [segm_gt[k[0]] for k in dice_bottom]
        targets_gen = [S_inputs[k[0]] for k in dice_bottom]
        pred = [segm_pred[k[0]] for k in dice_bottom]
        assd_bottom = [assd[k[0]] for k in dice_top]
        print(f"worst {k} dice: {dice_bottom} \nwith assd: {assd_bottom}")
        if plot:
            plot_predictions_separate_overlap(inputs, targets_gen, inputs_gen, targets, pred)
