########################################################################################################################
# Inference for Cycle GAN Generators
########################################################################################################################
import numpy as np
import tensorflow as tf
from heapq import nlargest, nsmallest

from inference.InferenceBase import InferenceBase
from losses.gan import generator_loss_lsgan

__author__ = "c.magg"


class InferenceGenerator(InferenceBase):
    """
    Inference of CycleGAN Generator.
    workflow: T -> GT2S -> S' or S -> G2ST -> T'
    """

    def __init__(self, G_dir, D_dir, data_gen, target="t2", source="t1"):
        """
        Create Inference object
        :param G_dir: path to Generator T2S
        :param segm_dir: path to segmentation network
        :param D_dir: path to Discriminator S
        :param data_gen: dataset generator
        """
        super(InferenceGenerator, self).__init__([G_dir, D_dir], data_gen)
        if self.data_gen._alpha != -1 and self.data_gen._beta != 1:
            raise ValueError("Dataset generator has wrong alpha, beta values.")
        self.source = source
        self.target = target

    def load_models(self):
        self.generator = tf.keras.models.load_model(self._saved_model_dir[0])
        self.discriminator = tf.keras.models.load_model(self._saved_model_dir[1])

    def infer(self, inputs, reference=False):
        """
        Create prediction of inputs and discriminator results if reference is given.
        """
        if reference:
            reference = inputs[self.target]
            inputs = inputs[self.source]
        else:
            reference = None
            inputs = inputs[self.source]
        if not isinstance(inputs, np.ndarray):
            inputs = np.stack(inputs)
        if len(np.shape(inputs)) == 2:
            inputs = tf.expand_dims(tf.expand_dims(inputs, 0), -1)
        S_generated = self.generator(inputs)
        S_d_gen = None
        S_d_gt = None
        if reference is not None:
            S_d_gen = self.discriminator(S_generated)
            S_d_gt = self.discriminator(reference)
        return S_generated, S_d_gen, S_d_gt

    def evaluate(self, opt_batch_size=1, do_print=True):
        """
        Evaluate inference pipeline
        """
        images, reference, S_pred, S_pred_d, S_gt_d = self._evaluate(opt_batch_size)
        mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(S_pred, tf.expand_dims(reference, -1)))
        gen_d_loss = generator_loss_lsgan(S_pred_d) if S_gt_d is not None else None
        d_mse_real = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.ones_like(S_gt_d), S_gt_d))
        d_mse_fake = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.zeros_like(S_pred_d), S_pred_d))
        result = {'MSE': mse_loss.numpy(),
                  'G_loss': gen_d_loss.numpy(),
                  'D_real': d_mse_real.numpy(),
                  'D_fake': d_mse_fake.numpy()
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
        S_pred_d = []
        S_gt_d = []
        reference = []
        images = []
        for idx in range(len(self.data_gen)):
            inputs, outputs = self.data_gen[idx]
            T2S, T2S_d, S_d = self.infer(inputs, True)
            for idx in range(len(T2S)):
                S_pred.append(T2S[idx])
                S_pred_d.append(T2S_d[idx])
                S_gt_d.append(S_d[idx])
                reference.append(inputs[self.target][idx])
                images.append(inputs[self.source][idx])
        self.data_gen.batch_size = self.data_gen._number_index
        return images, reference, S_pred, S_pred_d, S_gt_d

    def get_k_results(self, k=4, do_plot=True):
        """
        Get k best/worst results wrt Dice Coefficient.
        Optionally: plot the results of best/worst k results.
        """
        images, reference, S_pred, S_pred_d, S_gt_d = self._evaluate(1)
        mse_loss = tf.keras.losses.mean_squared_error(S_pred, tf.expand_dims(reference, -1))

        # top k dice results
        mse_top = nlargest(k, enumerate([tf.reduce_mean(d).numpy() for d in mse_loss]), key=lambda x: x[1])
        print(f"worst {k} mse: {mse_top}")

        # bottom k dice results
        mse_bottom = nsmallest(k, enumerate([tf.reduce_mean(d).numpy() for d in mse_loss]), key=lambda x: x[1])
        print(f"best {k} mse: {mse_bottom}")

