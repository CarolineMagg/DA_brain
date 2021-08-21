########################################################################################################################
# Segm based on GS2T pipeline
########################################################################################################################
import logging
import os
import random
from time import time

import cv2
import numpy as np
import tensorflow as tf

from data_utils.DataSet2DMixed import DataSet2DMixed
from losses.dice import DiceLoss, DiceCoefficient
from losses.gan import generator_loss_lsgan, cycle_consistency_loss, identity_loss, discriminator_loss_lsgan
from models import XNet, UNet
from models.ResnetGenerator import ResnetGenerator
from models.ConvDiscriminator import ConvDiscriminator
from pipelines.Checkpoint import Checkpoint
from pipelines.LinearDecay import LinearDecay

__author__ = "c.magg"


class SegmS2T:

    def __init__(self, data_dir, tensorboard_dir, checkpoints_dir, save_model_dir, sample_dir,
                 seed=13375, d_step=1, sample_step=500, segm_epoch=10,
                 cycle_loss_weight=10.0, identity_loss_weight=1.0, activation="relu",
                 dsize=(256, 256), batch_size=1, model_type="XNet"):
        """
        SegmS2T pipeline, ie segmentation training on synthetic T data.
        Uses pre-trained and frozen CycleGAN GS2T generator to generate synthetic data.
        """
        # random seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # directories
        self.dir_data = data_dir
        self.dir_tb = tensorboard_dir
        self.dir_cktp = checkpoints_dir
        self.dir_save_model = save_model_dir if save_model_dir is not None else ""
        self.dir_sample = sample_dir
        if not os.path.isdir(sample_dir):
            logging.info(f"SegmS2T: create {sample_dir}.")
            os.makedirs(sample_dir)

        # parameters
        self.cycle_loss_weight = cycle_loss_weight
        self.identity_loss_weight = identity_loss_weight
        self.d_step = d_step
        self.sample_step = sample_step
        self.dsize = dsize
        self.segm_epoch = segm_epoch
        self.batch_size = batch_size

        # data
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self._load_data()

        # generator
        self.G_S2T = tf.keras.models.load_model("/tf/workdir/DA_brain/saved_models/gan_13785/G_S2T")

        # segmentation
        if model_type == "XNet":
            logging.info("SegmS2T: using XNet.")
            self.model = XNet(input_shape=(256, 256, 1), output_classes=1,
                              filter_depth=(32, 64, 128, 256), activation=activation).generate_model()
        elif model_type == "UNet":
            logging.info("SegmS2T: using UNet.")
            self.model = UNet(input_shape=(256, 256, 1), output_classes=1,
                              filter_depth=(32, 64, 128, 256), activation=activation).generate_model()
        else:
            raise NotImplementedError("SegmS2T: 'model_type' not implemented.")

        # optimizer
        self.optimizer = None
        self._set_optimizer()

        # fake image history
        self._pool_size = 50
        self._fake_S = np.zeros((self._pool_size, 1, 256, 256, 1))
        self._fake_T = np.zeros((self._pool_size, 1, 256, 256, 1))
        self._num_fake = 0

        # checkpoints and template
        self.template = "{2}/{3} in {4:.4f} sec - Dice_loss: {0:.5f} - Dice_Coeff: {1:.5f}"
        self.checkpoint = None
        self.train_summary_writer = None

    def _set_optimizer(self, total_steps=50000, step_decay=25000):
        """
        Set optimizer with linear LR scheduler
        """
        lr_scheduler = LinearDecay(0.001, total_steps, step_decay)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    def _load_data(self):
        """
        Load data from training/validation/test folder with batch size, no augm and pixel value range [-1,1].
        Use only data where segmentation is available to ensure tumor presents.
        Training data - with shuffle, unpaired
        Validation data - without shuffle, paired
        """
        logging.info("SegmS2T: loading data ...")
        self.train_set = DataSet2DMixed(os.path.join(self.dir_data, "training"), input_data=["t1"],
                                        input_name=["image"], output_data=["t2", "vs"],
                                        output_name=["t2", "vs"],
                                        batch_size=self.batch_size, shuffle=True, p_augm=0.0,
                                        alpha=-1, beta=1, use_filter="vs",
                                        dsize=self.dsize)
        self.val_set = DataSet2DMixed(os.path.join(self.dir_data, "validation"), input_data=["t1"],
                                      input_name=["image"], output_data=["t2", "vs"],
                                      output_name=["t2", "vs"],
                                      batch_size=1, shuffle=False, p_augm=0.0,
                                      alpha=-1, beta=1, use_filter="vs",
                                      dsize=self.dsize)
        self.val_set._unpaired = False
        self.val_set.reset()
        logging.info("SegmS2T: training {0}, validation {1}".format(len(self.train_set), len(self.val_set)))

    @tf.function
    def train_step(self, S, S_mask):
        """
        Segmentation network training consists of following steps:
        1) generate synthetic T images
        2) train segm networks
        3) loss calculation with dice coefficient
        4) update gradients
        """
        # generate synthetic T images
        S2T = self.G_S2T(S, training=False)
        with tf.GradientTape() as tape:
            # segmentation with synthetic T
            pred = self.model(S2T, training=True)

            # loss and metrics
            dice_loss = DiceLoss()(S_mask, pred)
            dice_coeff = DiceCoefficient()(S_mask, pred)

        # calc and update gradients
        S_grad = tape.gradient(dice_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(S_grad, self.model.trainable_variables))

        return {'dice_loss': dice_loss, 'dice_coeff': dice_coeff}

    def _init_summary_file_writer(self, directory=os.path.join("logs", "train")):
        """
        Initialize the tensorboard summary file writer.
        """
        logging.info("SegmS2T: set up summary file writer with directory {}.".format(directory))
        self.train_summary_writer = tf.summary.create_file_writer(directory)

    def _init_checkpoint(self, directory=os.path.join("tmp", 'checkpoints'), restore=True):
        """
        Initialize the model checkpoints.
        """
        logging.info("SegmS2T: set up checkpoints with directory {}.".format(directory))
        self.checkpoint = Checkpoint(dict(SegmS2T=self.model,
                                          Segm_optimizer=self.optimizer),
                                     directory,
                                     max_to_keep=3)
        if restore:
            try:  # restore checkpoint including the epoch counter
                self.checkpoint.restore().assert_existing_objects_matched()
            except Exception as e:
                print("SegmS2T: " + e)

    @staticmethod
    def _collect_losses(S_loss_dict, S_loss_dict_list):
        """
        Collect losses from dict with one value to dict with list of values per epoch.
        """
        for k, v in S_loss_dict.items():
            if type(S_loss_dict_list[k]) == list:
                S_loss_dict_list[k].append(v.numpy())
            else:
                S_loss_dict_list[k] = [v.numpy()]
        return S_loss_dict_list

    def _save_models(self):
        """
        Save models to SavedModel
        """
        logging.info("SegmS2T: save models to {}".format(self.dir_save_model))
        self.model.save(os.path.join(self.dir_save_model, "SegmS2T"))

    @tf.function
    def sample(self, S, T):
        """
        Generate samples from current generators.
        """
        S2T = self.G_S2T(S, training=False)
        T_segm = self.model(T, training=False)
        S2T_segm = self.model(S2T, training=False)
        return S2T, S2T_segm, T_segm

    def train(self, epochs=50, data_nr=None, restore=True, step_decay=None):
        """
        Train SegmS2T pipeline:
        1) initialize local variables
        2) initialize tensorboard file writer and checkpoints
        3) initialize optimizer with LR scheduler
        4) for each epoch:
            for each step:
                draw next batch from training data
                perform training step
                collect the losses
                if sample generation step: generate sample
            write tensorboard summary
            save checkpoint
        5) save model
        """
        logging.info("SegmS2T: set up training.")

        S_loss_dict_list = {k: 0 for k in ['dice_loss', 'dice_coeff']}
        if data_nr is None or data_nr > len(self.train_set):
            data_nr = len(self.train_set)
        self._init_summary_file_writer(self.dir_tb)
        self._init_checkpoint(self.dir_cktp, restore=restore)
        if step_decay is None:
            step_decay = epochs // 2
        self._set_optimizer((epochs + 1) * data_nr, (step_decay + 1) * data_nr)
        sample_counter = 0

        logging.info("SegmS2T: start training.")
        for epoch in range(epochs + 1):
            total_time_per_epoch = 0
            print("Epoch {0}/{1}".format(epoch, epochs))
            for idx in range(data_nr):
                # load data
                S, T_ = self.train_set[idx]
                S = S["image"]
                S_mask = T_["vs"]
                # train step
                start = time()
                S_loss_dict = self.train_step(S, S_mask)
                elapsed = time() - start
                total_time_per_epoch += elapsed
                print(self.template.format(S_loss_dict["dice_loss"],
                                           S_loss_dict["dice_coeff"],
                                           idx, data_nr - 1,
                                           total_time_per_epoch),
                      end="\r")
                self.train_set.reset()
                # collect losses
                S_loss_dict_list = self._collect_losses(S_loss_dict, S_loss_dict_list)
                # sample
                if idx % self.sample_step == 0:
                    A, B_ = self.val_set[sample_counter]
                    A = A["image"]
                    B = B_["t2"]
                    A_mask = B_["vs"]
                    B_mask = B_["vs_2"]
                    S2T, S2T_segm, T_segm = self.sample(A, B)
                    img = np.hstack(
                        np.concatenate(
                            [tf.expand_dims(A, -1), tf.expand_dims(A + A_mask * 3, -1), S2T + S2T_segm * 3,
                             tf.expand_dims(B, -1), tf.expand_dims(B, -1) + T_segm * 3,
                             tf.expand_dims(B + B_mask * 3, -1)], axis=0))
                    img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    cv2.imwrite(os.path.join(self.dir_sample, 'iter-%03d-%05d.jpg' % (epoch, idx)),
                                img)
                    sample_counter = sample_counter + 1
                    if sample_counter >= len(self.val_set):
                        sample_counter = 0

            # tensorboard summary
            with self.train_summary_writer.as_default():
                for k, v in S_loss_dict_list.items():
                    tf.summary.scalar(k, np.mean(v), step=epoch)
                tf.summary.scalar("learning_rate",
                                  self.optimizer.learning_rate.current_learning_rate,
                                  step=epoch)
            print(self.template.format(np.mean(S_loss_dict_list["dice_loss"]),
                                       np.mean(S_loss_dict_list["dice_coeff"]),
                                       data_nr - 1, data_nr - 1, total_time_per_epoch))
            # checkpoint
            self.checkpoint.save(epoch)
        # save model
        self._save_models()
