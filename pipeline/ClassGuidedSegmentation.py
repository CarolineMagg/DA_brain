########################################################################################################################
# classification-guided Segmentation pipeline
########################################################################################################################
import logging
from time import time

import cv2
import numpy as np
import os

import tensorflow as tf

from data_utils.DataSet2DMixed import DataSet2DMixed
from losses.dice import DiceLoss, DiceCoefficient

__author__ = "c.magg"

from models.UNet_ClassGuided import UNet_ClassGuided
from pipeline.Checkpoint import Checkpoint


class ClassGuidedSegmentation:

    def __init__(self, data_dir, tensorboard_dir, checkpoints_dir, save_model_dir, sample_dir,
                 seed=13375, sample_step=1000, dsize=(256, 256)):
        """
        ClassGuidedSegmentation pipeline
        :param tensorboard_dir: directory for tensorboard logging
        :param save_model_dir: file path for saved models
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
            logging.info(f"CycleGan: create {sample_dir}.")
            os.makedirs(sample_dir)

        # parameters
        self.dsize = dsize
        self.sample_step = sample_step
        self.template = "{4}/{5} in {6:.4f} sec - Dice_loss: {0:.5f} - BCE_loss: {1:.5f} - Segm_loss: {2:.5f} - Dice_coeff: {3:.5f}"

        # data
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self._load_data()

        # model
        self.model = UNet_ClassGuided(input_shape=(*dsize, 1), input_name="image",
                                      output_name=["vs", "vs_class"]).generate_model()
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.losses = DiceLoss()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()
        self.metrics = DiceCoefficient()

    def _load_data(self):
        """
        Load data from training/validation/test folder with batch size 1, no augm and pixel value range [-1,1].
        Use only data where segmentation is available to ensure tumor presents.
        Training data - with shuffle, unpaired
        Validation data - with shuffle, paired
        Test data - without shuffle, paired
        """
        logging.info("ClassGuidedSegmentation: loading data ...")
        self.train_set = DataSet2DMixed(os.path.join(self.dir_data, "training"), input_data=["t1"],
                                        input_name=["image"], output_data=["vs", "vs_class"],
                                        output_name=["vs", "vs_class"],
                                        batch_size=1, shuffle=True, p_augm=0.0, alpha=0, beta=255,
                                        dsize=self.dsize)
        self.val_set = DataSet2DMixed(os.path.join(self.dir_data, "validation"), input_data=["t1"],
                                      input_name=["image"], output_data=["vs", "vs_class"],
                                      output_name=["vs", "vs_class"],
                                      batch_size=1, shuffle=False, p_augm=0.0, alpha=0, beta=255,
                                      dsize=self.dsize)
        self.val_set._unpaired = False
        self.val_set.reset()
        logging.info(
            "ClassGuidedSegmentation: training {0}, validation {1}".format(len(self.train_set), len(self.val_set)))

        self.test_set = DataSet2DMixed(os.path.join(self.dir_data, "test"), input_data=["t1"],
                                       input_name=["image"], output_data=["vs", "vs_class"],
                                       output_name=["vs", "vs_class"],
                                       batch_size=1, shuffle=False, p_augm=0.0, alpha=0, beta=255,
                                       dsize=self.dsize)
        self.test_set._unpaired = False
        self.test_set.reset()

        logging.info("ClassGuidedSegmentation: test {0}".format(len(self.test_set)))

    @tf.function
    def train_step(self, X, Y_seg, Y_class):
        """
        One training step.
        """
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(X, training=True)

            y_pred_seg = y_pred[:-1]
            y_pred_cls = y_pred[-1]

            segm_loss = self.losses(Y_seg, y_pred_seg)
            metric = self.metrics(Y_seg, y_pred_seg)
            class_loss = self.bce_loss(Y_class, y_pred_cls)
            loss = segm_loss + class_loss

        # calc and update gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        del tape

        return {"dice_loss": segm_loss,
                "bce_loss": class_loss,
                "segm_loss": segm_loss,
                "dice_coeff": metric}

    def _init_summary_file_writer(self, directory=os.path.join("logs", "train")):
        """
        Initialize the tensorboard summary file writer.
        """
        logging.info("ClassGuidedSegmentation: set up summary file writer with directory {}.".format(directory))
        self.train_summary_writer = tf.summary.create_file_writer(directory)

    def _init_checkpoint(self, directory=os.path.join("tmp", 'checkpoints'), restore=True):
        """
        Initialize the model checkpoints.
        """
        logging.info("ClassGuidedSegmentation: set up checkpoints with directory {}.".format(directory))
        self.checkpoint = Checkpoint(dict(Segm=self.model, Optimizer=self.optimizer),
                                     directory,
                                     max_to_keep=3)
        if restore:
            try:  # restore checkpoint including the epoch counter
                self.checkpoint.restore().assert_existing_objects_matched()
            except Exception as e:
                print("ClassGuidedSegmentation: " + e)

    @staticmethod
    def _collect_losses(loss_dict, loss_dict_list):
        """
        Collect losses from dict with one value to dict with list of values per epoch.
        """
        for k, v in loss_dict.items():
            if type(loss_dict_list[k]) == list:
                loss_dict_list[k].append(v.numpy())
            else:
                loss_dict_list[k] = [v.numpy()]
        return loss_dict_list

    def _save_models(self):
        """
        Save models to SavedModel.
        """
        logging.info("ClassGuidedSegmentation: save models to {}".format(self.dir_save_model))
        self.model.save(os.path.join(self.dir_save_model, "XNet_ClassGuided"))

    @tf.function
    def sample(self, X):
        """
        Generate samples from current segmentation network.
        """
        y_pred = self.model(X, training=False)
        return y_pred[:-1], y_pred[-1]

    def train(self, epochs=50, data_nr=None, restore=True, step_decay=None):
        """
        Train ClassGuidedSegmentation pipeline:
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
        logging.info("ClassGuidedSegmentation: set up training.")

        loss_dict_list = {k: 0 for k in ['dice_loss', 'dice_coeff', "bce_loss", "segm_loss"]}
        if data_nr is None or data_nr > len(self.train_set):
            data_nr = len(self.train_set)
        self._init_summary_file_writer(self.dir_tb)
        self._init_checkpoint(self.dir_cktp, restore=restore)
        sample_counter = 0

        logging.info("ClassGuidedSegmentation: start training.")
        for epoch in range(epochs + 1):
            total_time_per_epoch = 0
            print("Epoch {0}/{1}".format(epoch, epochs))
            for idx in range(data_nr):
                X, Y = self.train_set[idx]
                X = X["image"]
                Y_segm = Y["vs"]
                Y_class = Y["vs_class"]
                start = time()
                loss_dict = self.train_step(X, Y_segm, Y_class)
                elapsed = time() - start
                total_time_per_epoch += elapsed
                print(self.template.format(loss_dict["dice_loss"],
                                           loss_dict["bce_loss"],
                                           loss_dict["segm_loss"],
                                           loss_dict["dice_coeff"],
                                           idx, data_nr - 1,
                                           total_time_per_epoch),
                      end="\r")
                self.train_set.reset()
                loss_dict_list = self._collect_losses(loss_dict, loss_dict_list)
                if idx % self.sample_step == 0:
                    A, B = self.val_set[sample_counter]
                    A = A["image"]
                    B_segm = B["vs"]
                    B_class = B["vs_class"]
                    B_pred, _ = self.sample(A)
                    img = np.hstack(
                        np.concatenate([tf.expand_dims(A, -1), B_pred[0]*100, tf.expand_dims(B_segm, -1)*100], axis=0))
                    img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    cv2.imwrite(os.path.join(self.dir_sample, 'iter-%03d-%05d.jpg' % (epoch, idx)),
                                img)
                    sample_counter = sample_counter + 1
                    if sample_counter >= len(self.val_set):
                        sample_counter = 0

            with self.train_summary_writer.as_default():
                for k, v in loss_dict_list.items():
                    tf.summary.scalar(k, np.mean(v), step=epoch)
                tf.summary.scalar("learning_rate",
                                  self.optimizer.learning_rate,
                                  step=epoch)
            print(self.template.format(np.mean(loss_dict_list["dice_loss"]),
                                       np.mean(loss_dict_list["bce_loss"]),
                                       np.mean(loss_dict_list["segm_loss"]),
                                       np.mean(loss_dict_list["dice_coeff"]),
                                       data_nr - 1, data_nr - 1, total_time_per_epoch))
            self.checkpoint.save(epoch)

        self._save_models()
