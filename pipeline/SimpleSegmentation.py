########################################################################################################################
# Simple supervised Segmentation pipeline
########################################################################################################################
import logging

import tensorflow as tf
from losses.dice import DiceLoss, DiceCoefficient

__author__ = "c.magg"


class SimpleSegmentation:

    def __init__(self, tensorboard_dir, checkpoint_dir, save_model_dir):
        """
        SimpleSegmentation pipeline
        :param tensorboard_dir: directory for tensorboard logging
        :param save_model_dir: file path for saved models
        """
        # dataset
        self.train_set = None
        self.val_set = None
        self.test_set = None

        # callbacks
        self.callbacks = []
        self.dir_checkpoint = checkpoint_dir
        self.dir_save_model = save_model_dir
        self.dir_tb = tensorboard_dir
        self.set_callbacks()

        # model
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.losses = DiceLoss()
        self.metrics = DiceCoefficient()

    def set_callbacks(self):
        """
        Set following callbacks:
        * Early Stopping: based on val_loss with 15 epochs patients.
        * Model Checkpoint: save best model based on val_loss to hdf5 file
        * Tensorboard: log loss/metrics to TB
        """
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=20, verbose=1,
                                                     mode='min')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.dir_checkpoint, monitor="val_loss", verbose=1,
                                                        save_best_only=True, save_weights_only=False)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.dir_tb)

        self.callbacks = [earlystop,
                          checkpoint,
                          tensorboard]

    def set_model(self, model):
        """
        Set model and compile model with optimizer, loss, metric
        """
        self.model = model
        self.model.compile(optimizer=self.optimizer, loss=self.losses, metrics=self.metrics)

    def set_data(self, train, val, test):
        """
        Set training, validation and test data.
        """
        self.train_set = train
        self.val_set = val
        self.test_set = test

    def _check_for_training(self):
        """
        Check if everything is set.
        """
        if self.train_set is None or self.test_set is None or self.val_set is None:
            raise ValueError("SimpleSegmentation: set train/val/test datasets.")
        if self.model is None:
            raise ValueError("SimpleSegmentation: set model.")
        if len(self.callbacks) == 0:
            raise ValueError("SimpleSegmentation: set callbacks.")

    def fit(self, init_epoch=0, epochs=2):
        """
        Model fit with initial and total number of epochs.
        """
        logging.info("SimpleSegmentation: training ....")
        self._check_for_training()
        self.model.fit(self.train_set,
                       validation_data=self.val_set,
                       callbacks=self.callbacks,
                       initial_epoch=init_epoch,
                       epochs=epochs,
                       verbose=1)

    def evaluate(self):
        """
        Perform model evaluation on validation and test set with latest model.
        """
        logging.info("SimpleSegmentation: evaluation ....")
        self.model.load_weights(self.dir_checkpoint)
        self.model.evaluate(self.val_set)
        self.model.evaluate(self.test_set)

    def run(self, augm_step=True, epochs=None):
        """
        Train SimpleSegmentation pipeline:
        1) Check if everything is set
        2) Perform first training stage (without augmentation)
        3) Perform second training stage (with augmentation)
        4) Perform evaluation
        """
        if epochs is None:
            epochs = [0, 100, 150]
        elif epochs is not None and augm_step:
            if len(epochs) < 2:
                raise ValueError("SimpleSegmentation: epochs list must contain at least 3 entries.")

        logging.info("CycleGAN: set up training.")
        self._check_for_training()
        self.fit(init_epoch=epochs[0], epochs=epochs[1])

        if augm_step:
            self.train_set.p_augm = 0.5
            self.fit(init_epoch=epochs[1], epochs=epochs[2])

        self.evaluate()
        # self.model.load_weights(self.dir_checkpoint)
        # self.model.save(self.dir_save_model)
