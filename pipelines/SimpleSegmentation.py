########################################################################################################################
# Simple supervised Segmentation pipeline
########################################################################################################################
import importlib
import logging
import albumentations as A
import cv2
import tensorflow as tf
import numpy as np

from data_utils.DataSet2DMixed import DataSet2DMixed
from losses.dice import DiceLoss, DiceCoefficient

__author__ = "c.magg"


class SimpleSegmentation:

    def __init__(self, tensorboard_dir, checkpoint_dir, save_model_dir, data_type, model_type,
                 batch_size=4, dsize=(256, 256), activation="relu", seed=13375):
        """
        SimpleSegmentation pipeline, eg supervised UNet training
        """
        # random seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # parameters
        self.batch_size = batch_size
        self.dsize = dsize
        self.activation = activation
        self.seed = seed
        self.data_type = data_type
        self.model_type = model_type

        # data
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self._load_data()

        # model
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.losses = DiceLoss()
        self.metrics = DiceCoefficient()
        self._load_model()

        # callbacks
        self.callbacks = []
        self.dir_checkpoint = checkpoint_dir
        self.dir_save_model = save_model_dir
        self.dir_tb = tensorboard_dir
        self._set_callbacks()

    def _set_callbacks(self):
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

    def _load_data(self):
        """
        Load data from training/validation folder with batch size, no augm and pixel value range [0,1].
        Use only data where segmentation is available to ensure tumor presents.
        Training data - with shuffle, paired
        Validation data - with shuffle, paired
        """

        if self.data_type == "t1":
            logging.info("SimpleSegmentation: Training vs with t1.")

            # dataset
            self.train_set = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/training/",
                                            input_data=["t1"], input_name=["image"],
                                            output_data="vs", output_name="vs",
                                            batch_size=self.batch_size, shuffle=True, use_filter="vs",
                                            dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)
            self.train_set.augm_methods = [
                A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
                A.VerticalFlip(p=0.5),
                A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                         A.MedianBlur(p=0.5, blur_limit=5),
                         A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
            ]
            self.val_set = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/validation/",
                                          input_data=["t1"], input_name=["image"],
                                          output_data="vs", output_name="vs", batch_size=self.batch_size, shuffle=True,
                                          use_filter="vs", dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)

        elif self.data_type == "t2":
            logging.info("SimpleSegmentation: Training vs with t2.")
            self.train_set = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/training/",
                                            input_data=["t2"], input_name=["image"],
                                            output_data="vs", output_name="vs",
                                            batch_size=self.batch_size, shuffle=True, use_filter="vs",
                                            dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)
            self.train_set.augm_methods = [
                A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
                A.VerticalFlip(p=0.5),
                A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                         A.MedianBlur(p=0.5, blur_limit=5),
                         A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
            ]
            self.val_set = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/validation/",
                                          input_data=["t2"], input_name=["image"],
                                          output_data="vs", output_name="vs",
                                          batch_size=self.batch_size, shuffle=True, use_filter="vs", dsize=self.dsize,
                                          p_augm=0.0, alpha=0, beta=1)

        elif self.data_type == "t1_t2":
            logging.info("SimpleSegmentation: Training vs with t1 and t2.")

            # dataset
            self.train_set = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/training/",
                                            input_data=["t1", "t2"], input_name=["image", "t2"],
                                            output_data="vs", output_name="vs",
                                            batch_size=self.batch_size, shuffle=True, use_filter="vs",
                                            dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)
            self.train_set.augm_methods = [
                A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
                A.VerticalFlip(p=0.5),
                A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                         A.MedianBlur(p=0.5, blur_limit=5),
                         A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
            ]
            self.val_set = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/validation/",
                                          input_data=["t1", "t2"], input_name=["image", "t2"],
                                          output_data="vs", output_name="vs",
                                          batch_size=self.batch_size, shuffle=True, use_filter="vs", dsize=self.dsize,
                                          p_augm=0.0, alpha=0, beta=1)

        else:
            raise ValueError(f"Training {self.data_type} not valid.")

        logging.info(
            "SimpleSegmentation: training {0}, validation {1}".format(len(self.train_set), len(self.val_set)))

    def _load_model(self):
        """
        Set model and compile model with optimizer, loss, metric
        """
        module = importlib.import_module("models")
        model_class = getattr(module, self.model_type)

        if self.data_type == "t1":
            self.model = model_class(activation=self.activation, input_name="image", output_name="vs",
                                     input_shape=(*self.dsize, 1),
                                     seed=self.seed).generate_model()
        elif self.data_type == "t2":
            self.model = model_class(activation=self.activation, input_name="image", output_name="vs",
                                     input_shape=(*self.dsize, 1),
                                     seed=self.seed).generate_model()
        elif self.data_type == "t1_t2":
            self.model = model_class(activation=self.activation, input_name=["image", "t2"], output_name="vs",
                                     input_shape=(*self.dsize, 1), seed=self.seed).generate_model()
        self.model.compile(optimizer=self.optimizer, loss=self.losses, metrics=self.metrics)

    def _fit(self, init_epoch=0, epochs=2):
        """
        Model fit with initial and total number of epochs.
        """
        logging.info("SimpleSegmentation: training ....")
        self.model.fit(self.train_set,
                       validation_data=self.val_set,
                       callbacks=self.callbacks,
                       initial_epoch=init_epoch,
                       epochs=epochs,
                       verbose=1)

    def _evaluate(self):
        """
        Perform model evaluation on validation and test set with latest model.
        """
        logging.info("SimpleSegmentation: evaluation ....")
        self.model.load_weights(self.dir_checkpoint)
        self.model.evaluate(self.val_set)

    def _save_models(self):
        """
        Save models to SavedModel
        """
        self.model.load_weights(self.dir_checkpoint)
        self.model.save(self.dir_save_model)

    def train(self, augm_step=True, epochs=None, p_augm=0.5):
        """
        Train SimpleSegmentation pipeline:
        1) Check if everything is set
        2) Perform first training stage (without augmentation)
        3) Perform second training stage (with augmentation)
        4) Perform evaluation
        """
        logging.info("SimpleSegmentation: set up training.")

        # check and set epochs
        if epochs is None:
            epochs = [0, 100, 150]
        elif epochs is not None and augm_step:
            if len(epochs) < 2:
                raise ValueError("SimpleSegmentation: epochs list must contain at least 3 entries.")

        # training
        self._fit(init_epoch=epochs[0], epochs=epochs[1])
        if augm_step:
            self.train_set.p_augm = p_augm
            self._fit(init_epoch=epochs[1], epochs=epochs[2])

        # final evaluation
        self._evaluate()
        # save model
        self._save_models()
