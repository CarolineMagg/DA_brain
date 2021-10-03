########################################################################################################################
# Simple supervised Segmentation pipeline
########################################################################################################################
import logging
import albumentations as A
import cv2
import tensorflow as tf

from data_utils.DataSet2DMixed import DataSet2DMixed
from losses.dice import DiceCoefficient, DiceLoss
from models import XNet_ClassGuided, UNet_ClassGuided
from pipelines.SimpleSegmentation import SimpleSegmentation

__author__ = "c.magg"


class CGSegmentation(SimpleSegmentation):

    def __init__(self, tensorboard_dir, checkpoint_dir, save_model_dir, data_type, model_type,
                 batch_size=4, dsize=(256, 256), use_balance=True, activation="relu", seed=13375):
        """
        CGSegmentation pipeline, eg supervised UNet training
        """
        self.use_balance = use_balance
        super(CGSegmentation, self).__init__(tensorboard_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
                                             save_model_dir=save_model_dir, data_type=data_type,
                                             model_type=model_type, batch_size=batch_size, dsize=dsize,
                                             activation=activation, seed=seed)

    def _load_data(self):
        """
        Load data from training/validation folder with batch size, no augm and pixel value range [0,1].
        Use only data where segmentation is available to ensure tumor presents.
        Training data - with shuffle, paired
        Validation data - with shuffle, paired
        """

        logging.info(f"CGSegmentation: Training vs with {self.data_type}.")

        # dataset
        self.train_set = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/training/",
                                        input_data=[self.data_type], input_name=["image"],
                                        output_data=["vs", "vs_class"], output_name=["vs", "vs_class"],
                                        use_balance=self.use_balance,
                                        batch_size=self.batch_size, shuffle=True,
                                        dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)
        self.train_set.augm_methods = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                     A.MedianBlur(p=0.5, blur_limit=5),
                     A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
        ]
        self.val_set = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/validation/",
                                      input_data=[self.data_type], input_name=["image"],
                                      output_data=["vs", "vs_class"], output_name=["vs", "vs_class"],
                                      batch_size=self.batch_size, shuffle=True,
                                      use_balance=self.use_balance, dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)

        logging.info(
            "CGSegmentation: training {0}, validation {1}".format(len(self.train_set), len(self.val_set)))

    def _load_model(self):
        """
        Set model and compile model with optimizer, loss, metric
        """
        if self.model_type == "XNet":
            self.model = XNet_ClassGuided(activation=self.activation, input_name="image",
                                          output_name=["vs", "vs_class"], input_shape=(*self.dsize, 1),
                                          seed=self.seed).generate_model()
        elif self.model_type == "UNet":
            self.model = UNet_ClassGuided(activation=self.activation, input_name="image",
                                          output_name=["vs", "vs_class"], input_shape=(*self.dsize, 1),
                                          seed=self.seed).generate_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                           loss={"vs": DiceLoss(), "vs_class": tf.keras.losses.BinaryCrossentropy()},
                           metrics={"vs": DiceCoefficient(), "vs_class": tf.keras.metrics.BinaryAccuracy()})
