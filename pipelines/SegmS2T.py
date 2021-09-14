########################################################################################################################
# Segm based on GS2T pipeline
########################################################################################################################
import logging

import cv2
import albumentations as A

from data_utils.DataSet2DSynthetic import DataSet2DSynthetic

from pipelines.SimpleSegmentation import SimpleSegmentation

__author__ = "c.magg"


class SegmS2T(SimpleSegmentation):

    def __init__(self, tensorboard_dir, checkpoint_dir, save_model_dir, data_type,
                 model_type, cycle_gan_dir, batch_size=4, dsize=(256, 256), activation="relu", seed=13375):
        """
        SegmS2T pipeline, eg supervised UNet training.

        Note: very slow in training (!)
        """
        self.cycle_gan = cycle_gan_dir
        super(SegmS2T, self).__init__(tensorboard_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir,
                                      save_model_dir=save_model_dir, data_type=data_type,
                                      model_type=model_type, batch_size=batch_size, dsize=dsize, activation=activation,
                                      seed=seed)

    def _load_data(self):
        """
        Load data from training/validation folder with batch size, no augm and pixel value range [0,1].
        Use only data where segmentation is available to ensure tumor presents.
        Training data - with shuffle, paired
        Validation data - with shuffle, paired
        """

        if self.data_type == "t1":
            logging.info("SegmS2T: Training vs with t1.")

            # dataset
            self.train_set = DataSet2DSynthetic("/tf/workdir/data/VS_segm/VS_registered/training/",
                                                cycle_gan=self.cycle_gan,
                                                input_data=["t1"], input_name=["image"],
                                                output_data="vs", output_name="vs",
                                                batch_size=self.batch_size, shuffle=True, segm_size=0,
                                                dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)
            self.train_set.augm_methods = [
                A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
                A.VerticalFlip(p=0.5),
                A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                         A.MedianBlur(p=0.5, blur_limit=5),
                         A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
            ]
            self.val_set = DataSet2DSynthetic("/tf/workdir/data/VS_segm/VS_registered/validation/",
                                              cycle_gan=self.cycle_gan,
                                              input_data=["t1"], input_name=["image"],
                                              output_data="vs", output_name="vs", batch_size=self.batch_size,
                                              shuffle=True,
                                              segm_size=0, dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)

        elif self.data_type == "t2":
            logging.info("SegmS2T: Training vs with t2.")
            self.train_set = DataSet2DSynthetic("/tf/workdir/data/VS_segm/VS_registered/training/",
                                                cycle_gan=self.cycle_gan,
                                                input_data=["t2"], input_name=["image"],
                                                output_data="vs", output_name="vs",
                                                batch_size=self.batch_size, shuffle=True, segm_size=0,
                                                dsize=self.dsize, p_augm=0.0, alpha=0, beta=1)
            self.train_set.augm_methods = [
                A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
                A.VerticalFlip(p=0.5),
                A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                         A.MedianBlur(p=0.5, blur_limit=5),
                         A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
            ]
            self.val_set = DataSet2DSynthetic("/tf/workdir/data/VS_segm/VS_registered/validation/",
                                              cycle_gan=self.cycle_gan,
                                              input_data=["t2"], input_name=["image"],
                                              output_data="vs", output_name="vs",
                                              batch_size=self.batch_size, shuffle=True, segm_size=0, dsize=self.dsize,
                                              p_augm=0.0, alpha=0, beta=1)
        else:
            raise ValueError(f"Training {self.data_type} not valid.")

        logging.info(
            "SegmS2T: training {0}, validation {1}".format(len(self.train_set), len(self.val_set)))
