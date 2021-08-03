########################################################################################################################
# Container for IoU and Loss
########################################################################################################################

import tensorflow as tf

__author__ = "c.magg"


class IoU(tf.keras.losses.Loss):
    """
    Intersection over Union
    """

    def __init__(self, name="iou"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return iou(y_true, y_pred, self.smooth)


class IoULoss(tf.keras.losses.Loss):
    """
    Intersection over Union Loss = 1 - IoU
    """

    def __init__(self, name="iou_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return 1 - iou(y_true, y_pred, self.smooth)


def iou(y_true, y_pred, smooth=1):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersect = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersect
    return (intersect + smooth) / (union + smooth)