########################################################################################################################
# Container for Dice Coeffienc and Loss
########################################################################################################################
import tensorflow as tf

__author__ = "c.magg"


class DiceCoefficient(tf.keras.losses.Loss):
    """
    Dice Coefficient
    """

    def __init__(self, smooth=1, name="dice_coeff", **kwargs):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        return dice_coefficient(y_true, y_pred, self.smooth)


class DiceLoss(tf.keras.losses.Loss):
    """
    Dice Coefficient Loss = 1 - DC
    """

    def __init__(self, smooth=1, name="dice_loss", **kwargs):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred, self.smooth)


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersect = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_pred) + tf.keras.backend.sum(y_true)
    return (2. * intersect + smooth) / (union + smooth)
