########################################################################################################################
# Segmentation from SIFA
########################################################################################################################

import tensorflow as tf
from models.ModelBase import ModelBase

__author__ = "c.magg"


class SIFASegmentation(ModelBase):
    """
    Segmentation architecture from SIFA.

    based on github repo https://github.com/cchen-cc/SIFA
    """

    def __init__(self,
                 input_shape=(32, 32, 512),
                 output_classes=1,
                 activation="relu",
                 norm="instance_norm",
                 kernel_init='truncated_normal',
                 input_name="encoder",
                 output_name="segm",
                 seed=13317):
        super(SIFASegmentation, self).__init__(input_shape=input_shape,
                                            output_classes=output_classes,
                                            activation=activation,
                                            norm=norm,
                                            kernel_init=kernel_init,
                                            input_name=input_name,
                                            output_name=output_name,
                                            seed=seed)

        self._kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)

    def _residual_block(self, x):
        """
        Residual block according to https://github.com/cchen-cc/SIFA.
        """
        dim = x.shape[-1]
        h = x
        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        x = tf.pad(x, padding, mode="reflect")
        x = self._simple_conv_block(x, dim, kernel_size=3, padding="valid", use_bias=False, dropout_rate=0.75,
                                    do_norm=True, do_act=True)

        x = tf.pad(x, padding, mode="reflect")
        x = self._simple_conv_block(x, dim, kernel_size=3, padding="valid", use_bias=False, do_act=False,
                                    dropout_rate=0.75, do_norm=True)

        return self._get_act_layer(self._activation)(x + h)

    def generate_model(self) -> tf.keras.Model:
        """
        Generate SIFA Segmentation keras model
        """
        # set seed
        self._set_random_seed()

        # input layer
        inputs = tf.keras.Input(shape=self._input_shape, name=self._input_name)

        x = self._simple_conv_block(inputs, self._output_classes, kernel_size=1, strides=1, kernel_init=self._kernel_init,
                                    do_norm=False, do_act=False, dropout_rate=0.75, padding="same")
        x = tf.nn.sigmoid(x)
        out = tf.image.resize(x, (256, 256), name=self._output_name)
        return tf.keras.Model(inputs, out, name="Segmentation")
