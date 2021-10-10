########################################################################################################################
# Decoder from SIFA
########################################################################################################################

import tensorflow as tf
from models.ModelBase import ModelBase

__author__ = "c.magg"


class SIFADecoder(ModelBase):
    """
    Decoder architecture from SIFA.

    based on github repo https://github.com/cchen-cc/SIFA
    """

    def __init__(self,
                 input_shape=(32, 32, 512),
                 input_shape2=(256, 256, 1),
                 output_classes=1,
                 activation="relu",
                 norm="instance_norm",
                 kernel_init='truncated_normal',
                 input_name="encoder",
                 output_name="decoder",
                 skip=False,
                 seed=13317):

        super(SIFADecoder, self).__init__(input_shape=input_shape,
                                          output_classes=output_classes,
                                          activation=activation,
                                          norm=norm,
                                          kernel_init=kernel_init,
                                          input_name=input_name,
                                          output_name=output_name,
                                          seed=seed)

        self._kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self._skip = skip
        self._input_shape2 = input_shape2

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
        Generate SIFA Decoder keras model
        """
        # set seed
        self._set_random_seed()

        # input layer
        inputs = tf.keras.Input(shape=self._input_shape, name=self._input_name)

        # conv layer
        x = self._simple_conv_block(inputs, 128, kernel_size=3, strides=1, kernel_init=self._kernel_init,
                                    padding="same", do_norm=True)

        # 4 x res net block
        for idx in range(4):
            x = self._residual_block(x)

        # 3 deconv layers
        dims = [64, 64, 32]
        for idx in range(3):
            x = self._simple_deconv_block(x, dims[idx], kernel_size=7, strides=2, padding="same",
                                          kernel_init=self._kernel_init, do_norm=True, do_act=True)

        # output layer
        x = tf.keras.layers.Conv2D(self._output_classes, 7, strides=1, padding="same",
                                   kernel_initializer=self._kernel_init)(x)

        if self._skip is True:
            inputs2 = tf.keras.Input(shape=self._input_shape2, name="tmp")
            out = tf.tanh(inputs2 + x, name=self._output_name)
            return tf.keras.Model([inputs, inputs2], out, name="Decoder")
        else:
            out = tf.tanh(x, name=self._output_name)
            return tf.keras.Model(inputs, out, name="Decoder")

    def generate_model_small(self) -> tf.keras.Model:
        """
        Generate SIFA Decoder small keras model
        """
        # set seed
        self._set_random_seed()

        # input layer
        inputs = tf.keras.Input(shape=self._input_shape, name=self._input_name)

        # conv layer
        x = self._simple_conv_block(inputs, 128, kernel_size=3, strides=1, kernel_init=self._kernel_init,
                                    padding="same", do_norm=True)

        # 3 x res net block
        for idx in range(3):
            x = self._residual_block(x)

        # 3 deconv layers
        dims = [64, 64, 32]
        for idx in range(3):
            x = self._simple_deconv_block(x, dims[idx], kernel_size=7, strides=2, padding="same",
                                          kernel_init=self._kernel_init, do_norm=True, do_act=True)

        # output layer
        x = tf.keras.layers.Conv2D(self._output_classes, 7, strides=1, padding="same",
                                   kernel_initializer=self._kernel_init)(x)

        if self._skip is True:
            inputs2 = tf.keras.Input(shape=self._input_shape2, name="tmp")
            out = tf.tanh(inputs2 + x, name=self._output_name)
            return tf.keras.Model([inputs, inputs2], out, name="Decoder")
        else:
            out = tf.tanh(x, name=self._output_name)
            return tf.keras.Model(inputs, out, name="Decoder")