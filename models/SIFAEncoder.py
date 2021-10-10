########################################################################################################################
# Encoder from SIFA
########################################################################################################################

import tensorflow as tf
from models.ModelBase import ModelBase

__author__ = "c.magg"


class SIFAEncoder(ModelBase):
    """
    Encoder architecture from SIFA.

    based on github repo https://github.com/cchen-cc/SIFA
    """

    def __init__(self,
                 input_shape=(256, 256, 1),
                 output_classes=512,
                 activation="relu",
                 norm="batch_norm",
                 kernel_init='truncated_normal',
                 input_name="image",
                 output_name=None,
                 full_mode=True,
                 double_output=True,
                 seed=13317):

        if output_name is None:
            output_name = ["encoder1", "encoder2"]
        assert len(output_name) == 2, "SIFAEncoder needs 2 output names."

        super(SIFAEncoder, self).__init__(input_shape=input_shape,
                                          output_classes=output_classes,
                                          activation=activation,
                                          norm=norm,
                                          kernel_init=kernel_init,
                                          input_name=input_name,
                                          output_name=output_name,
                                          seed=seed)

        self._kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        self._full_model = full_mode
        self._double_output = double_output

    def _residual_block(self, x, dim, output_padding=False, name=None):
        """
        Residual block according to https://github.com/cchen-cc/SIFA.
        """
        dim_in = x.shape[-1]
        h = x
        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        x = tf.pad(x, padding, mode="constant")
        x = self._simple_conv_block(x, dim, kernel_size=3, padding="valid", use_bias=False, dropout_rate=0.75,
                                    do_norm=True, do_act=True)

        x = tf.pad(x, padding, mode="constant")
        x = self._simple_conv_block(x, dim, kernel_size=3, padding="valid", use_bias=False, do_act=False,
                                    dropout_rate=0.75, do_norm=True)

        if output_padding:
            h = tf.pad(h, [[0, 0], [0, 0], [0, 0], [(dim - dim_in) // 2, (dim - dim_in) // 2]], mode="reflect")

        if name is None:
            return self._get_act_layer(self._activation)(x + h)
        else:
            return self._get_act_layer_named(self._activation, name)(x + h)

    def _dilated_residual_block(self, x, dim):
        """
        Residual block with dilated conv layers according to https://github.com/cchen-cc/SIFA.
        """
        h = x
        padding = [[0, 0], [2, 2], [2, 2], [0, 0]]
        x = tf.pad(x, padding, mode="constant")
        x = self._dilated_conv_block(x, dim, kernel_size=3, dilation_rate=2,
                                     padding="valid", dropout_rate=0.75, do_act=True, do_norm=True)

        x = tf.pad(x, padding, mode="constant")
        x = self._dilated_conv_block(x, dim, kernel_size=3, dilation_rate=2,
                                     padding="valid", dropout_rate=0.75, do_act=True, do_norm=True)

        return self._get_act_layer(self._activation)(x + h)

    def generate_model(self) -> tf.keras.Model:
        """
        Generate SIFA Encoder keras model
        """
        # set seed
        self._set_random_seed()

        # input layer
        inputs = tf.keras.Input(shape=self._input_shape, name=self._input_name)

        # C16 + R16 + M
        x = self._simple_conv_block(inputs, 16, kernel_size=7, strides=1, padding="same", do_norm=True,
                                    kernel_init=self._kernel_init)
        x = self._residual_block(x, 16)
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(x)

        # R32 + M
        x = self._residual_block(x, 32, output_padding=True)
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(x)

        # 2xR64 + M
        x = self._residual_block(x, 64, output_padding=True)
        x = self._residual_block(x, 64)
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(x)

        # 2xR128 + 4xR256 + 2xR512
        x = self._residual_block(x, 128, output_padding=True)
        x = self._residual_block(x, 128)

        x = self._residual_block(x, 256, output_padding=True)
        x = self._residual_block(x, 256)
        x = self._residual_block(x, 256)
        x = self._residual_block(x, 256)

        x = self._residual_block(x, 512, output_padding=True)
        out1 = self._residual_block(x, 512, name=self._output_name[0])

        # 2xD512
        x = self._dilated_residual_block(out1, 512)
        x = self._dilated_residual_block(x, 512)

        # output layers (2xC512)
        x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding="same", kernel_initializer=self._kernel_init)(
            x)
        x = self._normalization_layer()(x)
        x = self._get_act_layer(self._activation)(x)

        x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding="same", kernel_initializer=self._kernel_init)(
            x)
        x = self._normalization_layer()(x)
        out2 = self._get_act_layer_named(self._activation, name=self._output_name[1])(x)

        if self._double_output:
            return tf.keras.Model(inputs, [out2, out1], name="Encoder")
        else:
            return tf.keras.Model(inputs, out2, name="Encoder")

    def generate_model_small(self) -> tf.keras.Model:
        """
        Generate SIFA Encoder keras model
        """
        # set seed
        self._set_random_seed()

        # input layer
        inputs = tf.keras.Input(shape=self._input_shape, name=self._input_name)

        # C16 + R16 + M
        x = self._simple_conv_block(inputs, 16, kernel_size=7, strides=1, padding="same", do_norm=True,
                                    kernel_init=self._kernel_init)
        x = self._residual_block(x, 16)
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(x)

        # R32 + M
        x = self._residual_block(x, 32, output_padding=True)
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(x)

        # 2xR64 + M
        x = self._residual_block(x, 64, output_padding=True)
        x = self._residual_block(x, 64)
        x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(x)

        # 4xR128 + 2xR256
        x = self._residual_block(x, 128, output_padding=True)
        x = self._residual_block(x, 128)

        x = self._residual_block(x, 128, output_padding=True)
        x = self._residual_block(x, 128)

        x = self._residual_block(x, 256, output_padding=True)
        out1 = self._residual_block(x, 256, name=self._output_name[0])

        # 2xD256
        x = self._dilated_residual_block(out1, 256)
        x = self._dilated_residual_block(x, 256)

        # output layers (2xC256)
        x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer=self._kernel_init)(
            x)
        x = self._normalization_layer()(x)
        x = self._get_act_layer(self._activation)(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer=self._kernel_init)(
            x)
        x = self._normalization_layer()(x)
        out2 = self._get_act_layer_named(self._activation, name=self._output_name[1])(x)

        if self._double_output:
            return tf.keras.Model(inputs, [out2, out1], name="Encoder")
        else:
            return tf.keras.Model(inputs, out2, name="Encoder")
