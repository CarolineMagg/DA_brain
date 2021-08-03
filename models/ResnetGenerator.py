########################################################################################################################
# ResnetGenerator architecture
########################################################################################################################

import tensorflow as tf
from models.ModelBase import ModelBase

__author__ = "c.magg"


class ResnetGenerator(ModelBase):
    """
    ResnetGenerator architecture.

    based on github repo https://github.com/LynnHo/CycleGAN-Tensorflow-2
    """

    def __init__(self,
                 input_shape=(256, 256, 1),
                 output_classes=1,
                 n_downsampling=2,
                 n_blocks=9,
                 dim=64,
                 activation="relu",
                 norm="instance_norm",
                 kernel_init='glorot_uniform',
                 padding="reflect",
                 input_name="image",
                 output_name="generated",
                 seed=13317):

        super(ResnetGenerator, self).__init__(input_shape=input_shape,
                                              output_classes=output_classes,
                                              activation=activation,
                                              norm=norm,
                                              kernel_init=kernel_init,
                                              input_name=input_name,
                                              output_name=output_name,
                                              seed=seed)

        self._n_downsampling = n_downsampling
        self._n_blocks = n_blocks
        self._dim = dim
        self._padding = padding

    def _residual_block(self, x):
        dim = x.shape[-1]
        h = x
        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        x = tf.pad(x, padding, mode=self._padding)
        x = self._simple_conv_block(x, dim, kernel_size=3, padding="valid", use_bias=False)

        x = tf.pad(x, padding, mode=self._padding)
        x = self._simple_conv_block(x, dim, kernel_size=3, padding="valid", use_bias=False, do_act=False)

        return tf.keras.layers.add([x, h])

    def generate_model(self) -> tf.keras.Model:
        """
        Generate ResnetGenerator keras model
        """
        # set seed
        self._set_random_seed()
        dim = self._dim

        # input layer
        inputs = tf.keras.Input(shape=self._input_shape, name=self._input_name)

        # first block
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode=self._padding)
        x = self._simple_conv_block(x, dim, kernel_size=7, padding="valid", use_bias=False)

        # down sampling
        for _ in range(self._n_downsampling):
            dim = dim * 2
            x = self._simple_conv_block(x, dim, kernel_size=3, strides=2, padding="same", use_bias=False)

        # blocks
        for _ in range(self._n_blocks):
            x = self._residual_block(x)

        # up sampling
        for _ in range(self._n_downsampling):
            dim = dim / 2
            x = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding="same", use_bias=False)(x)
            x = self._normalization_layer()(x)
            x = self._get_act_layer(self._activation)(x)

        # last layer
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode=self._padding)
        x = tf.keras.layers.Conv2D(self._output_classes, 7, padding="valid", use_bias=True)(x)
        out = tf.tanh(x, name=self._output_name)

        return tf.keras.Model(inputs, out, name="ResnetGenerator")
