########################################################################################################################
# ConvDiscriminator architecture - PatchGen
########################################################################################################################

import tensorflow as tf
from models.ModelBase import ModelBase
from models.utils import check_gpu

__author__ = "c.magg"


class ConvDiscriminator(ModelBase):
    """
    ConvDiscriminator architecture.

    based on github repo https://github.com/LynnHo/CycleGAN-Tensorflow-2
    """

    def __init__(self,
                 input_shape=(256, 256, 1),
                 output_classes=1,
                 n_downsampling=3,
                 dim=64,
                 activation="leaky_relu",
                 norm="instance_norm",
                 kernel_init='glorot_uniform',
                 input_name="image",
                 output_name="generated",
                 seed=13317):
        super(ConvDiscriminator, self).__init__(input_shape=input_shape,
                                                output_classes=output_classes,
                                                activation=activation,
                                                norm=norm,
                                                kernel_init=kernel_init,
                                                input_name=input_name,
                                                output_name=output_name,
                                                seed=seed)

        self._n_downsampling = n_downsampling
        self._dim = dim
        assert self._output_classes == 1

    def generate_model(self) -> tf.keras.Model:
        """
        Generate ConvDiscriminator keras model
        """
        # set seed
        self._set_random_seed()
        dim = self._dim
        dim_ = self._dim

        # input
        inputs = tf.keras.layers.Input(shape=self._input_shape, name=self._input_name)

        # first block
        x = self._simple_conv_block(inputs, dim, kernel_size=4, strides=2, padding="same",
                                    do_norm=False)

        # down sampling
        for _ in range(self._n_downsampling - 1):
            dim = min(dim * 2, dim_ * 8)
            x = self._simple_conv_block(x, dim, kernel_size=4, strides=2, padding="same", use_bias=False)

        dim = min(dim * 2, dim_ * 8)
        x = self._simple_conv_block(x, dim, kernel_size=4, strides=1, padding="same", use_bias=False)

        # last layer
        out = tf.keras.layers.Conv2D(self._output_classes, 4, strides=1, padding="same", name=self._output_name)(x)

        return tf.keras.Model(inputs, out, name="ConvDiscriminator")

    def generate_model_aux(self) -> tf.keras.Model:
        """
        Generate ConvDiscriminator keras model
        """
        # set seed
        self._set_random_seed()
        dim = self._dim
        dim_ = self._dim

        # input
        inputs = tf.keras.layers.Input(shape=self._input_shape, name=self._input_name)

        # first block
        x = self._simple_conv_block(inputs, dim, kernel_size=4, strides=2, padding="same",
                                    do_norm=False)

        # down sampling
        for _ in range(self._n_downsampling - 1):
            dim = min(dim * 2, dim_ * 8)
            x = self._simple_conv_block(x, dim, kernel_size=4, strides=2, padding="same", use_bias=False)

        dim = min(dim * 2, dim_ * 8)
        x = self._simple_conv_block(x, dim, kernel_size=4, strides=1, padding="same", use_bias=False)

        # last layer
        x = tf.keras.layers.Conv2D(2, 4, strides=1, padding="same", name=self._output_name)(x)
        out = tf.expand_dims(x[..., 0], axis=3), tf.expand_dims(x[..., 1], axis=3)

        return tf.keras.Model(inputs, out, name="ConvDiscriminator")


if __name__ == "__main__":
    check_gpu()
    model = ConvDiscriminator(input_shape=(256, 256, 1)).generate_model()
    print(model.summary(line_length=150))

    model = ConvDiscriminator(input_shape=(256, 256, 1), dim=32).generate_model()
    print(model.summary(line_length=150))
