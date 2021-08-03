########################################################################################################################
# Base class for model generation
########################################################################################################################

import tensorflow as tf
import tensorflow_addons as tfa

__author__ = "c.magg"


class ModelBase:
    """
    ModelBase used as base class for neural networks.
    Provides general functions:
    * select activation function
    * select normalization layer

    """

    def __init__(self,
                 input_shape=(256, 256, 1),
                 output_classes=1,
                 activation="relu",
                 norm="batch_norm",
                 input_name="image",
                 output_name="output",
                 kernel_init="he_normal",
                 seed=13317):

        self._input_shape = input_shape
        self._input_name = input_name
        self._output_classes = output_classes
        self._output_name = output_name
        self._kernel_init = kernel_init  # "he_normal"
        self._seed = seed

        self._activation = activation
        self._normalization_layer = self._get_norm_layer(norm)

    @staticmethod
    def _get_norm_layer(norm):
        """
        Get normalization layer, eg. BatchNormalization or InstanceNormalization
        """
        if norm == 'none' or norm is None:
            return lambda: lambda x: x
        elif norm == 'batch_norm':
            return tf.keras.layers.BatchNormalization
        elif norm == 'instance_norm':
            return tfa.layers.InstanceNormalization
        elif norm == 'layer_norm':
            return tf.keras.layers.LayerNormalization

    @staticmethod
    def _get_act_layer(act):
        """
        Get activation function layer, eg. LeakyRelu or Relu
        """
        if act == 'none' or act is None:
            return lambda: lambda x: x
        elif act == "relu":
            return tf.keras.layers.ReLU()
        elif act == "leaky_relu":
            return tf.keras.layers.LeakyReLU(alpha=0.2)
        else:
            return tf.keras.layers.Activation(act)

    def _simple_conv_block(self, x, dim, kernel_size=3, strides=1, padding="valid", use_bias=True,
                           do_act=True, do_norm=True, kernel_init="glorot_uniform"):
        """
        Conv block with Conv2D + Normalization + Activation
        """
        x = tf.keras.layers.Conv2D(dim, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                                   kernel_initializer=kernel_init)(x)
        if do_norm:
            x = self._normalization_layer()(x)
        if do_act:
            x = self._get_act_layer(self._activation)(x)
        return x

    def _double_conv_block(self, x, dim, kernel_size=3, strides=1, padding="valid", use_bias=True,
                           do_act=True, do_norm=True, kernel_init="glorot_uniform"):
        x = tf.keras.layers.Conv2D(dim, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                                   kernel_initializer=kernel_init)(x)
        x = tf.keras.layers.Conv2D(dim, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                                   kernel_initializer=kernel_init)(x)
        if do_norm:
            x = self._normalization_layer()(x)
        if do_act:
            x = self._get_act_layer(self._activation)(x)
        return x

    def _set_random_seed(self):
        tf.random.set_seed(self._seed)

    def generate_model(self) -> tf.keras.Model:
        """
        Generate tensorflow keras model
        """
        self._set_random_seed()
        raise NotImplementedError("ModelBase: 'generate_model' not implemented for {0}".format(self.__class__.__name__))
