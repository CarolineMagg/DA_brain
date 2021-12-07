########################################################################################################################
# Encoder+Decoder+Segmentation from SIFA
########################################################################################################################

import tensorflow as tf
import tensorflow_addons as tfa
from models.ModelBase import ModelBase
from models.utils import check_gpu

__author__ = "c.magg"


class CGSIFAGenerator(ModelBase):
    """
    Encoder+decoder+segmentation architecture for SIFA approach

    based on github repo https://github.com/cchen-cc/SIFA
    """

    def __init__(self,
                 input_shape=(256, 256, 1),
                 output_classes=1,
                 activation="relu",
                 norm="batch_norm",
                 kernel_init='truncated_normal',
                 input_name=None,
                 output_name=None,
                 double_output=True,
                 seed=13317):

        if input_name is None:
            input_name = ["image", "feature_space", "feature_space"]
        if output_name is None:
            output_name = ["feature_space", "generated", "vs", "vs_class"]
        assert len(input_name) == 3 or len(input_name) == 4, "SIFAGenerator needs 3 or 4 input names."
        assert len(output_name) == 3 or len(output_name) == 4, "SIFAGenerator needs 3 or 4 output names."

        super(CGSIFAGenerator, self).__init__(input_shape=input_shape,
                                              output_classes=output_classes,
                                              activation=activation,
                                              norm=norm,
                                              kernel_init=kernel_init,
                                              input_name=input_name,
                                              output_name=output_name,
                                              seed=seed)

        self._kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        self._double_output = double_output
        self._feature_shape = None
        self._feature_shape2 = None

    def _residual_block(self, x, dim=None, output_padding=False, name=None):
        """
        Residual block according to https://github.com/cchen-cc/SIFA.
        """
        dim_in = x.shape[-1]
        if dim is None:
            dim = x.shape[-1]
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

    def generate_encoder(self) -> tf.keras.Model:
        """
        Generate SIFA Encoder keras model
        """
        # set seed
        self._set_random_seed()

        # input layer
        inputs = tf.keras.Input(shape=self._input_shape, name=self._input_name[0])

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
        out1 = self._residual_block(x, 512, name="skip")

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
        out2 = self._get_act_layer_named(self._activation, name=self._output_name[0])(x)
        self._feature_shape = out2.shape[-3:]
        if self._double_output:
            self._feature_shape2 = out1.shape[-3:]
            return tf.keras.Model(inputs, [out2, out1], name="Encoder")
        else:
            return tf.keras.Model(inputs, out2, name="Encoder")

    def generate_encoder_small(self) -> tf.keras.Model:
        """
        Generate small SIFA Encoder keras model
        """
        # set seed
        self._set_random_seed()

        # input layer
        inputs = tf.keras.Input(shape=self._input_shape, name=self._input_name[0])

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
        out1 = self._residual_block(x, 256, name="skip")

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
        out2 = self._get_act_layer_named(self._activation, name=self._output_name[0])(x)
        self._feature_shape = out2.shape[-3:]
        if self._double_output:
            self._feature_shape2 = out1.shape[-3:]
            return tf.keras.Model(inputs, [out2, out1], name="Encoder")
        else:
            return tf.keras.Model(inputs, out2, name="Encoder")

    def generate_decoder(self) -> tf.keras.Model:
        """
        Generate SIFA Decoder keras model
        """
        # input layer
        inputs = tf.keras.Input(shape=self._feature_shape, name=self._input_name[1])

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

        if self._double_output is True:
            inputs2 = tf.keras.Input(shape=self._feature_shape2, name="skip")
            out = tf.tanh(inputs2 + x, name=self._output_name)
            return tf.keras.Model([inputs, inputs2], out, name="Decoder")
        else:
            out = tf.tanh(x, name=self._output_name)
            return tf.keras.Model(inputs, out, name="Decoder")

    def generate_decoder_small(self) -> tf.keras.Model:
        """
        Generate small SIFA Decoder keras model
        """
        # input layer
        inputs = tf.keras.Input(shape=self._feature_shape, name=self._input_name[1])

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

        if self._double_output is True:
            inputs2 = tf.keras.Input(shape=self._feature_shape2, name="skip")
            out = tf.tanh(inputs2 + x, name=self._output_name)
            return tf.keras.Model([inputs, inputs2], out, name="Decoder")
        else:
            out = tf.tanh(x, name=self._output_name)
            return tf.keras.Model(inputs, out, name="Decoder")

    def generate_segmentation(self):
        """
        Generate SIFA segmentation
        """

        # input layer
        inputs = tf.keras.Input(shape=self._feature_shape, name=self._input_name[2])
        # conv layer
        x = self._simple_conv_block(inputs, 128, kernel_size=3, strides=1, kernel_init=self._kernel_init,
                                    padding="same", do_norm=True)

        # 3 deconv layers
        dims = [64, 64, 32]
        for idx in range(3):
            x = self._simple_deconv_block(x, dims[idx], kernel_size=7, strides=2, padding="same",
                                          kernel_init=self._kernel_init, do_norm=True, do_act=True)

        # output layer
        out = tf.keras.layers.Conv2D(self._output_classes, 7, strides=1, padding="same",
                                     kernel_initializer=self._kernel_init, activation="sigmoid",
                                     name=self._output_name[2])(x)

        return tf.keras.Model(inputs, out, name="Segmentation")

    def generate_cg_segmentation(self):
        """
        Generate classification-guided SIFA segmentation
        """

        # input layer
        inputs = tf.keras.Input(shape=self._feature_shape, name=self._input_name[2])
        # conv layer
        x = self._simple_conv_block(inputs, 128, kernel_size=3, strides=1, kernel_init=self._kernel_init,
                                    padding="same", do_norm=True)

        # 3 deconv layers
        dims = [64, 64, 32]
        for idx in range(3):
            x = self._simple_deconv_block(x, dims[idx], kernel_size=7, strides=2, padding="same",
                                          kernel_init=self._kernel_init, do_norm=True, do_act=True)

        # cgm
        cgm = tf.keras.layers.Dropout(rate=0.5)(inputs)
        cgm = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                                     kernel_initializer=self._kernel_init)(cgm)
        cgm = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same",
                                     kernel_initializer=self._kernel_init)(cgm)
        #  cgm = tfa.layers.AdaptiveMaxPooling2D(output_size=1)(cgm)
        cgm = tf.keras.layers.GlobalMaxPool2D()(cgm)
        cgm = tf.keras.layers.Activation("sigmoid", name=self._output_name[3])(cgm)
        # cgm_channel = tf.where(cgm >= 0.5, 1., 0.)
        # cgm_channel = tf.expand_dims(tf.expand_dims(cgm_channel, axis=1), axis=1)

        # output layer
        out = tf.keras.layers.Conv2D(self._output_classes, 7, strides=1, padding="same",
                                     kernel_initializer=self._kernel_init, activation="sigmoid")(x)
        # output = tf.keras.layers.Multiply(name=self._output_name[2])([out, cgm_channel])

        return tf.keras.Model(inputs, [out, cgm], name="Segmentation")

    def generate_cgm(self):
        """
        Generate Classification-guided Module
        """
        # input layer
        inputs = tf.keras.Input(shape=self._feature_shape, name=self._input_name[2])

        # cgm
        cgm = tf.keras.layers.Dropout(rate=0.5)(inputs)
        cgm = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                                     kernel_initializer=self._kernel_init)(cgm)
        cgm = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same",
                                     kernel_initializer=self._kernel_init)(cgm)
        #  cgm = tfa.layers.AdaptiveMaxPooling2D(output_size=1)(cgm)
        cgm = tf.keras.layers.GlobalMaxPool2D()(cgm)
        cgm = tf.keras.layers.Activation("sigmoid", name=self._output_name[3])(cgm)
        # post-processing
        #  cgm_channel = tf.where(cgm >= 0.5, 1., 0.)
        #  cgm_channel = tf.expand_dims(tf.expand_dims(cgm_channel, axis=1), axis=1)

        return tf.keras.Model(inputs, cgm, name="CG Segmentation")


if __name__ == "__main__":
    check_gpu()
    factory = CGSIFAGenerator(input_shape=(256, 256, 1), double_output=False)
    encoder = factory.generate_encoder_small()
    segmentation = factory.generate_segmentation()
    decoder = factory.generate_decoder_small()
    print(encoder.summary(line_length=150))
    print(segmentation.summary(line_length=150))
    print(decoder.summary(line_length=150))
