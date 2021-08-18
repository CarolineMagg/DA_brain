########################################################################################################################
# classification-guided UNet model architecture
########################################################################################################################
import logging

import tensorflow as tf
import tensorflow_addons as tfa
from models.ModelBase import ModelBase

__author__ = "c.magg"


class UNet_ClassGuided(ModelBase):
    """
    classification-guided UNet architecture.

    based on github repo https://github.com/Crispy13/Unet3plus_tensorflow2
    and paper https://arxiv.org/abs/2004.08790.
    """

    def __init__(self,
                 input_shape=(256, 256, 1),
                 output_classes=1,
                 filter_depth=(32, 64, 128, 256),
                 activation="relu",
                 norm="batch_norm",
                 kernel_init="he_normal",
                 input_name="image",
                 output_name=["mask", "classification"],
                 seed=13317):

        super(UNet_ClassGuided, self).__init__(input_shape=input_shape,
                                               output_classes=output_classes,
                                               activation=activation,
                                               norm=norm,
                                               kernel_init=kernel_init,
                                               input_name=input_name,
                                               output_name=output_name,
                                               seed=seed)

        self._kernel_size = 3
        if len(filter_depth) != 4:
            raise ValueError("UNet: 'filter_depth' should have length 4, not {0}.".format(len(filter_depth)))
        self._filter_depth = filter_depth
        if activation == "selu" and kernel_init != "lecun_normal":
            logging.warning("UNet: 'kernel_init' is changed to 'lecun_normal' since 'activation' is selu.")
            self._kernel_init = "lecun_normal"

    def generate_model(self) -> tf.keras.Model:
        """
        Generate UNet keras model
        """
        # set seed
        self._set_random_seed()

        # input layer(s)
        if type(self._input_name) == str:
            img_input = tf.keras.layers.Input(shape=self._input_shape, name=self._input_name)
            conv1 = self._double_conv_block(img_input, self._filter_depth[0], kernel_size=self._kernel_size,
                                            kernel_init=self._kernel_init, padding="same")
        elif type(self._input_name) == list:
            img_input1 = tf.keras.layers.Input(shape=self._input_shape, name=self._input_name[0])
            img_input2 = tf.keras.layers.Input(shape=self._input_shape, name=self._input_name[1])
            img_input = [img_input1, img_input2]
            img_inputs = tf.keras.layers.Concatenate()(img_input)
            conv1 = self._double_conv_block(img_inputs, self._filter_depth[0], kernel_size=self._kernel_size,
                                            kernel_init=self._kernel_init, padding="same")
        else:
            raise ValueError("UNet: 'input_names' need to be either str or list.")

        # encoder (4x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

        conv2 = self._double_conv_block(x, self._filter_depth[1], kernel_size=self._kernel_size,
                                        kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = self._double_conv_block(x, self._filter_depth[2], kernel_size=self._kernel_size,
                                        kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

        # bridge
        # conv4 = self._double_conv_block(x, self._filter_depth[3], kernel_size=self._kernel_size,
        #                                kernel_init=self._kernel_init, padding="same")

        # cgm
        cgm = tf.keras.layers.SpatialDropout2D(rate=0.5)(x)
        cgm = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same",
                                     kernel_initializer=self._kernel_init)(cgm)
        cgm = tfa.layers.AdaptiveMaxPooling2D(output_size=1)(cgm)
        # cgm = tf.keras.layers.MaxPool2D(pool_size=(cgm.shape[1]-1, cgm.shape[2]-1))(cgm)  # AdaptiveMaxPooling2D(output_size=1)
        cgm = tf.keras.layers.Activation("sigmoid")(cgm)

        # decoder (4x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self._double_conv_block(x, self._filter_depth[3], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.Concatenate()([conv3, x])

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self._double_conv_block(x, self._filter_depth[2], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.Concatenate()([conv2, x])

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self._double_conv_block(x, self._filter_depth[1], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.Concatenate()([conv1, x])

        # last layers
        x = self._double_conv_block(x, self._filter_depth[0], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same", do_norm=False)
        # CGM
        cgm_channel = tf.where(cgm > 0.5, 1., 0.)
        cgm_out = tf.squeeze(cgm, axis=[-3, -2, -1], name=self._output_name[-1])

        if len(self._output_name) == 2:
            out = tf.keras.layers.Conv2D(self._output_classes, (1, 1), padding="valid", activation="sigmoid")(x)
            output = tf.multiply(out, cgm_channel, name=self._output_name[0])
            output = [output, cgm_out]
        else:
            out1 = tf.keras.layers.Conv2D(self._output_classes, (1, 1), padding="valid", activation="sigmoid",
                                          name=self._output_name[0])(x)
            output1 = tf.multiply(out1, cgm_channel, name=self._output_name[0])
            out2 = tf.keras.layers.Conv2D(self._output_classes, (1, 1), padding="valid", activation="sigmoid",
                                          name=self._output_name[1])(x)
            output2 = tf.multiply(out2, cgm_channel, name=self._output_name[1])
            output = [output1, output2, cgm_out]

        return tf.keras.Model(img_input, output, name="UNet")