########################################################################################################################
# XNet model architecture
########################################################################################################################
import logging
import tensorflow as tf
from models.ModelBase import ModelBase
from models.utils import check_gpu

__author__ = "c.magg"


class XNet(ModelBase):
    """
    XNet architecture.

    based on github repo https://github.com/JosephPB/XNet/tree/56b982f8a16ff781dab1137dce7dcdd01954eefd
    and paper https://arxiv.org/abs/1812.00548
    """

    def __init__(self,
                 input_shape=(256, 256, 1),
                 output_classes=1,
                 filter_depth=(32, 64, 128, 256),
                 activation="relu",
                 norm="batch_norm",
                 kernel_init="he_normal",
                 input_name="image",
                 output_name="mask",
                 seed=13317):

        super(XNet, self).__init__(input_shape=input_shape,
                                   output_classes=output_classes,
                                   activation=activation,
                                   norm=norm,
                                   kernel_init=kernel_init,
                                   input_name=input_name,
                                   output_name=output_name,
                                   seed=seed)

        self._kernel_size = 3
        if len(filter_depth) != 4:
            raise ValueError("XNet: 'filter_depth' should have length 4, not {0}.".format(len(filter_depth)))
        self._filter_depth = filter_depth
        if activation == "selu" and kernel_init != "lecun_normal":
            logging.warning("UNet: 'kernel_init' is changed to 'lecun_normal' since 'activation' is selu.")
            self._kernel_init = "lecun_normal"

    def generate_model(self) -> tf.keras.Model:
        """
        Generate XNet model
        :return: tensorflow keras XNet model
        """
        # set random seed
        self._set_random_seed()

        # input layer(s)
        if type(self._input_name) == str:
            img_input = tf.keras.layers.Input(shape=self._input_shape, name=self._input_name)
            conv1 = self._simple_conv_block(img_input, self._filter_depth[0], kernel_size=self._kernel_size,
                                            kernel_init=self._kernel_init, padding="same")
        elif type(self._input_name) == list:
            img_input1 = tf.keras.layers.Input(shape=self._input_shape, name=self._input_name[0])
            img_input2 = tf.keras.layers.Input(shape=self._input_shape, name=self._input_name[1])
            img_input = [img_input1, img_input2]
            img_inputs = tf.keras.layers.Concatenate()(img_input)
            conv1 = self._simple_conv_block(img_inputs, self._filter_depth[0], kernel_size=self._kernel_size,
                                            kernel_init=self._kernel_init, padding="same")
        else:
            raise ValueError("XNet: 'input_names' need to be either str or list.")

        # encoder (3x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

        conv2 = self._simple_conv_block(x, self._filter_depth[1], kernel_size=self._kernel_size,
                                        kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = self._simple_conv_block(x, self._filter_depth[2], kernel_size=self._kernel_size,
                                        kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

        # bridge (2x)
        x = self._simple_conv_block(x, self._filter_depth[3], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")

        # decoder (2x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self._simple_conv_block(x, self._filter_depth[2], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.Concatenate()([conv3, x])

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self._simple_conv_block(x, self._filter_depth[1], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.Concatenate()([conv2, x])

        # encoder (2x)
        conv4 = self._simple_conv_block(x, self._filter_depth[1], kernel_size=self._kernel_size,
                                        kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = self._simple_conv_block(x, self._filter_depth[2], kernel_size=self._kernel_size,
                                        kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv5)

        # bridge (2x)
        x = self._simple_conv_block(x, self._filter_depth[3], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")

        # decoder (3x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self._simple_conv_block(x, self._filter_depth[2], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.Concatenate()([conv5, x])

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self._simple_conv_block(x, self._filter_depth[1], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.Concatenate()([conv4, x])

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = self._simple_conv_block(x, self._filter_depth[0], kernel_size=self._kernel_size,
                                    kernel_init=self._kernel_init, padding="same")
        x = tf.keras.layers.Concatenate()([conv1, x])

        # last layers
        x = tf.keras.layers.Conv2D(self._filter_depth[0], self._kernel_size, padding="same")(x)
        if type(self._output_name) == str:
            output = tf.keras.layers.Conv2D(self._output_classes, (1, 1), padding="valid", activation="sigmoid",
                                            name=self._output_name)(x)
        elif type(self._output_name) == list:
            output1 = tf.keras.layers.Conv2D(self._output_classes, (1, 1), padding="valid", activation="sigmoid",
                                             name=self._output_name[0])(x)
            output2 = tf.keras.layers.Conv2D(self._output_classes, (1, 1), padding="valid", activation="sigmoid",
                                             name=self._output_name[1])(x)
            output = [output1, output2]

        return tf.keras.Model(img_input, output, name="XNet")


if __name__ == "__main__":
    check_gpu()
    model = XNet(input_shape=(256, 256, 1)).generate_model()
    print(model.summary())
