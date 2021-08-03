from unittest import TestCase
import tensorflow as tf

from models.ConvDiscriminator import ConvDiscriminator
from models.ResnetGenerator import ResnetGenerator
from models.UNet import UNet
from models.XNet import XNet
from models.utils import check_gpu


class TestXNet(TestCase):

    def setUp(self) -> None:
        check_gpu()

    def test_generate_model(self):
        model = XNet(activation="leaky_relu", input_shape=(256, 256, 1)).generate_model()
        print(model.summary())
        # tf.keras.utils.plot_model(model)

    def test_generate_model_multiple_input_multiple_output(self):
        model = XNet(activation="leaky_relu", input_shape=(256, 256, 1), input_name=["t1", "t2"],
                     output_name=["mask", "mask1"]).generate_model()
        print(model.summary())

    def test_run(self):
        model = XNet().generate_model()
        res = model(tf.zeros(shape=(1, 256, 256, 1)))
        self.assertEqual((1, 256, 256, 1), res.shape)


class TestUNet(TestCase):

    def setUp(self) -> None:
        check_gpu()

    def test_generate_model(self):
        model = UNet(activation="selu", input_shape=(256, 256, 1)).generate_model()
        print(model.summary())
        # tf.keras.utils.plot_model(model)

    def test_generate_model_multiple_input_multiple_output(self):
        model = UNet(activation="selu", input_shape=(256, 256, 1), input_name=["t1", "t2"],
                     output_name=["mask", "mask1"]).generate_model()
        print(model.summary())
        # tf.keras.utils.plot_model(model)

    def test_run(self):
        model = UNet().generate_model()
        res = model(tf.zeros(shape=(1, 256, 256, 1)))
        self.assertEqual((1, 256, 256, 1), res.shape)


class TestResnetGenerator(TestCase):

    def setUp(self) -> None:
        check_gpu()

    def test_generate_model(self):
        model = ResnetGenerator(n_blocks=2).generate_model()
        print(model.summary())
        # tf.keras.utils.plot_model(model)

    def test_run(self):
        model = ResnetGenerator(n_blocks=2).generate_model()
        res = model(tf.zeros(shape=(1, 256, 256, 1)))
        self.assertEqual((1, 256, 256, 1), res.shape)


class TestConvDiscriminator(TestCase):

    def setUp(self) -> None:
        check_gpu()

    def test_generate_model(self):
        model = ConvDiscriminator().generate_model()
        print(model.summary())
        # tf.keras.utils.plot_model(model)

    def test_run(self):
        model = ConvDiscriminator().generate_model()
        res = model(tf.zeros(shape=(1, 256, 256, 1)))
        self.assertEqual((1, 32, 32, 1), res.shape)
