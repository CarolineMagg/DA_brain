import time
from time import sleep
from unittest import TestCase

from data_utils.DataSetGenerator import DataSetGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A

from data_utils.data_visualization import plot_gen_separately, plot_gen_overlap
from losses.dice import DiceLoss, DiceCoefficient
from models.UNet import UNet
from models.utils import check_gpu

import logging
logging.basicConfig(level=logging.INFO)


def benchmark(dataset, num_epochs=2):
    """Taken from https://www.tensorflow.org/guide/data_performance#the_training_loop"""
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time per epoch:", (time.perf_counter() - start_time) / num_epochs,
          (time.perf_counter() - start_time))


class TestDataSetGenerator(TestCase):

    def setUp(self) -> None:
        self.path = "../../data/VS_segm/VS_registered/validation/"

    def test_benchmark_time_paired(self):
        start_time = time.perf_counter()
        for iter in range(5):
            ds_generator = DataSetGenerator("../../data/VS_segm/VS_registered/validation/",
                                            input_data="t1", output_data="vs",
                                            dsize=(256, 256), shuffle=True, p_augm=0.5, use_filter="vs")
            ds = ds_generator.generate_dataset()
            benchmark(ds)
        print("Total execution time per iter: ", (time.perf_counter() - start_time)/5)
        print("DS length ", len(list(ds.as_numpy_iterator())))

    def test_benchmark_time_unpaired(self):
        start_time = time.perf_counter()
        for iter in range(5):
            ds_generator = DataSetGenerator("../../data/VS_segm/VS_registered/validation/",
                                            input_data="t1", output_data="t2",
                                            dsize=(256, 256), shuffle=True, p_augm=0.5,
                                            use_unpaired=True)
            ds = ds_generator.generate_dataset()
            benchmark(ds)
            print("Total execution time per iter: ", (time.perf_counter() - start_time)/5)
            print("DS length ", len(list(ds.as_numpy_iterator())))

    def test_generator_dataset(self):
        ds_generator = DataSetGenerator(self.path, input_data="t1", dsize=(128, 128))
        generator = ds_generator.generator_data(["t1"])
        for idx in range(5):
            blub = next(generator)
            print(np.shape(blub["t1"]), " - ", blub["t1"].dtype)
        plt.imshow(blub["t1"])
        plt.show()

    def test_generate_dataset_only_input(self):
        print("check output shapes for only inputs")
        ds_generator = DataSetGenerator(self.path, input_data=["t1"], dsize=(128, 128), shuffle=True)
        ds = ds_generator.generate_dataset()
        print("single input")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem["t1"]))
            if idx == 4:
                break

        ds_generator = DataSetGenerator(self.path, input_data=["t1", "t2"], shuffle=True)
        ds = ds_generator.generate_dataset()
        print("double input")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem["t1"]), " and ", np.shape(elem["t2"]))
            if idx == 4:
                break

    def test_generate_dataset(self):
        print("check output shapes for inputs and outputs")
        ds_generator = DataSetGenerator(self.path, input_data=["t1"], output_data=["vs"], shuffle=False)
        ds = ds_generator.generate_dataset()
        print("single input, single output")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem[0]["t1"]), np.shape(elem[1]["vs"]))
            if idx == 4:
                break

        ds_generator = DataSetGenerator(self.path, input_data=["t1"], output_data=["vs", "cochlea"], shuffle=False)
        ds = ds_generator.generate_dataset()
        print("single input, double output")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem[0]["t1"]), np.shape(elem[1]["vs"]), np.shape(elem[1]["cochlea"]))
            if idx == 4:
                break

        ds_generator = DataSetGenerator(self.path, input_data=["t1", "t2"], output_data=["vs"], shuffle=True)
        ds = ds_generator.generate_dataset()
        print("double input, single output")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem[0]["t1"]), np.shape(elem[0]["t2"]), np.shape(elem[1]["vs"]))
            if idx == 4:
                break

        ds_generator = DataSetGenerator(self.path, input_data=["t1", "t2"], output_data=["vs", "cochlea"], shuffle=True)
        ds = ds_generator.generate_dataset()
        print("double input, double output")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem[0]["t1"]), np.shape(elem[0]["t2"]), np.shape(elem[1]["vs"]),
                  np.shape(elem[1]["cochlea"]))
            if idx == 4:
                break

    def test_generate_dataset_augm_only_image(self):
        ds_generator = DataSetGenerator(self.path, input_data=["t1", "t2"], p_augm=1.0)
        ds_generator._augm_methods = [A.Rotate(p=1.0, limit=(40, 45)),
                                      A.Resize(128, 128, interpolation=1, always_apply=False, p=1)]
        ds = ds_generator.generate_dataset()
        print("double input, double output")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem["t1"]), np.shape(elem["t2"]))
            if idx == 4:
                break
        sleep(0.05)
        plt.imshow(elem["t1"][1, :, :, 0])
        plt.show()

    def test_generate_dataset_augm_with_targets(self):
        ds_generator = DataSetGenerator(self.path, input_data=["t1", "t2"], output_data=["vs", "cochlea"], p_augm=1.0,
                                        use_filter="vs")
        ds_generator._augm_methods = [A.Rotate(p=1.0, limit=(40, 45)),
                                      A.Resize(128, 128, interpolation=1, always_apply=False, p=1)]
        ds = ds_generator.generate_dataset()
        print("double input, double output")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem[0]["t1"]), np.shape(elem[0]["t2"]),
                  np.shape(elem[1]["vs"]), np.shape(elem[1]["cochlea"]))
            if idx == 4:
                break
        sleep(0.1)
        plt.imshow(elem[0]["t1"][1, :, :, 0] + elem[1]["vs"][1, :, :, 0]*100)
        plt.show()

    def test_generate_dataset_filter(self):
        ds_generator = DataSetGenerator(self.path, input_data=["t1", "t2"], output_data=["vs", "cochlea"], use_filter="vs")
        ds = ds_generator.generate_dataset()
        print("double input, double output")
        print(ds.element_spec)
        for idx, elem in enumerate(ds):
            print(idx, " - ", np.shape(elem[0]["t1"]), np.shape(elem[0]["t2"]), np.shape(elem[1]["vs"]),
                  np.shape(elem[1]["cochlea"]))
            if idx == 4:
                break

    def test_plotting_sidebyside(self):
        print("Dataset with unpaired t1, t2; no shuffle; augm")
        ds_generator = DataSetGenerator(self.path, input_data=["t1"], output_data=["t2"], shuffle=False, batch_size=4,
                                        use_unpaired=True, p_augm=1.0)
        ds = ds_generator.generate_dataset()
        batch = next(iter(ds))
        plot_gen_separately(batch, nrows=2, ncols=2)

    def test_plot_overlap(self):
        ds_generator = DataSetGenerator(self.path, input_data=["t1"], output_data=["vs"], shuffle=True,
                                        batch_size=4, use_filter="vs")
        ds = ds_generator.generate_dataset()
        batch = list(ds.take(1).as_numpy_iterator())[0]
        plot_gen_overlap(batch, nrows=2, ncols=2)

    def test_with_model(self):
        check_gpu()
        print("single input, single output")
        unet = UNet(activation="relu", input_name="t1", output_name="vs", input_shape=(256, 256, 1),
                    filter_depth=(16, 32, 64, 128), seed=1334).generate_model()
        ds_generator = DataSetGenerator(self.path, input_data=["t1"], output_data=["vs"]).generate_dataset()

        unet.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=DiceLoss(), metrics=DiceCoefficient())
        unet.fit(ds_generator,
                 initial_epoch=0,
                 epochs=2,
                 verbose=1)

        print("double input, single output")
        unet = UNet(activation="relu", input_name=["t1", "t2"], output_name="vs", input_shape=(256, 256, 1),
                    filter_depth=(16, 32, 64, 128), seed=1334).generate_model()
        ds_generator = DataSetGenerator(self.path, input_data=["t1", "t2"], output_data=["vs"]).generate_dataset()

        unet.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=DiceLoss(), metrics=DiceCoefficient())
        unet.fit(ds_generator,
                 initial_epoch=0,
                 epochs=1,
                 verbose=1)

        print("double input, double output")
        unet = UNet(activation="relu", input_name=["t1", "t2"], output_name=["vs", "cochlea"],
                    input_shape=(256, 256, 1),
                    filter_depth=(16, 32, 64, 128), seed=1334).generate_model()
        ds_generator = DataSetGenerator(self.path, input_data=["t1", "t2"],
                                        output_data=["vs", "cochlea"]).generate_dataset()

        unet.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=DiceLoss(), metrics=DiceCoefficient())
        unet.fit(ds_generator,
                 initial_epoch=0,
                 epochs=1,
                 verbose=1)
