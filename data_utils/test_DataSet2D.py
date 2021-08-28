import time
import unittest
from unittest import TestCase
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from data_utils.DataSet2D import DataSet2D
from data_utils.DataSet2DMixed import DataSet2DMixed

import logging

logging.basicConfig(level=logging.INFO)


def benchmark(dataset, num_epochs=2):
    """Taken from https://www.tensorflow.org/guide/data_performance#the_training_loop"""
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for idx in range(len(dataset)):
            sample = dataset[idx]
            # Performing a training step
            time.sleep(0.01)
    print("Execution time per epoch:", (time.perf_counter() - start_time) / num_epochs,
          (time.perf_counter() - start_time))


class TestDataSet2D(TestCase):

    def test_init(self):
        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                              input_name=["image"], batch_size=1, shuffle=True)
        self.assertEqual(1, train_set.batch_size)
        self.assertEqual(0.0, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(True, train_set.shuffle)
        self.assertEqual(10425, len(train_set))
        self.assertListEqual(["image"], train_set._input_name)
        self.assertListEqual(["t1"], train_set._input_data)

        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data="t1",
                              input_name="image", batch_size=4, shuffle=False, p_augm=0.5)
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(0.5, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(False, train_set.shuffle)
        self.assertEqual(2606, len(train_set))
        self.assertEqual(10425, train_set._number_index)
        self.assertListEqual(["image"], train_set._input_name)
        self.assertListEqual(["t1"], train_set._input_data)

        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1", "t2"],
                              input_name=["image", "image_aux"], batch_size=4, shuffle=True, p_augm=0.0)
        self.assertListEqual(["image", "image_aux"], train_set._input_name)
        self.assertListEqual(["t1", "t2"], train_set._input_data)

    def test_check_for_empty_slices(self):
        np.random.seed(5)
        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                              input_name=["image"], batch_size=1, shuffle=True, use_filter="vs")
        train_set.batch_size = 10450 // 50
        for idx in range(len(train_set)):
            res = train_set[idx]
            blub = [idx for idx, r in enumerate(res["image"]) if np.sum(r)==0]
            if len(blub) != 0:
                print(f"idx {blub} is empty.")

        np.random.seed(5)
        train_set = DataSet2D("../../data/VS_segm/VS_registered/validation/", input_data=["t1"],
                              input_name=["image"], batch_size=1, shuffle=True, use_filter="vs")
        train_set.batch_size = 2960 // 16
        for idx in range(len(train_set)):
            res = train_set[idx]
            blub = [idx for idx, r in enumerate(res["image"]) if np.sum(r) == 0]
            if len(blub) != 0:
                print(f"idx {blub} is empty.")

        np.random.seed(5)
        train_set = DataSet2D("../../data/VS_segm/VS_registered/test/", input_data=["t1"],
                              input_name=["image"], batch_size=1, shuffle=True, use_filter="vs")
        train_set.batch_size = 3960 // 30
        for idx in range(len(train_set)):
            res = train_set[idx]
            blub = [idx for idx, r in enumerate(res["image"]) if np.sum(r) == 0]
            if len(blub) != 0:
                print(f"idx {blub} is empty.")

    def test_setter(self):
        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                              input_name=["image"], batch_size=1, shuffle=True, p_augm=0.0)
        # batch_size
        train_set.batch_size = 4
        res = train_set[0]["image"]
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(4, len(res))
        # dsize
        train_set.dsize = (200, 200)
        res = train_set[0]["image"]
        self.assertEqual((200, 200), train_set.dsize)
        self.assertEqual((200, 200), res.shape[1:])
        # p augm
        train_set.p_augm = 0.5
        self.assertEqual(0.5, train_set.p_augm)
        # shuffle
        train_set.shuffle = False
        self.assertEqual(False, train_set.shuffle)
        # augm methods
        train_set.augm_methods = []
        self.assertEqual([], train_set.augm_methods)

    def test_reduce_to_nonzero_segm(self):
        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                              input_name=["image"], batch_size=4, shuffle=True, p_augm=0.0)
        train_set.reduce_to_nonzero_segm("vs")
        self.assertEqual(377, len(train_set))

    def test_balance_data(self):
        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                              input_name=["image"], batch_size=4, shuffle=True, p_augm=0.0)
        train_set.balance_dataset()
        self.assertEqual(831, len(train_set))

    def test_single_result(self):
        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                              input_name=["image"], batch_size=4, shuffle=True, p_augm=0.0)
        result = train_set[1]
        self.assertEqual(dict, type(result))
        self.assertEqual(["image"], list(result.keys()))
        self.assertEqual((4, 256, 256), np.shape(result["image"]))

    def test_double_result(self):
        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1", "t2"],
                              input_name=["image", "image_aux"], batch_size=4, shuffle=True, p_augm=0.0)
        result = train_set[1]
        self.assertEqual(dict, type(result))
        self.assertEqual(["image", "image_aux"], list(result.keys()))
        self.assertEqual((4, 256, 256), np.shape(result["image"]))
        self.assertEqual((4, 256, 256), np.shape(result["image_aux"]))

    @unittest.skip
    def test_mockup_plot(self):
        train_set = DataSet2D("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                              input_name=["image"], batch_size=4, shuffle=True, p_augm=0.5)
        with self.assertRaises(ValueError):
            train_set.plot_random_images(4, 4)
        train_set.plot_random_images(2, 2)


class TestDataSet2DMixed(TestCase):

    def test_init(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                                   input_name=["image"], output_data=["t2", "vs", "vs_class"],
                                   output_name=["image_t2", "mask", "class1"],
                                   batch_size=1, shuffle=True)
        self.assertEqual(1, train_set.batch_size)
        self.assertEqual(0.0, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(True, train_set.shuffle)
        self.assertEqual(10425, len(train_set))
        self.assertListEqual(["image"], train_set._input_name)
        self.assertListEqual(["image_t2", "mask", "class1"], train_set._output_name)
        self.assertListEqual(["t1"], train_set._input_data)
        self.assertListEqual(["t2", "vs", "vs_class"], train_set._output_data)

        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "cochlea"], output_name=["t2_", "cochlea_"],
                                   batch_size=4, shuffle=False, p_augm=0.5)
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(0.5, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(False, train_set.shuffle)
        self.assertEqual(2606, len(train_set))
        self.assertEqual(10425, train_set._number_index)
        self.assertListEqual(["image"], train_set._input_name)
        self.assertListEqual(["t2_", "cochlea_"], train_set._output_name)
        self.assertListEqual(["t1"], train_set._input_data)
        self.assertListEqual(["t2", "cochlea"], train_set._output_data)

    def test_init_only_segm(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["vs", "cochlea", "vs_class"],
                                   output_name=["vs", "cochlea", "vs_class"],
                                   batch_size=4, shuffle=False, p_augm=0.5)
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(0.5, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(False, train_set.shuffle)
        self.assertEqual(2606, len(train_set))
        self.assertEqual(10425, train_set._number_index)
        self.assertListEqual(["image"], train_set._input_name)
        self.assertListEqual(["vs", "cochlea", "vs_class"], train_set._output_name)
        self.assertListEqual(["t1"], train_set._input_data)
        self.assertListEqual(["vs", "cochlea", "vs_class"], train_set._output_data)
        #train_set.plot_random_images(2, 2)

    def test_init_only_images(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2"],
                                   output_name=["t2_"],
                                   batch_size=4, shuffle=False, p_augm=0.5)
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(0.5, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(False, train_set.shuffle)
        self.assertEqual(2606, len(train_set))
        self.assertEqual(10425, train_set._number_index)
        self.assertListEqual(["image"], train_set._input_name)
        self.assertListEqual(["t2_"], train_set._output_name)
        self.assertListEqual(["t1"], train_set._input_data)
        self.assertListEqual(["t2"], train_set._output_data)
        #train_set.plot_random_images(2, 2)

        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2"],
                                   output_name=["t2_"],
                                   batch_size=4, shuffle=False, p_augm=0.5)
        train_set._unpaired = False
        train_set.reset()
        #train_set.plot_random_images(2, 2)

    def test_init_multi_images_segm(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data=["t1", "t2"],
                                   input_name=["image", "t2_"], output_data=["vs"],
                                   output_name=["vs_"], use_filter="vs",
                                   batch_size=4, shuffle=False, p_augm=0.5)
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(0.5, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(False, train_set.shuffle)
        self.assertEqual(377, len(train_set))
        self.assertEqual(1509, train_set._number_index)
        self.assertListEqual(["image", "t2_"], train_set._input_name)
        self.assertListEqual(["vs_"], train_set._output_name)
        self.assertListEqual(["t1", "t2"], train_set._input_data)
        self.assertListEqual(["vs"], train_set._output_data)
        # train_set.plot_random_images(2, 2)

    def test_init_only_class(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["vs_class"],
                                   output_name=["vs_class_"],
                                   batch_size=4, shuffle=False, p_augm=0.5)
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(0.5, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(False, train_set.shuffle)
        self.assertEqual(2606, len(train_set))
        self.assertEqual(10425, train_set._number_index)
        self.assertListEqual(["image"], train_set._input_name)
        self.assertListEqual(["vs_class_"], train_set._output_name)
        self.assertListEqual(["t1"], train_set._input_data)
        self.assertListEqual(["vs_class"], train_set._output_data)
        #train_set.plot_random_images(2, 2)

    def test_init_only_class_balanced(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["vs_class"],
                                   output_name=["vs_class_"], use_balance=True,
                                   batch_size=4, shuffle=False, p_augm=0.5, seed=56459)
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(0.5, train_set.p_augm)
        self.assertEqual((256, 256), train_set.dsize)
        self.assertEqual(False, train_set.shuffle)
        self.assertEqual(831, len(train_set))
        self.assertEqual(3326, train_set._number_index)
        self.assertListEqual(["image"], train_set._input_name)
        self.assertListEqual(["vs_class_"], train_set._output_name)
        self.assertListEqual(["t1"], train_set._input_data)
        self.assertListEqual(["vs_class"], train_set._output_data)

        train_set.batch_size = 1
        counter_0 = 0
        counter_1 = 0
        for idx in range(len(train_set)):
            data = train_set[idx]
            if data[1]["vs_class_"][0] == 0:
                counter_0 += 1
            elif data[1]["vs_class_"][0] == 1:
                counter_1 += 1
        self.assertEqual(counter_0, counter_1)

    def test_setter(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs"], output_name=["output", "mask"],
                                   batch_size=1, shuffle=True, p_augm=0.0)
        # batch_size
        train_set.batch_size = 4
        res = train_set[0]
        self.assertEqual(4, train_set.batch_size)
        self.assertEqual(4, len(res[0]["image"]))
        self.assertEqual(4, len(res[1]["output"]))
        self.assertEqual(4, len(res[1]["mask"]))
        self.assertEqual(4, len(res[1]["mask_2"]))
        # dsize
        train_set.dsize = (200, 200)
        res = train_set[0]
        self.assertEqual((200, 200), train_set.dsize)
        self.assertEqual((200, 200), res[0]["image"].shape[1:])
        self.assertEqual((200, 200), res[1]["output"].shape[1:])
        self.assertEqual((200, 200), res[1]["mask"].shape[1:])
        self.assertEqual((200, 200), res[1]["mask_2"].shape[1:])
        # p augm
        train_set.p_augm = 0.5
        self.assertEqual(0.5, train_set.p_augm)
        # shuffle
        train_set.shuffle = False
        self.assertEqual(False, train_set.shuffle)
        # augm methods
        train_set.augm_methods = []
        self.assertEqual([], train_set.augm_methods)

    def test_reduce_to_nonzero_segm(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs"], output_name=["t2", "mask"],
                                   batch_size=4, shuffle=True, p_augm=0.0)
        train_set.reduce_to_nonzero_segm("vs")
        self.assertEqual(377, len(train_set))

    def test_single_result(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs"], output_name=["output", "mask"],
                                   batch_size=4, shuffle=True, p_augm=0.0)
        result = train_set[1]
        self.assertEqual(tuple, type(result))
        self.assertEqual(2, len(result))
        self.assertEqual(dict, type(result[0]))
        self.assertEqual(dict, type(result[1]))
        self.assertEqual(["image"], list(result[0].keys()))
        self.assertEqual(["output", "mask", "mask_2"], list(result[1].keys()))
        self.assertEqual((4, 256, 256), np.shape(result[0]["image"]))
        self.assertEqual((4, 256, 256), np.shape(result[1]["output"]))
        self.assertEqual((4, 256, 256), np.shape(result[1]["mask"]))
        self.assertEqual((4, 256, 256), np.shape(result[1]["mask_2"]))

    def test_single_result_with_class(self):
        np.random.seed(5)
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs", "vs_class"],
                                   output_name=["output", "mask", "class1"],
                                   batch_size=4, shuffle=False, p_augm=0.0,
                                   alpha=-1, beta=1)
        result = train_set[1]
        self.assertEqual(tuple, type(result))
        self.assertEqual(2, len(result))
        self.assertEqual(dict, type(result[0]))
        self.assertEqual(dict, type(result[1]))
        self.assertEqual(["image"], list(result[0].keys()))
        self.assertEqual(["output", "mask", "class1", "mask_2", "class1_2"], list(result[1].keys()))
        self.assertEqual((4, 256, 256), np.shape(result[0]["image"]))
        self.assertTrue(1 >= np.min(result[0]["image"]) >= -1)
        self.assertTrue(1 >= np.max(result[0]["image"]) >= -1)
        self.assertEqual((4, 256, 256), np.shape(result[1]["output"]))
        self.assertTrue(1 >= np.min(result[1]["output"]) >= -1)
        self.assertTrue(1 >= np.max(result[1]["output"]) >= -1)
        self.assertEqual((4, 256, 256), np.shape(result[1]["mask"]))
        self.assertListEqual([0], list(np.unique(result[1]["mask"])))
        self.assertEqual((4, 256, 256), np.shape(result[1]["mask_2"]))
        self.assertListEqual([0], list(np.unique(result[1]["mask_2"])))
        self.assertEqual((4,), np.shape(result[1]["class1"]))
        self.assertListEqual([0], list(np.unique(result[1]["class1"])))
        self.assertEqual((4,), np.shape(result[1]["class1_2"]))
        self.assertListEqual([0], list(np.unique(result[1]["class1_2"])))

        train_set.reduce_to_nonzero_segm("vs")
        result = train_set[1]
        self.assertListEqual([0, 1], list(np.unique(result[1]["mask"])))
        self.assertListEqual([0, 1], list(np.unique(result[1]["mask_2"])))
        self.assertListEqual([1], list(np.unique(result[1]["class1"])))
        self.assertListEqual([1], list(np.unique(result[1]["class1_2"])))

    def test_range(self):
        np.random.seed(5)
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs", "vs_class"],
                                   output_name=["output", "mask", "class1"],
                                   batch_size=4, shuffle=False, p_augm=0.0,
                                   alpha=0, beta=255)
        result = train_set[1]
        self.assertTrue(255 >= np.min(result[0]["image"]) >= 0)
        self.assertTrue(255 >= np.max(result[0]["image"]) >= 0)
        self.assertTrue(255 >= np.min(result[1]["output"]) >= 0)
        self.assertTrue(255 >= np.max(result[1]["output"]) >= 0)

        np.random.seed(5)
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs", "vs_class"],
                                   output_name=["output", "mask", "class1"],
                                   batch_size=4, shuffle=False, p_augm=0.0,
                                   alpha=-1, beta=1)
        result = train_set[1]
        self.assertTrue(1 >= np.min(result[0]["image"]) >= -1)
        self.assertTrue(1 >= np.max(result[0]["image"]) >= -1)
        self.assertTrue(1 >= np.min(result[1]["output"]) >= -1)
        self.assertTrue(1 >= np.max(result[1]["output"]) >= -1)

        np.random.seed(5)
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs", "vs_class"],
                                   output_name=["output", "mask", "class1"],
                                   batch_size=4, shuffle=False, p_augm=0.0,
                                   alpha=0, beta=1)
        result = train_set[1]
        self.assertTrue(1 >= np.min(result[0]["image"]) >= 0)
        self.assertTrue(1 >= np.max(result[0]["image"]) >= 0)
        self.assertTrue(1 >= np.min(result[1]["output"]) >= 0)
        self.assertTrue(1 >= np.max(result[1]["output"]) >= 0)

    @unittest.skip
    def test_mockup_plot_unpaired(self):
        np.random.seed(1335)
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t2",
                                   input_name="image", output_data=["t1", "vs"], output_name=["output", "mask"],
                                   batch_size=16, shuffle=True, p_augm=0.5)
        train_set.reduce_to_nonzero_segm("vs")
        train_set.plot_random_images(2, 2)

    @unittest.skip
    def test_mockup_plot_unpaired_class(self):
        np.random.seed(1335)
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t2",
                                   input_name="image", output_data=["t1", "vs", "vs_class"],
                                   output_name=["output", "mask", "class1"],
                                   batch_size=16, shuffle=False, p_augm=0.5)
        train_set.plot_random_images(2, 2)

    @unittest.skip
    def test_mockup_plot_paired(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs"], output_name=["output", "mask"],
                                   batch_size=16, shuffle=True, p_augm=1.0)
        train_set.reduce_to_nonzero_segm("vs")
        train_set._unpaired = False
        train_set.reset()
        train_set.plot_random_images(4, 4)

    @unittest.skip
    def test_mockup_plot_paired_class(self):
        train_set = DataSet2DMixed("../../data/VS_segm/VS_registered/training/", input_data="t1",
                                   input_name="image", output_data=["t2", "vs", "vs_class"],
                                   output_name=["output", "mask", "class"],
                                   batch_size=16, shuffle=True, p_augm=1.0)
        train_set._unpaired = False
        train_set.reset()
        train_set.plot_random_images(4, 4)
