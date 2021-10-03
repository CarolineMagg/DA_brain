from unittest import TestCase
import numpy as np

from data_utils.DataSet2DMixed import DataSet2DMixed
from inference.InferenceCGSegmentation import InferenceCGSegmentation
from inference.InferenceGT2SSegmS import InferenceGT2SSegmS
from inference.InferenceSimpleSegmentation import InferenceSimpleSegmentation
from models.utils import check_gpu


class TestInferenceSimpleSegmentation(TestCase):
    def setUp(self) -> None:
        self.data = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/validation",
                                   input_data=["t1"], input_name=["image"],
                                   output_data="vs", output_name="vs", use_filter="vs",
                                   batch_size=4, shuffle=False, p_augm=0.0, dsize=(256, 256),
                                   alpha=0, beta=1)

        self.data_multi = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/validation",
                                         input_data=["t1", "t2"], input_name=["image", "t2"],
                                         output_data="vs", output_name="vs", use_filter="vs",
                                         batch_size=4, shuffle=False, p_augm=0.0, dsize=(256, 256),
                                         alpha=0, beta=1)
        check_gpu()

    def test_infer(self):
        inputs = self.data[0][0]["image"]
        inf = InferenceSimpleSegmentation("/tf/workdir/DA_brain/saved_models/XNet_t1_relu_segm_13318", self.data)
        result = inf.infer(inputs)
        self.assertTupleEqual((4, 256, 256, 1), np.shape(result))

    def test_get_k_results(self):
        inf = InferenceSimpleSegmentation("/tf/workdir/DA_brain/saved_models/XNet_t1_relu_segm_13318", self.data)
        inf.get_k_results(do_plot=False)

    def test_evaluate(self):
        inf = InferenceSimpleSegmentation("/tf/workdir/DA_brain/saved_models/XNet_t1_relu_segm_13318", self.data)
        res = inf.evaluate()

    def test_evaluate_multi(self):
        inf = InferenceSimpleSegmentation("/tf/workdir/DA_brain/saved_models/XNet_t1_t2_relu_segm_13318",
                                          self.data_multi)
        res = inf.evaluate()


class TestInferenceGT2SSegmS(TestCase):

    def setUp(self) -> None:
        self.data = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/validation",
                                   input_data=["t1", "t2"], input_name=["image", "t2"],
                                   output_data="vs", output_name="vs", use_filter="vs",
                                   batch_size=1, shuffle=False, p_augm=0.0, dsize=(256, 256),
                                   alpha=-1, beta=1)
        check_gpu()

    def test_infer(self):
        inf = InferenceGT2SSegmS("/tf/workdir/DA_brain/saved_models/gan_2_100_50_13785/G_T2S",
                                 "/tf/workdir/DA_brain/saved_models/XNet_t1_relu_segm_13318",
                                 "/tf/workdir/DA_brain/saved_models/gan_2_100_50_13785/D_S",
                                 self.data)
        self.data.batch_size = 4
        S_gen, segm_pred, a, b = inf.infer(self.data[0][0]["generated_t2"])
        self.assertTupleEqual((4, 256, 256, 1), np.shape(S_gen.numpy()))
        self.assertTupleEqual((4, 256, 256, 1), np.shape(segm_pred))
        self.assertEqual(None, a)
        self.assertEqual(None, b)

    def test_evaluate(self):
        inf = InferenceGT2SSegmS("/tf/workdir/DA_brain/saved_models/gan_2_100_50_13785/G_T2S",
                                 "/tf/workdir/DA_brain/saved_models/XNet_t1_relu_segm_13318",
                                 "/tf/workdir/DA_brain/saved_models/gan_2_100_50_13785/D_S",
                                 self.data)
        result = inf.evaluate()

    def test_get_k_result(self):
        inf = InferenceGT2SSegmS("/tf/workdir/DA_brain/saved_models/gan_2_100_50_13785/G_T2S",
                                 "/tf/workdir/DA_brain/saved_models/XNet_t1_relu_segm_13318",
                                 "/tf/workdir/DA_brain/saved_models/gan_2_100_50_13785/D_S",
                                 self.data)
        inf.get_k_results(do_plot=False)


class TestInferenceCGSegm(TestCase):

    def setUp(self) -> None:
        self.data = DataSet2DMixed("/tf/workdir/data/VS_segm/VS_registered/validation",
                                   input_data=["t1"], input_name=["image"],
                                   output_data=["vs", "vs_class"], output_name=["vs", "vs_class"],
                                   segm_size=0, batch_size=1, shuffle=False, p_augm=0.0, dsize=(256, 256),
                                   alpha=-1, beta=1)
        check_gpu()

    def test_evaluate(self):
        inf = InferenceCGSegmentation("/tf/workdir/DA_brain/saved_models/cg_XNet_t1_selu_True_segm_13319/",
                                      self.data)
        result1 = inf.evaluate()
