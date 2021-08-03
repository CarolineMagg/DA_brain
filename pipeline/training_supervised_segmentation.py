import importlib
import sys
import os
import cv2
import albumentations as A
import argparse

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from pipeline.SimpleSegmentation import SimpleSegmentation
from data_utils.DataSet2DPaired import DataSet2DPaired
from models.utils import check_gpu

parser = argparse.ArgumentParser(description='Process supervised segm pipeline parameters.')
parser.add_argument('--training', dest='training', default="t1", choices=["t1", "t2", "t1_t2"],
                    help='training pipeline')
parser.add_argument('--model', dest='model', default="UNet", choices=["UNet", "XNet"],
                    help='model architecture')
parser.add_argument('--dsize', dest='dsize', default=(256, 256), type=tuple,
                    help='image size')
parser.add_argument('--batch_size', dest='batch_size', default=4, type=int,
                    help='batch size')
parser.add_argument('--activation', dest='activation', default="relu",
                    help='activation function')
parser.add_argument('--seed', dest='seed', default=1334, type=int,
                    help='set seed for model init')


args = parser.parse_args()

if __name__ == "__main__":
    check_gpu()

    training = args.training  # args.training
    module = importlib.import_module("models")
    model_type = args.model
    model_class = getattr(module, model_type)
    print(f"Training with {model_type}.")
    dsize = args.dsize
    batch_size = args.batch_size
    activation = args.activation
    seed = args.seed

    if training == "t1":
        print("Training vs with t1.")

        # dataset
        train_set = DataSet2DPaired("../../data/VS_segm/VS_registered/training/", input_data=["t1"],
                                    input_name=["image"], output_data="vs", output_name="vs", batch_size=batch_size,
                                    shuffle=True, use_filter="vs", dsize=dsize, p_augm=0.0)
        train_set.augm_methods = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                     A.MedianBlur(p=0.5, blur_limit=5),
                     A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
        ]
        val_set = DataSet2DPaired("../../data/VS_segm/VS_registered/validation/", input_data=["t1"],
                                  input_name=["image"], output_data="vs", output_name="vs", batch_size=batch_size,
                                  shuffle=True, use_filter="vs", dsize=dsize, p_augm=0.0)
        test_set = DataSet2DPaired("../../data/VS_segm/VS_registered/test/", input_data=["t1"],
                                   input_name=["image"], output_data="vs", output_name="vs", batch_size=batch_size,
                                   shuffle=False, use_filter="vs", dsize=dsize, p_augm=0.0)

        # model
        model = model_class(activation=activation, input_name="image", output_name="vs", input_shape=(*dsize, 1),
                            seed=seed).generate_model()

    elif training == "t2":
        print("Training vs with t2.")

        # dataset
        train_set = DataSet2DPaired("../../data/VS_segm/VS_registered/training/", input_data=["t2"],
                                    input_name=["image"], output_data="vs", output_name="vs", batch_size=batch_size,
                                    shuffle=True, use_filter="vs", dsize=dsize, p_augm=0.0)
        train_set.augm_methods = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                     A.MedianBlur(p=0.5, blur_limit=5),
                     A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
        ]
        val_set = DataSet2DPaired("../../data/VS_segm/VS_registered/validation/", input_data=["t2"],
                                  input_name=["image"], output_data="vs", output_name="vs", batch_size=batch_size,
                                  shuffle=True, use_filter="vs", dsize=dsize, p_augm=0.0)
        test_set = DataSet2DPaired("../../data/VS_segm/VS_registered/test/", input_data=["t2"],
                                   input_name=["image"], output_data="vs", output_name="vs", batch_size=batch_size,
                                   shuffle=False, use_filter="vs", dsize=dsize, p_augm=0.0)

        # model
        model = model_class(activation=activation, input_name="image", output_name="vs", input_shape=(*dsize, 1),
                            seed=seed).generate_model()

    elif training == "t1_t2":
        print("Training vs with t1 and t2.")

        # dataset
        train_set = DataSet2DPaired("../../data/VS_segm/VS_registered/training/", input_data=["t1", "t2"],
                                    input_name=["image", "t2"], output_data="vs", output_name="vs",
                                    batch_size=batch_size,
                                    shuffle=True, use_filter="vs", dsize=dsize, p_augm=0.0)
        train_set.augm_methods = [
            A.ShiftScaleRotate(p=0.5, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.OneOf([A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                     A.MedianBlur(p=0.5, blur_limit=5),
                     A.MotionBlur(p=0.5, blur_limit=(3, 5))], p=0.5)
        ]
        val_set = DataSet2DPaired("../../data/VS_segm/VS_registered/validation/", input_data=["t1", "t2"],
                                  input_name=["image", "t2"], output_data="vs", output_name="vs", batch_size=batch_size,
                                  shuffle=True, use_filter="vs", dsize=dsize, p_augm=0.0)
        test_set = DataSet2DPaired("../../data/VS_segm/VS_registered/test/", input_data=["t1", "t2"],
                                   input_name=["image", "t2"], output_data="vs", output_name="vs",
                                   batch_size=batch_size,
                                   shuffle=False, use_filter="vs", dsize=dsize, p_augm=0.0)

        # model
        model = model_class(activation=activation, input_name=["image", "t2"], output_name="vs",
                            input_shape=(*dsize, 1), seed=seed).generate_model()

    else:
        raise ValueError("training not valid.")

    # callbacks
    identifier = f"{model_type}_{training}_segm_Aug02"

    pipeline = SimpleSegmentation(identifier)
    pipeline.set_model(model)
    pipeline.set_data(train_set, val_set, test_set)

    pipeline.fit(init_epoch=0, epochs=2)

    pipeline.train_set.p_augm = 0.5
    pipeline.fit(init_epoch=2, epochs=3)

    pipeline.evaluate()
