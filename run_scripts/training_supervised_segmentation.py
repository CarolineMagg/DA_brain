import sys
import os
import argparse

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from pipelines.SimpleSegmentation import SimpleSegmentation
from models.utils import check_gpu

parser = argparse.ArgumentParser(description='Process supervised segmentation pipeline parameters.')
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
parser.add_argument('--p_augm', dest='p_augm', default=0.5, type=float,
                    help='augmentation probability')
parser.add_argument('--do_augm', dest='do_augm', default=True, type=bool,
                    help='perform augmentation or not')
parser.add_argument('--epochs', dest='epochs', default=150, type=int,
                    help='total number of epochs')
parser.add_argument('--start_augm_epoch', dest='start_augm_epoch', default=100, type=int,
                    help='epoch for starting augmentation')

args = parser.parse_args()

if __name__ == "__main__":
    check_gpu()

    training = args.training
    model_type = args.model
    print(f"Training with {model_type}.")
    dsize = args.dsize
    batch_size = args.batch_size
    activation = args.activation
    seed = args.seed
    do_augm = args.do_augm
    p_augm = args.p_augm
    total_epochs = args.epochs
    start_augm_epoch = args.start_augm_epoch
    if do_augm:
        epochs = [0, start_augm_epoch, total_epochs]
    else:
        epochs = [0, total_epochs]

    # directory names
    identifier = f"{model_type}_{training}_{activation}_segm_{seed}"
    save_model = f"/tf/workdir/DA_brain/saved_models/{identifier}"
    checkpoint_hdf5 = f"/tf/workdir/DA_brain/saved_models/{identifier}/{identifier}.hdf5"
    tensorboard_dir = f"/tf/workdir/DA_brain/logs/{identifier}"

    # start the pipeline
    pipeline = SimpleSegmentation(save_model_dir=save_model,
                                  checkpoint_dir=checkpoint_hdf5,
                                  tensorboard_dir=tensorboard_dir,
                                  batch_size=batch_size,
                                  dsize=dsize,
                                  activation=activation,
                                  seed=seed,
                                  data_type=training,
                                  model_type=model_type)
    pipeline.train(do_augm, epochs, p_augm)
