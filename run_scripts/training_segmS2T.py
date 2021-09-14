import sys
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from pipelines.SegmS2T import SegmS2T
from models.utils import check_gpu

parser = argparse.ArgumentParser(description='Process SegmS2T pipeline parameters.')
parser.add_argument('--training', dest='training', default="t1", choices=["t1", "t2"],
                    help='training pipeline')
parser.add_argument('--model', dest='model', default="XNet", choices=["UNet", "XNet"],
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
parser.add_argument('--do_augm', dest='do_augm', default=False, type=bool,
                    help='perform augmentation or not')
parser.add_argument('--epochs', dest='epochs', default=2, type=int,
                    help='total number of epochs')
parser.add_argument('--start_augm_epoch', dest='start_augm_epoch', default=51, type=int,
                    help='epoch for starting augmentation')
parser.add_argument('--cycle_gan', dest='cycle_gan', default="gan_2_100_50_13785/G_S2T", type=str,
                    help='cycle gan identifier')

args = parser.parse_args()


if __name__ == "__main__":
    check_gpu()

    training = args.training
    model_type = args.model
    print(f"Training SegmS2T with {model_type}.")
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

    cycle_gan_dir = args.cycle_gan
    cycle_identifier = '_'.join(cycle_gan_dir.split("_")[0:2])
    identifier = f"segmS2T_{model_type}_{activation}_{start_augm_epoch}_{total_epochs}_{cycle_identifier}_{seed}"
    tensorboard_dir = f"/tf/workdir/DA_brain/logs/{identifier}/"
    checkpoints_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/{identifier}.hdf5"
    save_model_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/SegmS2T"
    sample_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/sample_dir/"
    data_dir = "/tf/workdir/data/VS_segm/VS_registered/"
    cyle_gan_segm = SegmS2T(cycle_gan_dir=cycle_gan_dir,
                            tensorboard_dir=tensorboard_dir,
                            save_model_dir=save_model_dir,
                            checkpoint_dir=checkpoints_dir,
                            data_type=training,
                            seed=seed,
                            batch_size=batch_size,
                            model_type=model_type,
                            activation=activation)
    cyle_gan_segm.train(do_augm, epochs, p_augm)
