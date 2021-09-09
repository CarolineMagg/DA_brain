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
parser.add_argument('--seed', dest='seed', default=1334, type=int,
                    help='set seed for pipeline init')
parser.add_argument('--sample_step', dest='sample_step', default=1000, type=int,
                    help='sample generation step')
parser.add_argument('--dsize', dest='dsize', default=(256, 256), type=tuple,
                    help='image size')
parser.add_argument('--epochs', dest='epochs', default=50, type=int,
                    help='number of epochs')
parser.add_argument('--data_nr', dest='data_nr', default=0, type=int,
                    help='number of images per epoch/number of steps per epoch')
parser.add_argument('--step_decay', dest='step_decay', default=0, type=int,
                    help='epoch to start linear decay')
parser.add_argument('--batch_size', dest='batch_size', default=4, type=int,
                    help='batch size')
parser.add_argument('--model_type', dest='model_type', default="XNet", type=str,
                    help='segmentation network architecture')
parser.add_argument('--activation', dest='activation', default="relu", type=str,
                    help='activation function')
parser.add_argument('--cycle_gan', dest='cycle_gan', default="gan_10_100_50_13785", type=str,
                    help='cycle gan identifier')

args = parser.parse_args()

if __name__ == "__main__":
    check_gpu()

    print("Training with SegmS2T.")
    seed = args.seed
    dsize = args.dsize
    epochs = args.epochs
    batch_size = args.batch_size
    model_type = args.model_type
    activation = args.activation
    data_nr = args.data_nr if args.data_nr != 0 else None
    step_decay = args.step_decay if args.step_decay != 0 else None

    identifier = f"segmS2T_{model_type}_{epochs}_{step_decay}_{seed}_4"
    tensorboard_dir = f"/tf/workdir/DA_brain/logs/{identifier}/"
    checkpoints_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/checkpoints/"
    save_model_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/"
    sample_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/sample_dir/"
    data_dir = "/tf/workdir/data/VS_segm/VS_registered/"
    cycle_gan_dir = args.cycle_gan
    cyle_gan_segm = SegmS2T(data_dir=data_dir,
                            tensorboard_dir=tensorboard_dir,
                            checkpoints_dir=checkpoints_dir,
                            save_model_dir=save_model_dir,
                            sample_dir=sample_dir,
                            cycle_gan_dir=cycle_gan_dir,
                            seed=seed,
                            batch_size=batch_size,
                            model_type=model_type,
                            activation=activation)
    cyle_gan_segm.train(epochs=epochs, data_nr=data_nr, restore=False, step_decay=step_decay)
