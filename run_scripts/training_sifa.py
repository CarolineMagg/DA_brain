import sys
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from pipelines.SIFA import SIFA
from models.utils import check_gpu

parser = argparse.ArgumentParser(description='Process supervised segm pipeline parameters.')
parser.add_argument('--seed', dest='seed', default=1334, type=int,
                    help='set seed for pipeline init')
parser.add_argument('--sample_step', dest='sample_step', default=500, type=int,
                    help='sample generation step')
parser.add_argument('--d_step', dest='d_step', default=1, type=int,
                    help='discriminator training steps')
parser.add_argument('--cycle_loss_weight', dest='cycle_loss_weight', default=10.0, type=float,
                    help='weight for cycle consistency loss')
parser.add_argument('--identity_loss_weight', dest='identity_loss_weight', default=1.0, type=float,
                    help='weight for identity loss')
parser.add_argument('--dsize', dest='dsize', default=(256, 256), type=tuple,
                    help='image size')
parser.add_argument('--epochs', dest='epochs', default=50, type=int,
                    help='number of epochs')
parser.add_argument('--data_nr', dest='data_nr', default=0, type=int,
                    help='number of images per epoch/number of steps per epoch')
parser.add_argument('--step_decay', dest='step_decay', default=0, type=int,
                    help='epoch to start linear decay')

args = parser.parse_args()

if __name__ == "__main__":
    check_gpu()

    print(f"Training with SIFA.")
    seed = args.seed
    d_step = args.d_step
    dsize = args.dsize
    sample_step = args.sample_step
    cycle_consistency_loss = args.cycle_loss_weight
    identity_loss = args.identity_loss_weight
    epochs = args.epochs
    data_nr = args.data_nr if args.data_nr != 0 else None
    step_decay = args.step_decay if args.step_decay != 0 else None

    tensorboard_dir = "/tf/workdir/DA_brain/logs/sifa_{}/".format(seed)
    checkpoints_dir = "/tf/workdir/DA_brain/saved_models/sifa_{}/checkpoints/".format(seed)
    save_model_dir = "/tf/workdir/DA_brain/saved_models/sifa_{}".format(seed)
    sample_dir = "/tf/workdir/DA_brain/saved_models/sifa_{}/sample_dir/".format(seed)
    data_dir = "/tf/workdir/data/VS_segm/VS_registered/"
    sifa = SIFA(data_dir=data_dir,
                tensorboard_dir=tensorboard_dir,
                checkpoints_dir=checkpoints_dir,
                save_model_dir=save_model_dir,
                sample_dir=sample_dir,
                seed=seed,
                d_step=d_step,
                sample_step=sample_step,
                cycle_loss_weight=cycle_consistency_loss,
                identity_loss_weight=identity_loss)
    sifa.train(epochs=epochs, data_nr=data_nr, restore=False, step_decay=step_decay)