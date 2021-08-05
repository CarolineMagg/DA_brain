import importlib
import sys
import os
import argparse

from pipeline.CycleGAN import CycleGAN

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

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

args = parser.parse_args()

if __name__ == "__main__":
    check_gpu()

    training = args.training  # args.training
    module = importlib.import_module("models")
    model_type = args.model
    model_class = getattr(module, model_type)
    print(f"Training with {model_type}.")
    seed = args.seed
    d_step = args.d_step
    dsize = args.dsize
    sample_step = args.sample_step
    cycle_consistency_loss = args.cycle_loss_weight
    identity_loss = args.identity_loss_weight
    epochs = args.epoch
    data_nr = args.data_nr if args.data_nr != 0 else None

    tensorboard_dir = "/tf/workdir/DA_brain/logs/gan_{}/".format(seed)
    checkpoints_dir = "/tf/workdir/DA_brain/saved_models/gan_{}/checkpoints/".format(seed)
    save_model_dir = "/tf/workdir/DA_brain/saved_models/gan_{}".format(seed)
    sample_dir = "/tf/workdir/DA_brain/saved_models/gan_{}/sample_dir/".format(seed)
    data_dir = "/tf/workdir/data/VS_segm/VS_registered/"
    cyclegan = CycleGAN(data_dir=data_dir,
                        tensorboard_dir=tensorboard_dir,
                        checkpoints_dir=checkpoints_dir,
                        save_model_dir=save_model_dir,
                        sample_dir=sample_dir,
                        seed=seed,
                        d_step=d_step,
                        sample_step=sample_step,
                        cycle_loss_weight=cycle_consistency_loss,
                        identity_loss_weight=identity_loss)
    cyclegan.train(epochs=epochs, data_nr=data_nr, restore=False)

