import sys
import os
import argparse
import logging
import shutil


logging.basicConfig(level=logging.INFO)

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from pipelines.CGSIFA import CGSIFA
from models.utils import check_gpu

parser = argparse.ArgumentParser(description='Process CG-SIFA pipeline parameters.')
parser.add_argument('--seed', dest='seed', default=1335, type=int,
                    help='set seed for pipeline init')
parser.add_argument('--sample_step', dest='sample_step', default=200, type=int,
                    help='sample generation step')
parser.add_argument('--d_step', dest='d_step', default=5, type=int,
                    help='discriminator training steps')
parser.add_argument('--cycle_loss_weight', dest='cycle_loss_weight', default=10.0, type=float,
                    help='weight for cycle consistency loss')
parser.add_argument('--identity_loss_weight', dest='identity_loss_weight', default=1.0, type=float,
                    help='weight for identity loss')
parser.add_argument('--segm_loss_weight', dest='segm_loss_weight', default=10.0, type=float,
                    help='weight for segm task loss')
parser.add_argument('--segm_g_loss_weight', dest='segm_g_loss_weight', default=1.0, type=float,
                    help='weight for segmentation generator loss')
parser.add_argument('--T_g_loss_weight', dest='T_g_loss_weight', default=1.0, type=float,
                    help='weight for T_g_loss')
parser.add_argument('--dsize', dest='dsize', default=(256, 256), type=tuple,
                    help='image size')
parser.add_argument('--epochs', dest='epochs', default=20, type=int,
                    help='number of epochs')
parser.add_argument('--segm_epoch', dest='segm_epoch', default=0, type=int,
                    help='number of epoch where segmentation training starts')
parser.add_argument('--data_nr', dest='data_nr', default=0, type=int,
                    help='number of images per epoch/number of steps per epoch')
parser.add_argument('--step_decay', dest='step_decay', default=200, type=int,
                    help='epoch to start linear decay')
parser.add_argument('--pretrained', dest='pretrained', default="", type=str,
                    help='pretrained CGSIFA version')

args = parser.parse_args()

if __name__ == "__main__":
    check_gpu()

    print(f"Training with CG-SIFA.")
    seed = 13336  # args.seed
    d_step = args.d_step
    dsize = args.dsize
    sample_step = args.sample_step
    cycle_consistency_loss = args.cycle_loss_weight
    identity_loss = args.identity_loss_weight
    segm_loss_weight = args.segm_loss_weight
    segm_g_loss_weight = args.segm_g_loss_weight
    T_g_loss_weight = args.T_g_loss_weight
    epochs = 15  # args.epochs
    segm_epoch = -1  # args.segm_epoch
    data_nr = args.data_nr if args.data_nr != 0 else None
    step_decay = args.step_decay if args.step_decay != 0 else None

    identifier = f"cgsifa_{d_step}_{epochs}_{segm_epoch}_{seed}"
    tensorboard_dir = f"/tf/workdir/DA_brain/logs/{identifier}/"
    checkpoints_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/checkpoints/"
    save_model_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/"
    sample_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/sample_dir/"
    data_dir = "/tf/workdir/data/VS_segm/VS_registered/"

    # copy pretrained version to checkpoint dir
    args.pretrained = "pretrained_cgsifa_5_40_51_13385"
    if len(args.pretrained) != 0:
        logging.info(f"Set saved model to state of {args.pretrained}.")
        shutil.copytree(f"/tf/workdir/DA_brain/saved_models/{args.pretrained}", save_model_dir)

    steps = 5
    numbers = epochs // steps
    for idx in range(0, numbers):
        sifa = CGSIFA(data_dir=data_dir,
                      tensorboard_dir=tensorboard_dir,
                      checkpoints_dir=checkpoints_dir,
                      save_model_dir=save_model_dir,
                      sample_dir=sample_dir,
                      seed=seed,
                      d_step=d_step,
                      sample_step=sample_step,
                      cycle_loss_weight=cycle_consistency_loss,
                      identity_loss_weight=identity_loss,
                      segm_loss_weight=segm_loss_weight,
                      segm_g_loss_weight=segm_g_loss_weight,
                      T_g_loss_weight=T_g_loss_weight)
        sifa.train(epochs0=idx * steps, epochs=idx * steps + steps, segm_epoch=segm_epoch, data_nr=data_nr,
                   restore=True, step_decay=step_decay)
