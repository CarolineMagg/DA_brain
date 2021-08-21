import sys
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from pipelines.ClassGuidedSegmentation import ClassGuidedSegmentation
from models.utils import check_gpu

parser = argparse.ArgumentParser(description='Process class-guided, supervised segm pipeline parameters.')
parser.add_argument('--sample_step', dest='sample_step', default=500, type=int,
                    help='sample generation step')
parser.add_argument('--training', dest='training', default="t1", choices=["t1", "t2", "t1_t2"],
                    help='training pipeline')
parser.add_argument('--dsize', dest='dsize', default=(256, 256), type=tuple,
                    help='image size')
parser.add_argument('--batch_size', dest='batch_size', default=4, type=int,
                    help='batch size')
parser.add_argument('--activation', dest='activation', default="relu",
                    help='activation function')
parser.add_argument('--seed', dest='seed', default=1334, type=int,
                    help='set seed for model init')
# parser.add_argument('--p_augm', dest='p_augm', default=0.5, type=float,
#                     help='augmentation probability')
# parser.add_argument('--do_augm', dest='do_augm', default=True, type=bool,
#                     help='perform augmentation or not')
# parser.add_argument('--start_augm_epoch', dest='total_epochs', default=100, type=int,
#                     help='epoch for starting augmentation')
parser.add_argument('--epochs', dest='epochs', default=150, type=int,
                    help='total number of epochs')
parser.add_argument('--data_nr', dest='data_nr', default=0, type=int,
                    help='number of images per epoch/number of steps per epoch')

args = parser.parse_args()

if __name__ == "__main__":
    check_gpu()

    print("Training with UNet_ClassGuided.")
    training = args.training
    dsize = args.dsize
    batch_size = args.batch_size
    activation = args.activation
    seed = args.seed
    # do_augm = args.do_augm
    # p_augm = args.p_augm
    # start_augm_epoch = args.start_augm_epoch
    epochs = args.epochs
    sample_step = args.sample_step
    data_nr = args.data_nr if args.data_nr != 0 else None

    identifier = f"unet_cg_{training}_{activation}_segm_{seed}"
    tensorboard_dir = f"/tf/workdir/DA_brain/logs/{identifier}/"
    checkpoints_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/checkpoints/"
    save_model_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/"
    sample_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/sample_dir/"
    data_dir = "/tf/workdir/data/VS_segm/VS_registered/"
    cg_segm = ClassGuidedSegmentation(data_dir=data_dir,
                                      tensorboard_dir=tensorboard_dir,
                                      checkpoints_dir=checkpoints_dir,
                                      save_model_dir=save_model_dir,
                                      sample_dir=sample_dir,
                                      seed=seed,
                                      sample_step=sample_step,
                                      batch_size=batch_size,
                                      dsize=dsize,
                                      activation=activation)
    cg_segm.train(epochs=epochs, data_nr=data_nr, restore=False)
