import sys
import os
import logging

logging.basicConfig(level=logging.INFO)

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from pipelines.CGSIFA import CGSIFA
from models.utils import check_gpu

if __name__ == "__main__":
    check_gpu()

    print(f"Pre-Training with CGSIFA.")
    seed = 13387
    d_step = 5
    dsize = (256, 256)
    sample_step = 5
    cycle_consistency_loss = 10
    segm_loss_weight = 5
    identity_loss = 1
    epochs = 51
    segm_epochs = [25]  # [0, 5, 10, 25, 51]
    data_nr = None
    step_decay = 51

    for segm_epoch in segm_epochs:

        identifier = f"pretrained_cgsifa_{d_step}_{epochs}_{segm_epoch}_{seed}_v2"
        tensorboard_dir = f"/tf/workdir/DA_brain/logs/{identifier}/"
        checkpoints_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/checkpoints/"
        save_model_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/"
        sample_dir = f"/tf/workdir/DA_brain/saved_models/{identifier}/sample_dir/"
        data_dir = "/tf/workdir/data/VS_segm/VS_registered/"

        steps = 5
        numbers = epochs // steps
        for idx in range(0, numbers):
            cgsifa = CGSIFA(data_dir=data_dir,
                            tensorboard_dir=tensorboard_dir,
                            checkpoints_dir=checkpoints_dir,
                            save_model_dir=save_model_dir,
                            sample_dir=sample_dir,
                            seed=seed,
                            d_step=d_step,
                            sample_step=sample_step,
                            cycle_loss_weight=cycle_consistency_loss,
                            identity_loss_weight=identity_loss,
                            segm_loss_weight=segm_loss_weight)
            cgsifa.train(epochs0=idx * steps, epochs=idx * steps + steps, segm_epoch=segm_epoch, data_nr=data_nr,
                         restore=True, step_decay=step_decay)
