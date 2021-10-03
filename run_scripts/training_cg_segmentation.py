import sys
import os
import time

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from pipelines.CGSegmentation import CGSegmentation
from models.utils import check_gpu

if __name__ == "__main__":
    check_gpu()

    trainings = ["t1", "t2"]
    dsize = (256, 256)
    batch_size = 8
    activations = ["relu", "leaky_relu", "selu"]
    seed = 13319

    model_type = "XNet"
    print(f"Training with {model_type}.")
    use_balances = ["True"]
    do_augm = True
    p_augm = 0.5
    total_epochs = 150
    start_augm_epoch = 100
    epochs = [0, 100, 150]

    for use_balance in use_balances:
        for activation in activations:
            for training in trainings:

                # directory names
                identifier = f"cg_{model_type}_{training}_{activation}_{use_balance}_segm_{seed}"
                save_model = f"/tf/workdir/DA_brain/saved_models/{identifier}"
                checkpoint_hdf5 = f"/tf/workdir/DA_brain/saved_models/{identifier}/{identifier}.hdf5"
                tensorboard_dir = f"/tf/workdir/DA_brain/logs/{identifier}"

                # start the pipeline
                pipeline = CGSegmentation(save_model_dir=save_model,
                                          checkpoint_dir=checkpoint_hdf5,
                                          tensorboard_dir=tensorboard_dir,
                                          batch_size=batch_size,
                                          dsize=dsize,
                                          activation=activation,
                                          seed=seed,
                                          use_balance=use_balance,
                                          data_type=training,
                                          model_type=model_type)
                pipeline.train(do_augm, epochs, p_augm)
