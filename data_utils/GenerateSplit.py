########################################################################################################################
# Script to split the dataset into train/val/test subsets.
########################################################################################################################

import os.path
import shutil
from natsort import natsorted

__author__ = "c.magg"


if __name__ == "__main__":

    dataset_path = "/tf/workdir/data/VS_segm/VS_registered"
    folders = natsorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path)])
    paths = ["/tf/workdir/data/VS_segm/VS_registered/training/",
             "/tf/workdir/data/VS_segm/VS_registered/validation/",
             "/tf/workdir/data/VS_segm/VS_registered/test/"]

    print("Total dataset size: ", len(folders))
    test_size = int(len(folders)*0.2)
    train_size = int((len(folders) - test_size)*0.8)
    val_size = len(folders) - test_size - train_size
    print("Train size: ", train_size)
    print("Val size: ", val_size)
    print("Test size: ", test_size)

    indices = [idx for idx in range(len(folders))]
    train_ind = indices[:train_size]
    val_ind = indices[train_size: train_size+val_size]
    test_ind = indices[-test_size:]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)
    for f in folders[:train_size]:
        new_name = os.path.join(paths[0], os.path.basename(f))
        shutil.copytree(f, new_name)
        shutil.rmtree(f)
    for f in folders[train_size: train_size+val_size]:
        new_name = os.path.join(paths[1], os.path.basename(f))
        shutil.copytree(f, new_name)
        shutil.rmtree(f)
    for f in folders[-test_size:]:
        new_name = os.path.join(paths[2], os.path.basename(f))
        shutil.copytree(f, new_name)
        shutil.rmtree(f)
