########################################################################################################################
# Functions for visualizing the 2D data
########################################################################################################################
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2

from data_utils.data_loader import load_data_from_folder, get_non_zero_slices_segmentation

__author__ = "c.magg"


def plot_predictions_overlap(inputs, targets, predictions):
    if type(inputs) == dict:
        images = inputs[list(inputs.keys())[0]]
    else:
        images = inputs
    if type(targets) == dict:
        gt = targets[list(targets.keys())[0]]
    else:
        gt = targets
    nrows = np.shape(images)[0]  # batch_size
    ncols = 2  # GT and Pred
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    counter = 0
    for row in range(nrows):
        for col in range(0, ncols, 2):
            axes[row, col].imshow(images[counter], cmap="gray")
            axes[row, col].imshow(gt[counter], alpha=0.5)
            axes[row, col].set_title("GT")
            axes[row, col].axis("off")

            axes[row, col + 1].imshow(images[counter], cmap="gray")
            axes[row, col + 1].imshow(predictions[counter], alpha=0.5)
            axes[row, col + 1].set_title("Prediction")
            axes[row, col + 1].axis("off")
            counter += 1
    plt.show()


def plot_predictions_separate(inputs, targets, predictions):
    if type(inputs) == dict:
        images = inputs[list(inputs.keys())[0]]
    else:
        images = inputs
    if type(targets) == dict:
        gt = targets[list(targets.keys())[0]]
    else:
        gt = targets
    nrows = np.shape(images)[0]  # batch_size
    ncols = 3  # Input, GT and prediction
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    counter = 0
    for row in range(nrows):
        for col in range(0, ncols, 3):
            axes[row, col].imshow(images[counter], cmap="gray")
            axes[row, col].set_title("Input")
            axes[row, col].axis("off")

            axes[row, col + 1].imshow(gt[counter], cmap="gray")
            axes[row, col + 1].set_title("GT")
            axes[row, col + 1].axis("off")

            axes[row, col + 2].imshow(predictions[counter], cmap="gray")
            axes[row, col + 2].set_title("Prediction")
            axes[row, col + 2].axis("off")
            counter += 1
    plt.show()


def tile_plot_timepoint(data_folder, slice_to_vis=None, figsize=(15, 15), cmap=None):
    """
    Tile plot of one timepoint with T1, T2 and segmentation mask.
    :param data_folder: path to data folder
    :param slice_to_vis: slice to be visualized; if None  - the median slice of non-zero slices will be taken
    :param figsize: figure size
    :param cmap: color map
    """
    # load NIFTI images
    data = load_data_from_folder(data_folder)

    # determine median slice that has segmentation mask
    if slice_to_vis is None:
        slice_to_vis = int(np.round(np.mean(get_non_zero_slices_segmentation(data[2]))))

    # plot
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    fig.tight_layout()
    ax[0].imshow(data[0][:, :, slice_to_vis], cmap=cmap)
    ax[0].set_title("T1")
    ax[1].imshow(data[1][:, :, slice_to_vis], cmap=cmap)
    ax[1].set_title("T2")
    ax[2].imshow(data[2][:, :, slice_to_vis], cmap=cmap)
    ax[2].set_title("SEG")
    fig.suptitle("Data slice {}".format(slice_to_vis), fontsize=16)
    plt.show()


def tile_plot(data_folder, nrows=5, ncols=6, skip=2, cmap=None):
    """
    Tile plot with multiple slides in a grid order.
    :param data_folder: path to data folder
    :param nrows: number of rows in the grid
    :param ncols: number of columns in the grid
    :param skip: number of slices to be skipped (default: every second slice is shown)
    :param cmap: color map
    :return:
    """
    # load NIFTI images
    data = load_data_from_folder(data_folder)

    slices = data[0].shape[2]
    if nrows * ncols > slices:
        raise ValueError("nrows*ncols larger than {}".format(slices))

    # plot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    counter = 0
    for row in range(nrows):
        for col in range(ncols):
            axes[row, col].imshow(data[0][:, :, counter], cmap=cmap)
            axes[row, col].set_title(counter)
            axes[row, col].axis("off")
            counter += skip
    plt.show()


def overlap_plot(data_folder, slice_to_vis=None, figsize=(15, 15), modality="T1", cmap="gray"):
    """
    Plot segmentation mask on top of either T1 or T2 image for one time point.
    :param data_folder: path to data folder
    :param slice_to_vis: slice to be visualized; if None  - the median slice of non-zero slices will be taken
    :param figsize: figure size
    :param cmap: color map
    :param modality: modality to be shown; either T1 or T2
    """
    # load NIFTI images
    if "t1" in modality.lower():
        data = load_data_from_folder(data_folder, subfiles=["vs_gk_t1_refT2.nii", "vs_gk_seg_refT2.nii"])
    elif "t2" in modality.lower():
        data = load_data_from_folder(data_folder, subfiles=["vs_gk_t2_refT2.nii", "vs_gk_seg_refT2.nii"])
    else:
        raise ValueError("Modality needs to be either T1 or T2.")

    # determine median slice that has segmentation mask
    if slice_to_vis is None:
        slice_to_vis = int(np.round(np.mean(get_non_zero_slices_segmentation(data[1]))))

    # process data
    tmp = copy.deepcopy(data[0][:, :, slice_to_vis])
    toshow = cv2.normalize(tmp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    toshow3 = data[1][:, :, slice_to_vis]

    # plot
    if cmap == "gray":
        fig, ax = plt.subplots()
        ax.imshow(toshow, cmap="gray")
        ax.imshow(toshow3 * 200, alpha=0.3)
    else:
        dst = cv2.addWeighted(toshow / 255., 1.0, toshow3, 1.0, 0.0)
        fig = plt.figure(figsize=figsize)
        plt.imshow(dst)
    fig.suptitle("{}, data slice {}".format(modality, slice_to_vis), fontsize=16)
    plt.show()


def overlap_plot_comparison(data_folder, slice_to_vis=None, figsize=(15, 15), cmap="gray"):
    """
    Plot segmentation mask on top of either T1 and T2 image for one time point.
    :param data_folder: path to data folder
    :param slice_to_vis: slice to be visualized; if None  - the median slice of non-zero slices will be taken
    :param figsize: figure size
    :param cmap: color map
    """
    # load NIFTI images
    data = load_data_from_folder(data_folder)

    # determine median slice that has segmentation mask
    if slice_to_vis is None:
        slice_to_vis = int(np.round(np.mean(get_non_zero_slices_segmentation(data[2]))))

    # process data
    tmp = copy.deepcopy(data[0][:, :, slice_to_vis])
    tmp2 = copy.deepcopy(data[1][:, :, slice_to_vis])
    toshow = np.zeros_like(tmp)
    toshow2 = np.zeros_like(tmp2)
    toshow = cv2.normalize(tmp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    toshow2 = cv2.normalize(tmp2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    toshow3 = data[2][:, :, slice_to_vis]

    # plot
    if cmap == "gray":
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(toshow, cmap="gray")
        ax[0].imshow(toshow3 * 200, alpha=0.3)
        ax[0].set_title("T1")
        ax[0].axis("off")
        ax[1].imshow(toshow2, cmap="gray")
        ax[1].imshow(toshow3 * 200, alpha=0.3)
        ax[1].set_title("T2")
        ax[1].axis("off")
    else:
        img = cv2.addWeighted(toshow / 255., 1.0, toshow3, 0.5, 0.0)
        img2 = cv2.addWeighted(toshow2 / 255., 1.0, toshow3, 0.5, 0.0)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(img)
        ax[0].set_title("T1")
        ax[0].axis("off")
        ax[1].imshow(img2)
        ax[1].set_title("T2")
        ax[1].axis("off")
    fig.suptitle("Data slice {}".format(slice_to_vis), fontsize=16)
    plt.show()


def plot_gen_separately(batch, nrows=4, ncols=2, use_random=True):
    """
    Plot examples of dataset in a tile plot with input and target side-by-side.
    :param nrows: number of rows
    :param ncols: number of columns
    :param use_random: use random input/output if more are available
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, figsize=(15, 15))
    counter = 0
    in_keys = list(batch[0].keys())
    out_keys = list(batch[1].keys())

    if nrows * ncols > batch[0][in_keys[0]].shape[0]:
        raise ValueError(f"Not enough data in the batch to plot a tile with {nrows}x{ncols}.")

    for row in range(nrows):
        for col in range(0, ncols * 2, 2):
            in_rand = randrange(len(in_keys)) if use_random else 0
            out_rand = randrange(len(out_keys)) if use_random else 0
            axes[row, col].imshow(batch[0][in_keys[in_rand]][counter], cmap="gray")
            axes[row, col + 1].imshow(batch[1][out_keys[out_rand]][counter], cmap="gray")
            axes[row, col].set_title("{0}, mod: {1}".format(counter, in_keys[in_rand]))
            axes[row, col + 1].set_title("{0}, mod: {1}".format(counter, out_keys[out_rand]))
            axes[row, col].axis("off")
            axes[row, col + 1].axis("off")
            counter += 1
    fig.suptitle("Batch plot side-by-side.", fontsize=16)
    plt.show()


def plot_gen_overlap(batch, nrows=4, ncols=2, use_random=True):
    """
    Plot examples of dataset in a tile plot with input and target overlap.
    :param nrows: number of rows
    :param ncols: number of columns
    :param use_random: use random input/output if more are available
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    counter = 0
    in_keys = list(batch[0].keys())
    out_keys = list(batch[1].keys()) if len(batch) > 1 else []

    if nrows * ncols > batch[0][in_keys[0]].shape[0]:
        raise ValueError(f"Not enough data in the batch to plot a tile with {nrows}x{ncols}.")

    for row in range(nrows):
        for col in range(0, ncols):
            in_rand = randrange(len(in_keys)) if use_random is True else 0
            axes[row, col].imshow(batch[0][in_keys[in_rand]][counter], cmap="gray")
            for out_key in out_keys:
                if np.sum(batch[1][out_key][counter]) != 0:
                    axes[row, col].imshow(batch[1][out_key][counter], alpha=0.3)
            axes[row, col].set_title("{0}, mod: {1}".format(counter, in_keys[in_rand]))
            axes[row, col].axis("off")
            counter += 1
    fig.suptitle("Batch plot overlap.", fontsize=16)
    plt.show()
