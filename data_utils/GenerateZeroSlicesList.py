########################################################################################################################
# Script to generate additional information about T1 slices that are empty
# will generate a txt file in the dataset folder called "vs_gk_t1_info.txt" including the first non-empty slice index
########################################################################################################################

import logging
import os
import cv2
import numpy as np
from data_utils.DataSet2D import DataSet2D


class DataSetListed(DataSet2D):

    def __init__(self, dataset_folder, batch_size=4, input_data="t1", input_name="image", shuffle=True, p_augm=0.0,
                 use_filter=None, dsize=(256, 256), alpha=-1, beta=1, seed=13375):
        super().__init__(dataset_folder, batch_size, input_data, input_name, shuffle, p_augm, use_filter, False, dsize, alpha,
                         beta, seed)
        self.listed = []

    def _load_data_sample(self, ds, data_type, item):
        sample = getattr(self._data[ds], self.lookup_data_call()[data_type])(item)
        if data_type in ["t1", "t2"]:
            if np.sum(sample) != 0:
                sample = np.clip(sample, np.percentile(sample, 1), np.percentile(sample, 99))  # clip extrem values
                sample = (sample - np.mean(sample)) / np.std(sample)  # z score normalization
                sample = ((sample - sample.min()) / (sample.max() - sample.min())) * 2 - 1  # range [-1, 1]
                # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            else:
                self.listed.append((self._data[ds]._path_dir, item))
                logging.debug(f"DataSet2D: image data of {self._data[ds]._path_dir}, {item} is empty.")
            return cv2.resize(sample, dsize=self._dsize, interpolation=cv2.INTER_CUBIC)
        elif data_type in ["vs", "cochlea"]:
            return cv2.resize(sample, dsize=self._dsize, interpolation=cv2.INTER_CUBIC)
        elif data_type in ["vs_class", "cochlea_class"]:
            return sample
        else:
            raise ValueError(f"DataSet2D: datatype {data_type} not supported.")


if __name__ == "__main__":

    paths = ["/tf/workdir/data/VS_segm/VS_registered/training/",
             "/tf/workdir/data/VS_segm/VS_registered/validation/",
             "/tf/workdir/data/VS_segm/VS_registered/test/"]
    batch_sizes = [10450 // 50, 2960 // 16, 3960 // 30]
    for path, bs in zip(paths, batch_sizes):
        dataset = DataSetListed(path)
        dataset.batch_size = bs
        for idx in range(len(dataset)):
            res = dataset[idx]
        listed_path = []
        listed_idx = []
        for elem in dataset.listed:
            p, idx = elem[0], elem[1]
            if p not in listed_path:
                listed_path.append(p)
                listed_idx.append(idx)
            else:
                ix = listed_path.index(p)
                if listed_idx[ix] < idx:
                    listed_idx[ix] = idx
        [print(p, idx) for p, idx in zip(listed_path, listed_idx)]
        for p, idx in zip(listed_path, listed_idx):
            files = os.path.join(p, "vs_gk_t1_info.txt")
            with open(files, 'w') as f:
                f.write(str(idx+1))

