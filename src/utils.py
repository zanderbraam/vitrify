import os
import numpy as np


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def balanced_sample_maker(x, y, data_size, random_seed=None):
    """ Return a balanced data set by sampling all classes with sample_size
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = np.unique(y)

    class_size = data_size / len(uniq_levels)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # Oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=int(class_size), replace=False).tolist()
        balanced_copy_idx += over_sample_idx

    np.random.shuffle(balanced_copy_idx)

    return x[balanced_copy_idx, :], y[balanced_copy_idx], balanced_copy_idx
