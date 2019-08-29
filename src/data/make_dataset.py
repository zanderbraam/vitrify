import os

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from emnist import extract_training_samples, extract_test_samples, list_datasets
import matplotlib.pyplot as plt

from pathlib import Path

import src.utils as utils


def make_dataset(dataset="MNIST"):
    """
    Fetches the raw datasets, split them into train, test and validation sets,
    does the necessary processing and saves it to data (../data).
    """

    project_dir = Path(__file__).resolve().parents[2]

    data_dir = str(project_dir) + "/data/"
    utils.check_folder(data_dir)

    if dataset == "MNIST":
        # See if directory exists
        data_dir_mnist = data_dir + dataset
        utils.check_folder(data_dir_mnist)

        # Download the MNIST dataset from source
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Hold out last 10000 training samples for validation
        x_valid, y_valid = x_train[-10000:], y_train[-10000:]
        x_train, y_train = x_train[:-10000], y_train[:-10000]

        # Retrieve label shapes from training data
        n_classes = np.unique(y_train).shape[0]

        # Convert labels to 1-hot vectors
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, n_classes)
        y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, n_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, n_classes)

        # Normalize inputs and cast to float
        x_train = (x_train / np.max(x_train)).astype(np.float32)
        x_valid = (x_valid / np.max(x_valid)).astype(np.float32)
        x_test = (x_test / np.max(x_test)).astype(np.float32)

        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_valid_flat = x_valid.reshape((x_valid.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))

        # Save all data
        np.save(data_dir_mnist + "/x_train.npy", x_train)
        np.save(data_dir_mnist + "/x_valid.npy", x_valid)
        np.save(data_dir_mnist + "/x_test.npy", x_test)

        np.save(data_dir_mnist + "/x_train_flat.npy", x_train_flat)
        np.save(data_dir_mnist + "/x_valid_flat.npy", x_valid_flat)
        np.save(data_dir_mnist + "/x_test_flat.npy", x_test_flat)

        np.save(data_dir_mnist + "/y_train.npy", y_train)
        np.save(data_dir_mnist + "/y_valid.npy", y_valid)
        np.save(data_dir_mnist + "/y_test.npy", y_test)

        np.save(data_dir_mnist + "/y_train_one_hot.npy", y_train_one_hot)
        np.save(data_dir_mnist + "/y_valid_one_hot.npy", y_valid_one_hot)
        np.save(data_dir_mnist + "/y_test_one_hot.npy", y_test_one_hot)

    elif dataset == "FashionMNIST":
        # See if directory exists
        data_dir_fashion_mnist = data_dir + dataset
        utils.check_folder(data_dir_fashion_mnist)

        # Download the Fashion MNIST dataset from source
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # Hold out last 10000 training samples for validation
        x_valid, y_valid = x_train[-10000:], y_train[-10000:]
        x_train, y_train = x_train[:-10000], y_train[:-10000]

        # Retrieve label shapes from training data
        n_classes = np.unique(y_train).shape[0]

        # Convert labels to 1-hot vectors
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, n_classes)
        y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, n_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, n_classes)

        # Normalize inputs and cast to float
        x_train = (x_train / np.max(x_train)).astype(np.float32)
        x_valid = (x_valid / np.max(x_valid)).astype(np.float32)
        x_test = (x_test / np.max(x_test)).astype(np.float32)

        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_valid_flat = x_valid.reshape((x_valid.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))

        # Save all data
        np.save(data_dir_fashion_mnist + "/x_train.npy", x_train)
        np.save(data_dir_fashion_mnist + "/x_valid.npy", x_valid)
        np.save(data_dir_fashion_mnist + "/x_test.npy", x_test)

        np.save(data_dir_fashion_mnist + "/x_train_flat.npy", x_train_flat)
        np.save(data_dir_fashion_mnist + "/x_valid_flat.npy", x_valid_flat)
        np.save(data_dir_fashion_mnist + "/x_test_flat.npy", x_test_flat)

        np.save(data_dir_fashion_mnist + "/y_train.npy", y_train)
        np.save(data_dir_fashion_mnist + "/y_valid.npy", y_valid)
        np.save(data_dir_fashion_mnist + "/y_test.npy", y_test)

        np.save(data_dir_fashion_mnist + "/y_train_one_hot.npy", y_train_one_hot)
        np.save(data_dir_fashion_mnist + "/y_valid_one_hot.npy", y_valid_one_hot)
        np.save(data_dir_fashion_mnist + "/y_test_one_hot.npy", y_test_one_hot)

    elif dataset == "EMNIST_Letter":
        # See if directory exists
        data_dir_emnist = data_dir + dataset
        utils.check_folder(data_dir_emnist)

        # Download the EMNIST Letters dataset from source
        x_train, y_train = extract_training_samples("letters")
        x_test, y_test = extract_test_samples("letters")

        # Shift labels from [1:26] to [0:25], to make use of tf.keras.utils.to_categorical
        y_train = y_train - 1
        y_test = y_test - 1

        # Hold out last 10000 training samples for validation
        x_valid, y_valid = x_train[-20800:], y_train[-20800:]
        x_train, y_train = x_train[:-20800], y_train[:-20800]

        # Retrieve label shapes from training data
        n_classes = np.unique(y_train).shape[0]

        # Convert labels to 1-hot vectors
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, n_classes)
        y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, n_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, n_classes)

        # Normalize inputs and cast to float
        x_train = (x_train / np.max(x_train)).astype(np.float32)
        x_valid = (x_valid / np.max(x_valid)).astype(np.float32)
        x_test = (x_test / np.max(x_test)).astype(np.float32)

        # Flatten to 1D vectors
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_valid_flat = x_valid.reshape((x_valid.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))

        # Save all data
        np.save(data_dir_emnist + "/x_train.npy", x_train)
        np.save(data_dir_emnist + "/x_valid.npy", x_valid)
        np.save(data_dir_emnist + "/x_test.npy", x_test)

        np.save(data_dir_emnist + "/x_train_flat.npy", x_train_flat)
        np.save(data_dir_emnist + "/x_valid_flat.npy", x_valid_flat)
        np.save(data_dir_emnist + "/x_test_flat.npy", x_test_flat)

        np.save(data_dir_emnist + "/y_train.npy", y_train)
        np.save(data_dir_emnist + "/y_valid.npy", y_valid)
        np.save(data_dir_emnist + "/y_test.npy", y_test)

        np.save(data_dir_emnist + "/y_train_one_hot.npy", y_train_one_hot)
        np.save(data_dir_emnist + "/y_valid_one_hot.npy", y_valid_one_hot)
        np.save(data_dir_emnist + "/y_test_one_hot.npy", y_test_one_hot)

    elif dataset == "EMNIST_Letter_Uppercase":
        # See if directory exists
        data_dir_emnist_uppercase = data_dir + dataset
        utils.check_folder(data_dir_emnist_uppercase)

        # Download the EMNIST Letters dataset from source
        x_train, y_train = extract_training_samples("byclass")
        x_test, y_test = extract_test_samples("byclass")

        # Extract uppercase data
        ix_train = []
        for i, x in enumerate(y_train):
            if 9 < x < 36:
                ix_train.append(i)

        x_train = x_train[ix_train]
        y_train = y_train[ix_train]

        ix_test = []
        for i, x in enumerate(y_test):
            if 9 < x < 36:
                ix_test.append(i)

        x_test = x_test[ix_test]
        y_test = y_test[ix_test]

        # Shift labels from [10:35] to [0:25], to make use of tf.keras.utils.to_categorical
        y_train = y_train - 10
        y_test = y_test - 10

        # Flatten datasets
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))

        # Create balanced dataset
        x_train_flat, y_train, indices = utils.balanced_sample_maker(x_train_flat, y_train, 54100, random_seed=1234)
        x_test_flat, y_test, indices = utils.balanced_sample_maker(x_test_flat, y_test, 9880, random_seed=1234)

        # Hold out last 10000 training samples for validation
        x_valid_flat, y_valid = x_train_flat[-9880:], y_train[-9880:]
        x_train_flat, y_train = x_train_flat[:-9880], y_train[:-9880]

        # Retrieve label shapes from training data
        n_classes = np.unique(y_train).shape[0]

        # Convert labels to 1-hot vectors
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, n_classes)
        y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, n_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, n_classes)

        # Normalize inputs and cast to float
        x_train_flat = (x_train_flat / np.max(x_train_flat)).astype(np.float32)
        x_valid_flat = (x_valid_flat / np.max(x_valid_flat)).astype(np.float32)
        x_test_flat = (x_test_flat / np.max(x_test_flat)).astype(np.float32)

        # Recreate images
        x_train = np.array([np.reshape(x_train_flat[i], [28, 28]) for i in range(len(x_train_flat))])
        x_valid = np.array([np.reshape(x_valid_flat[i], [28, 28]) for i in range(len(x_valid_flat))])
        x_test = np.array([np.reshape(x_test_flat[i], [28, 28]) for i in range(len(x_test_flat))])

        # Save all data
        np.save(data_dir_emnist_uppercase + "/x_train.npy", x_train)
        np.save(data_dir_emnist_uppercase + "/x_valid.npy", x_valid)
        np.save(data_dir_emnist_uppercase + "/x_test.npy", x_test)

        np.save(data_dir_emnist_uppercase + "/x_train_flat.npy", x_train_flat)
        np.save(data_dir_emnist_uppercase + "/x_valid_flat.npy", x_valid_flat)
        np.save(data_dir_emnist_uppercase + "/x_test_flat.npy", x_test_flat)

        np.save(data_dir_emnist_uppercase + "/y_train.npy", y_train)
        np.save(data_dir_emnist_uppercase + "/y_valid.npy", y_valid)
        np.save(data_dir_emnist_uppercase + "/y_test.npy", y_test)

        np.save(data_dir_emnist_uppercase + "/y_train_one_hot.npy", y_train_one_hot)
        np.save(data_dir_emnist_uppercase + "/y_valid_one_hot.npy", y_valid_one_hot)
        np.save(data_dir_emnist_uppercase + "/y_test_one_hot.npy", y_test_one_hot)

    else:
        print("Please choose either: 'MNIST', 'FashionMNIST', 'EMNIST_Letter' or 'EMNIST_Letter_Uppercase'.")


def load_data(dataset: str, already_downloaded=True):

    # Create an empty data dictionary that contains all the relevant data
    data_dict = {}
    if already_downloaded:
        project_dir = Path(__file__).resolve().parents[2]
        data_dir = str(project_dir) + '/data/' + dataset

        for f in os.listdir(data_dir):
            data_dict[f.replace(".npy", "")] = np.load(os.path.join(data_dir, f))

        return data_dict

    else:
        make_dataset(dataset)
        project_dir = Path(__file__).resolve().parents[2]
        data_dir = str(project_dir) + '/data/' + dataset

        for f in os.listdir(data_dir):
            data_dict[f.replace(".npy", "")] = np.load(os.path.join(data_dir, f))

        return data_dict


def join_data(data: list) -> np.ndarray:
    """
    Join two datasets, e.g. X_1 (60000, 28, 28) + X_2 (40000, 28, 28) => X_12 (100000, 28, 28)
    """
    new_data = data[0]
    for i in range(len(data) - 1):
        new_data = np.vstack((new_data, data[i + 1]))
    return new_data


if __name__ == "__main__":

    make_dataset(dataset="MNIST")
