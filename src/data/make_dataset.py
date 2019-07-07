import logging

import tensorflow as tf
import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

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

    elif dataset == "Letter":
        # See if directory exists
        data_dir_letter = data_dir + dataset
        utils.check_folder(data_dir_letter)

        # Download data from UCI
        data = pd.read_csv(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
            header=None)

        # Separate labels
        y = data.iloc[:, 0]
        x = data.iloc[:, 1:]

        # Create training, validation and test data
        x_train = np.array(x[:10000])
        x_valid = np.array(x[10000:15000])
        x_test = np.array(x[15000:])

        y_train = np.array(y[:10000])
        y_valid = np.array(y[10000:15000])
        y_test = np.array(y[15000:])

        # Normalize inputs and cast to float
        x_train = (x_train / np.max(x_train)).astype(np.float32)
        x_valid = (x_valid / np.max(x_valid)).astype(np.float32)
        x_test = (x_test / np.max(x_test)).astype(np.float32)

        # Change letters to numbers
        y_train = np.array([ord(char) - 65 for char in y_train])
        y_valid = np.array([ord(char) - 65 for char in y_valid])
        y_test = np.array([ord(char) - 65 for char in y_test])

        # Retrieve label shapes from training data
        n_classes = np.unique(y_train).shape[0]

        # Convert labels to 1-hot vectors
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, n_classes)
        y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, n_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, n_classes)

        # Save all data
        np.save(data_dir_letter + "/x_train.npy", x_train)
        np.save(data_dir_letter + "/x_valid.npy", x_valid)
        np.save(data_dir_letter + "/x_test.npy", x_test)

        np.save(data_dir_letter + "/y_train.npy", y_train)
        np.save(data_dir_letter + "/y_valid.npy", y_valid)
        np.save(data_dir_letter + "/y_test.npy", y_test)

        np.save(data_dir_letter + "/y_train_one_hot.npy", y_train_one_hot)
        np.save(data_dir_letter + "/y_valid_one_hot.npy", y_valid_one_hot)
        np.save(data_dir_letter + "/y_test_one_hot.npy", y_test_one_hot)

    elif dataset == "Connect4":
        # See if directory exists
        data_dir_letter = data_dir + dataset
        utils.check_folder(data_dir_letter)

        # Download data from OPENML
        data = pd.read_csv(
            "https://www.openml.org/data/get_csv/4965243/connect-4.arff")

        # Separate labels
        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]

        # Create training, validation and test data
        x_train = np.array(x[:33757])
        x_valid = np.array(x[33757:50657])
        x_test = np.array(x[50657:])

        y_train = np.array(y[:33757])
        y_valid = np.array(y[33757:50657])
        y_test = np.array(y[50657:])

        # Normalize inputs and cast to float
        x_train = (x_train / np.max(x_train)).astype(np.float32)
        x_valid = (x_valid / np.max(x_valid)).astype(np.float32)
        x_test = (x_test / np.max(x_test)).astype(np.float32)

        # Retrieve label shapes from training data
        n_classes = np.unique(y_train).shape[0]

        # Convert labels to 1-hot vectors
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, n_classes)
        y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, n_classes)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, n_classes)

        # Save all data
        np.save(data_dir_letter + "/x_train.npy", x_train)
        np.save(data_dir_letter + "/x_valid.npy", x_valid)
        np.save(data_dir_letter + "/x_test.npy", x_test)

        np.save(data_dir_letter + "/y_train.npy", y_train)
        np.save(data_dir_letter + "/y_valid.npy", y_valid)
        np.save(data_dir_letter + "/y_test.npy", y_test)

        np.save(data_dir_letter + "/y_train_one_hot.npy", y_train_one_hot)
        np.save(data_dir_letter + "/y_valid_one_hot.npy", y_valid_one_hot)
        np.save(data_dir_letter + "/y_test_one_hot.npy", y_test_one_hot)

    else:
        print("Please choose either: 'MNIST', 'Letter' or 'Connect4'.")


if __name__ == "__main__":

    make_dataset(dataset="MNIST")
