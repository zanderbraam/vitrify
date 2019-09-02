from src.models.multi_layer_perceptron import MultiLayerPerceptron
from src.models.soft_decision_tree import SoftBinaryDecisionTree
from src.models.variational_autoencoder import VariationalAutoEncoder
from src.models.convolutional_dnn import ConvDNN
from src.data.make_dataset import load_data, join_data
from src.visualization.visualize import draw_tree
from src.utils import balanced_sample_maker

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


def train_models():

    # Load the data
    data = load_data(dataset="MNIST", already_downloaded=True)

    # Get the number of input features
    n_rows, n_cols = np.shape(data["x_train"])[1:]
    n_features = n_rows * n_cols
    n_classes = np.unique(data["y_train"]).shape[0]

    # Downsample the data
    x_train_flat_ds, y_train_ds, indices = balanced_sample_maker(data["x_train_flat"], data["y_train"], 10000,
                                                                 random_seed=1234)

    x_valid_flat_ds, y_valid_ds, indices = balanced_sample_maker(data["x_valid_flat"], data["y_valid"], 5000,
                                                                 random_seed=1234)

    x_test_flat_ds, y_test_ds, indices = balanced_sample_maker(data["x_test_flat"], data["y_test"], 5000,
                                                               random_seed=1234)

    # Create other data
    x_train_ds = x_train_flat_ds.reshape((x_train_flat_ds.shape[0], n_rows, n_cols))
    y_train_one_hot_ds = tf.keras.utils.to_categorical(y_train_ds, n_classes)

    x_valid_ds = x_valid_flat_ds.reshape((x_valid_flat_ds.shape[0], n_rows, n_cols))
    y_valid_one_hot_ds = tf.keras.utils.to_categorical(y_valid_ds, n_classes)

    x_test_ds = x_test_flat_ds.reshape((x_test_flat_ds.shape[0], n_rows, n_cols))
    y_test_one_hot_ds = tf.keras.utils.to_categorical(y_test_ds, n_classes)

    # Print some info
    unique, counts = np.unique(y_train_ds, return_counts=True)
    print("y_train_ds class count: ", dict(zip(unique, counts)))

    unique, counts = np.unique(y_valid_ds, return_counts=True)
    print("y_valid_ds class count: ", dict(zip(unique, counts)))

    unique, counts = np.unique(y_test_ds, return_counts=True)
    print("y_test_ds class count: ", dict(zip(unique, counts)))

    # --------------------------------------------------------------------------------
    # TRAIN THE VARIATIONAL AUTOENCODER TO FIT THE UNIT FUNCTION

    # Create VAE
    vae = VariationalAutoEncoder(
        name="vae_mnist_ds",
        num_inputs=n_features,
        keras_verbose=True
    )

    # Train VAE
    vae.train(x_train_flat_ds, x_valid_flat_ds, load_model=True)

    # Evaluate VAE
    vae_results = vae.evaluate(x_test_flat_ds)

    # Generate new data
    x_gen_flat = vae.sample(40000)

    # Reshape to images for CCN
    x_gen = np.array([np.reshape(x_gen_flat[i], [n_rows, n_cols]) for i in range(len(x_gen_flat))])

    # --------------------------------------------------------------------------------
    # TRAIN A CNN TO FIT THE MAPPING FUNCTION

    # Create CNN
    cnn = ConvDNN(
        name="cnn_mnist_ds",
        img_rows=n_rows,
        img_cols=n_cols,
        num_outputs=n_classes
    )

    # Train CNN
    cnn.train(x_train_ds, y_train_one_hot_ds, x_valid_ds, y_valid_one_hot_ds, load_model=True)

    # Evaluate CNN
    cnn_results = cnn.evaluate(x_test_ds, y_test_one_hot_ds)

    # Get CNN labels
    y_cnn_train = cnn.predict(x_train_ds)
    y_gen = cnn.predict(x_gen)

    x_both = join_data([x_train_ds, x_gen])
    x_both = x_both.reshape((x_both.shape[0], -1))

    y_both = join_data([y_cnn_train, y_gen])
    x_both, y_both = shuffle(x_both, y_both)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO FIT THE MAPPING FUNCTION

    # Create SDT
    sdt_raw = SoftBinaryDecisionTree(
        name="sdt_raw_mnist_ds",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train SDT RAW
    sdt_raw.train(x_train_flat_ds, y_train_one_hot_ds, x_valid_flat_ds, y_valid_one_hot_ds, load_model=True)

    # Evaluate SDT RAW
    sdt_raw_results = sdt_raw.evaluate(x_test_flat_ds, y_test_one_hot_ds)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE CNN

    # Create SDT CNN
    sdt_cnn = SoftBinaryDecisionTree(
        name="sdt_cnn_mnist_ds",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train SDT CNN
    sdt_cnn.train(x_train_flat_ds, y_cnn_train, x_valid_flat_ds, y_valid_one_hot_ds, load_model=True)

    # Evaluate SDT CNN
    sdt_cnn_results = sdt_cnn.evaluate(x_test_flat_ds, y_test_one_hot_ds)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE CNN WITH VAE

    # Create SDT VAE
    sdt_vae = SoftBinaryDecisionTree(
        name="sdt_cnn_vae_mnist_ds",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train SDT VAE
    sdt_vae.train(x_both, y_both, x_valid_flat_ds, y_valid_one_hot_ds, load_model=True)

    # Evaluate SDT VAE
    sdt_vae_results = sdt_vae.evaluate(x_test_flat_ds, y_test_one_hot_ds)

    # --------------------------------------------------------------------------------

    return vae_results, cnn_results, sdt_raw_results, sdt_cnn_results, sdt_vae_results


if __name__ == '__main__':

    train_models()
