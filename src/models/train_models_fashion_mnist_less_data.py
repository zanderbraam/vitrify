from src.models.multi_layer_perceptron import MultiLayerPerceptron
from src.models.soft_decision_tree import SoftBinaryDecisionTree
from src.models.variational_autoencoder import VariationalAutoEncoder
from src.models.convolutional_dnn import ConvDNN
from src.data.make_dataset import load_data, join_data
from src.visualization.visualize import draw_tree
from src.utils import balanced_sample_maker

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf


def train_models():

    # Load the data
    data = load_data(dataset="FashionMNIST", already_downloaded=True)

    # Get the number of input features
    n_rows, n_cols = np.shape(data["x_train"])[1:]
    n_features = n_rows * n_cols
    n_classes = np.unique(data["y_train"]).shape[0]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Downsample the data
    x_train_flat_ds, y_train_ds, indices = balanced_sample_maker(data["x_train_flat"], data["y_train"], 5000,
                                                                 random_seed=1234)

    x_valid_flat_ds, y_valid_ds, indices = balanced_sample_maker(data["x_valid_flat"], data["y_valid"], 2500,
                                                                 random_seed=1234)

    x_test_flat_ds, y_test_ds, indices = balanced_sample_maker(data["x_test_flat"], data["y_test"], 2500,
                                                               random_seed=1234)

    # Create other data
    x_train_ds = x_train_flat_ds.reshape((x_train_flat_ds.shape[0], n_rows, n_cols))
    y_train_one_hot_ds = tf.keras.utils.to_categorical(y_train_ds, n_classes)

    x_valid_ds = x_valid_flat_ds.reshape((x_valid_flat_ds.shape[0], n_rows, n_cols))
    y_valid_one_hot_ds = tf.keras.utils.to_categorical(y_valid_ds, n_classes)

    x_test_ds = x_test_flat_ds.reshape((x_test_flat_ds.shape[0], n_rows, n_cols))
    y_test_one_hot_ds = tf.keras.utils.to_categorical(y_test_ds, n_classes)

    # # --------------------------------------------------------------------------------
    # TRAIN THE VARIATIONAL AUTOENCODER TO FIT THE UNIT FUNCTION

    # Create VAE
    vae = VariationalAutoEncoder(
        name="vae_fashion_mnist_ds",
        keras_verbose=2,
        num_inputs=n_features,
        encoder_layers=[
            [512, "relu", 0.0, 0.1, True, "gaussian"],
            [256, "relu", 0.0, 0.1, True, "gaussian"],
            [128, "relu", 0.0, 0.1, True, "gaussian"]
        ],
        decoder_layers=[
            [128, "relu", 0.0, 0.1, True, "gaussian"],
            [256, "relu", 0.0, 0.1, True, "gaussian"],
            [512, "relu", 0.0, 0.1, True, "gaussian"]
        ],
        batch_size=128,
        learning_rate=0.001,
        stopping_patience=5,
        epochs=2000
    )

    # Train VAE
    vae.train(x_train_flat_ds, x_valid_flat_ds, load_model=True)

    # Evaluate VAE
    vae_results = vae.evaluate(x_test_flat_ds)

    # Generate new data
    x_gen_flat = vae.sample(50000)

    # Reshape to images for CCN
    x_gen = np.array([np.reshape(x_gen_flat[i], [n_rows, n_cols]) for i in range(len(x_gen_flat))])

    # import matplotlib.pyplot as plt
    #
    # for img in x_gen:
    #     plt.figure(figsize=(1, 1))
    #     plt.axis('off')
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
    #     plt.close()
    #
    # quit(-1)

    # import matplotlib.pyplot as plt
    #
    # x_test_encoded, _, _ = vae_mnist.predict(data["x_test_flat"])
    # f = plt.figure(figsize=(6, 6))
    # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=data["y_test"])
    # plt.colorbar()
    # plt.show()
    # plt.close()

    # --------------------------------------------------------------------------------
    # TRAIN A CNN TO FIT THE MAPPING FUNCTION

    # Create CNN
    cnn = ConvDNN(
        name="cnn_fashion_mnist_ds",
        img_rows=n_rows,
        img_cols=n_cols,
        num_outputs=n_classes,
        keras_verbose=2,
        print_model_summary=False,
        conv_layers=[
            ["conv2d", 64, (3, 3), (1, 1), "valid", "relu", True, 0.0, 0.0, 0.0, 0.2],
            ["conv2d", 64, (3, 3), (1, 1), "valid", "relu", True, 0.0, 0.0, 0.0, 0.2],
            ["max_pool2d", (2, 2), None, "valid", 0.0],
            ["conv2d", 64, (3, 3), (1, 1), "valid", "relu", True, 0.0, 0.0, 0.0, 0.2],
            ["conv2d", 64, (3, 3), (1, 1), "valid", "relu", True, 0.0, 0.0, 0.0, 0.2],
            ["max_pool2d", (2, 2), None, "valid", 0.0]
        ],
        ff_layers=[
            [512, "relu", 0.0, 0.2, True, "normal"],
            [512, "relu", 0.0, 0.2, True, "normal"]
        ],
        stopping_patience=20,
        epochs=1000
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
        name="sdt_raw_fashion_mnist_ds",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=4,
        keras_verbose=2,
        penalty_decay=0.5,
        inv_temp=0.01,
        ema_win_size=1000,
        penalty_strength=1e+1,
        batch_size=4,
        learning_rate=5e-03,
        stopping_patience=10
    )

    # Train SDT RAW
    sdt_raw.train(x_train_flat_ds, y_train_one_hot_ds, x_valid_flat_ds, y_valid_one_hot_ds, load_model=True)

    # Evaluate SDT RAW
    sdt_raw_results = sdt_raw.evaluate(x_test_flat_ds, y_test_one_hot_ds)

    # # Visualize tree
    # draw_tree(sdt_raw_mnist, n_rows, n_cols)
    #
    # digit = 2
    #
    # sample_index = np.random.choice(np.where(np.argmax(data["y_test_one_hot"], axis=1) == digit)[0])
    # input_img = data["x_test"][sample_index]
    #
    # draw_tree(sdt_raw_mnist, n_rows, n_cols, input_img=input_img, show_correlation=True)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE CNN

    # Create SDT CNN
    sdt_cnn = SoftBinaryDecisionTree(
        name="sdt_cnn_fashion_mnist_ds",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=4,
        keras_verbose=2,
        penalty_decay=0.5,
        inv_temp=0.01,
        ema_win_size=1000,
        penalty_strength=1e+1,
        batch_size=4,
        learning_rate=5e-03,
        stopping_patience=5
    )

    # Train SDT CNN
    sdt_cnn.train(x_train_flat_ds, y_cnn_train, x_valid_flat_ds, y_valid_one_hot_ds, load_model=True)

    # Evaluate SDT CNN
    sdt_cnn_results = sdt_cnn.evaluate(x_test_flat_ds, y_test_one_hot_ds)

    # digit = 8
    #
    # sample_index = np.random.choice(np.where(np.argmax(data["y_test_one_hot"], axis=1) == digit)[0])
    # input_img = data["x_test"][sample_index]
    #
    # draw_tree(sdt_cnn, n_rows, n_cols, input_img=input_img, show_correlation=True)
    # quit(-1)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE CNN WITH VAE

    # Create SDT VAE
    sdt_vae = SoftBinaryDecisionTree(
        name="sdt_cnn_vae_fashion_mnist_ds",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=4,
        keras_verbose=2,
        penalty_decay=0.5,
        inv_temp=0.01,
        ema_win_size=1000,
        penalty_strength=1e+1,
        batch_size=4,
        learning_rate=5e-03,
        stopping_patience=5,
    )

    # Train SDT VAE
    sdt_vae.train(x_both, y_both, x_valid_flat_ds, y_valid_one_hot_ds, load_model=True)

    # Evaluate SDT VAE
    sdt_vae_results = sdt_vae.evaluate(x_test_flat_ds, y_test_one_hot_ds)

    # --------------------------------------------------------------------------------

    return vae_results, sdt_raw_results, sdt_cnn_results, sdt_vae_results


if __name__ == '__main__':

    train_models()


