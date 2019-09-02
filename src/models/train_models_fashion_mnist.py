from src.models.multi_layer_perceptron import MultiLayerPerceptron
from src.models.soft_decision_tree import SoftBinaryDecisionTree
from src.models.variational_autoencoder import VariationalAutoEncoder
from src.models.convolutional_dnn import ConvDNN
from src.data.make_dataset import load_data, join_data
from src.visualization.visualize import draw_tree

import numpy as np
from sklearn.utils import shuffle


def train_models():

    # Load the data
    data = load_data(dataset="FashionMNIST", already_downloaded=True)

    # Get the number of input features
    n_rows, n_cols = np.shape(data["x_train"])[1:]
    n_features = n_rows * n_cols
    n_classes = np.unique(data["y_train"]).shape[0]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Print some info
    for key in data.keys():
        x = data[key]
        print(key, " : ", np.shape(x))

    unique, counts = np.unique(data["y_train"], return_counts=True)
    print("y_train class count: ", dict(zip(unique, counts)))

    unique, counts = np.unique(data["y_valid"], return_counts=True)
    print("y_valid class count: ", dict(zip(unique, counts)))

    unique, counts = np.unique(data["y_test"], return_counts=True)
    print("y_test class count: ", dict(zip(unique, counts)))

    # --------------------------------------------------------------------------------
    # TRAIN THE VARIATIONAL AUTOENCODER TO FIT THE UNIT FUNCTION

    # Create VAE
    vae = VariationalAutoEncoder(
        name="vae_fashion_mnist",
        keras_verbose=True,
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
        stopping_patience=20
    )

    # Train VAE
    vae.train(data["x_train_flat"], data["x_valid_flat"], load_model=True)

    # Evaluate VAE
    vae_results = vae.evaluate(data["x_test_flat"])

    # Generate new data
    x_gen_flat = vae.sample(20000)

    # Reshape to images for CCN
    x_gen = np.array([np.reshape(x_gen_flat[i], [n_rows, n_cols]) for i in range(len(x_gen_flat))])

    # --------------------------------------------------------------------------------
    # TRAIN THE MULTI-LAYER PERCEPTRON TO FIT THE MAPPING FUNCTION

    # Create MLP
    mlp = MultiLayerPerceptron(
        name="mlp_fashion_mnist",
        num_inputs=n_features,
        num_outputs=n_classes,
        keras_verbose=True,
        ff_layers=[
            [512, "relu", 0.0, 0.2, True, "gaussian"],
            [512, "relu", 0.0, 0.2, True, "gaussian"],
            [512, "relu", 0.0, 0.2, True, "gaussian"],
            [512, "relu", 0.0, 0.2, True, "gaussian"]
        ],
        epochs=40,
        batch_size=64,
        stopping_patience=10
    )

    # Train MLP
    mlp.train(data["x_train_flat"], data["y_train_one_hot"], data["x_valid_flat"], data["y_valid_one_hot"], load_model=True)

    # Evaluate MLP
    mlp_results = mlp.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------
    # TRAIN A CNN TO FIT THE MAPPING FUNCTION

    # Create CNN
    cnn = ConvDNN(
        name="cnn_fashion_mnist",
        img_rows=n_rows,
        img_cols=n_cols,
        num_outputs=n_classes
    )

    # Train CNN
    cnn.train(data["x_train"], data["y_train_one_hot"], data["x_valid"], data["y_valid_one_hot"], load_model=True)

    # Evaluate CNN
    cnn_results = cnn.evaluate(data["x_test"], data["y_test_one_hot"])

    # Get CNN labels
    y_cnn_train = cnn.predict(data["x_train"])
    y_gen = cnn.predict(x_gen)

    x_both = join_data([data["x_train"], x_gen])
    x_both = x_both.reshape((x_both.shape[0], -1))

    y_both = join_data([y_cnn_train, y_gen])
    x_both, y_both = shuffle(x_both, y_both)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO FIT THE MAPPING FUNCTION

    # Create SDT
    sdt_raw = SoftBinaryDecisionTree(
        name="sdt_raw_fashion_mnist",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=4,
        keras_verbose=2,
        penalty_decay=0.5,
        inv_temp=0.01,
        ema_win_size=1000,
        penalty_strength=1e+1,
        batch_size=4,
        learning_rate=5e-03
    )

    # Train SDT RAW
    sdt_raw.train(data["x_train_flat"], data["y_train_one_hot"], data["x_valid_flat"], data["y_valid_one_hot"], load_model=True)

    # Evaluate SDT RAW
    sdt_raw_results = sdt_raw.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------
    # # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE MULTI-LAYER PERCEPTRON
    #
    # # Create SDT MLP
    # sdt_mlp = SoftBinaryDecisionTree(
    #     name="sdt_mlp",
    #     num_inputs=n_features,
    #     num_outputs=n_classes
    # )
    #
    # # Train SDT MLP
    # sdt_mlp.train(data["x_train"], y_mlp_train, data["x_valid"], data["y_valid_one_hot"], load_model=True)
    #
    # # Evaluate SDT MLP
    # sdt_mlp_results = sdt_mlp.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE CNN

    # Create SDT CNN
    sdt_cnn = SoftBinaryDecisionTree(
        name="sdt_cnn_fashion_mnist",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=4,
        keras_verbose=2,
        penalty_decay=0.5,
        inv_temp=0.01,
        ema_win_size=1000,
        penalty_strength=1e+1,
        batch_size=4,
        learning_rate=5e-03
    )

    # Train SDT CNN
    sdt_cnn.train(data["x_train_flat"], y_cnn_train, data["x_valid_flat"], data["y_valid_one_hot"], load_model=True)

    # Evaluate SDT CNN
    sdt_cnn_results = sdt_cnn.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE CNN WITH VAE

    # Create SDT VAE
    sdt_vae = SoftBinaryDecisionTree(
        name="sdt_cnn_vae_fashion_mnist",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=4,
        keras_verbose=2,
        penalty_decay=0.5,
        inv_temp=0.01,
        ema_win_size=1000,
        penalty_strength=1e+1,
        batch_size=4,
        learning_rate=5e-03
    )

    # Train SDT VAE
    sdt_vae.train(x_both, y_both, data["x_valid_flat"], data["y_valid_one_hot"], load_model=True)

    # Evaluate SDT VAE
    sdt_vae_results = sdt_vae.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------

    return vae_results, cnn_results, sdt_raw_results, sdt_cnn_results, sdt_vae_results


if __name__ == '__main__':

    train_models()
