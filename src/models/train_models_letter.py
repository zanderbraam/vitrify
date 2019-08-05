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
    data = load_data(dataset="Letter", already_downloaded=True)

    # Get the number of input features
    n_features = np.shape(data["x_train"])[1]
    n_classes = np.unique(data["y_train"]).shape[0]

    for key in data.keys():
        x = data[key]
        print(key, " : ", np.shape(x))

    # --------------------------------------------------------------------------------
    # TRAIN THE VARIATIONAL AUTOENCODER TO FIT THE UNIT FUNCTION

    # Create VAE
    vae = VariationalAutoEncoder(
        name="vae_letter",
        num_inputs=n_features
    )

    # Train VAE
    vae.train(data["x_train"], data["x_valid"], load_model=True)

    # Evaluate VAE
    vae_results = vae.evaluate(data["x_test"])

    # Generate new data
    x_gen = vae.sample(20000)

    # # --------------------------------------------------------------------------------
    # TRAIN THE MULTI-LAYER PERCEPTRON TO FIT THE MAPPING FUNCTION

    # Create MLP
    mlp = MultiLayerPerceptron(
        name="mlp_letter",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train MLP
    mlp.train(data["x_train"], data["y_train_one_hot"], data["x_valid"], data["y_valid_one_hot"], load_model=True)

    # Evaluate MLP
    mlp_results = mlp.evaluate(data["x_test"], data["y_test_one_hot"])

    # Get MLP labels
    y_mlp_train = mlp.predict(data["x_train"])
    y_gen = mlp.predict(x_gen)

    x_both = join_data([data["x_train"], x_gen])
    y_both = join_data([y_mlp_train, y_gen])
    x_both, y_both = shuffle(x_both, y_both)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO FIT THE MAPPING FUNCTION

    # Create SDT
    sdt_raw = SoftBinaryDecisionTree(
        name="sdt_raw_letter",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=9
    )

    # Train SDT RAW
    sdt_raw.train(data["x_train"], data["y_train_one_hot"], data["x_valid"], data["y_valid_one_hot"], load_model=False)

    # Evaluate SDT RAW
    sdt_raw_results = sdt_raw.evaluate(data["x_test"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE MULTI-LAYER PERCEPTRON

    # Create SDT MLP
    sdt_mlp = SoftBinaryDecisionTree(
        name="sdt_mlp_letter",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=9
    )

    # Train SDT MLP
    sdt_mlp.train(data["x_train"], y_mlp_train, data["x_valid"], data["y_valid_one_hot"], load_model=False)

    # Evaluate SDT MLP
    sdt_mlp_results = sdt_mlp.evaluate(data["x_test"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE MLP WITH VAE

    # Create SDT VAE
    sdt_vae = SoftBinaryDecisionTree(
        name="sdt_mlp_vae_letter",
        num_inputs=n_features,
        num_outputs=n_classes,
        max_depth=9
    )

    # Train SDT MLP
    sdt_vae.train(x_both, y_both, data["x_valid"], data["y_valid_one_hot"], load_model=False)

    # Evaluate SDT MLP
    sdt_vae_results = sdt_vae.evaluate(data["x_test"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------

    return vae_results, mlp_results, sdt_raw_results, sdt_mlp_results, sdt_vae_results


if __name__ == '__main__':

    train_models()


