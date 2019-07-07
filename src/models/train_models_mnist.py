from multi_layer_perceptron import MultiLayerPerceptron
from soft_decision_tree import SoftBinaryDecisionTree
from variational_autoencoder import VariationalAutoEncoder
from convolutional_dnn import ConvDNN
from pathlib import Path
import os
import numpy as np
from sklearn.utils import shuffle


def load_data(dataset: str, already_downloaded=True):

    # Create an empty data dictionary that contains all the relevant data
    data_dict = {}
    if already_downloaded:
        project_dir = Path(__file__).resolve().parents[2]
        data_dir = str(project_dir) + '/data/' + dataset

        for f in os.listdir(data_dir):
            data_dict[f.replace(".npy", "")] = np.load(os.path.join(data_dir, f))

        return data_dict


def join_data(data: list) -> np.ndarray:
    new_data = data[0]
    for i in range(len(data) - 1):
        new_data = np.vstack((new_data, data[i + 1]))
    return new_data


def save_results(json_file, path):
    pass


def train_model():

    # Load the data
    data = load_data(dataset="MNIST")

    # Get the number of input features
    n_rows, n_cols = np.shape(data["x_train"])[1:]
    n_features = n_rows * n_cols
    n_classes = np.unique(data["y_train"]).shape[0]

    # for key in data:
    #     print(key, ": ", np.shape(data[key]))

    # # # --------------------------------------------------------------------------------
    # TRAIN THE VARIATIONAL AUTOENCODER TO FIT THE UNIT FUNCTION

    # Create VAE
    vae = VariationalAutoEncoder(
        name="vae",
        num_inputs=n_features
    )

    vae.load()

    # Train VAE
    # vae.train(data["x_train_flat"], data["x_valid_flat"])
    #
    # Save VAE weights
    # vae.save()

    # Evaluate VAE
    vae_results = vae.evaluate(data["x_test_flat"])
    # vae.plot_model()
    # quit(-1)
    # save_results(vae_results, "vae_results")

    # vae.load()
    x_gen_flat = vae.sample(20000)

    x_gen = np.array([np.reshape(x_gen_flat[i], [28, 28]) for i in range(len(x_gen_flat))])

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
    # x_test_encoded, _, _ = vae.predict(data["x_test_flat"])
    # f = plt.figure(figsize=(6, 6))
    # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=data["y_test"])
    # plt.colorbar()
    # plt.show()
    # plt.close()

    # # Generate data
    # x_gen = vae.sample(1000)
    #
    #
    # # --------------------------------------------------------------------------------
    # TRAIN THE MULTI-LAYER PERCEPTRON TO FIT THE MAPPING FUNCTION

    # Create MLP
    mlp = MultiLayerPerceptron(
        name="mlp",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # # Train MLP
    # mlp.train(data["x_train_flat"], data["y_train_one_hot"], data["x_valid_flat"], data["y_valid_one_hot"])
    #
    # # Evaluate MLP
    # mlp_results = mlp.evaluate(data["x_test_flat"], data["y_test_one_hot"])
    # # save_results(mlp_results, "mlp_results")
    #
    # # Save MLP
    # mlp.save()

    mlp.load()
    mlp_results = mlp.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # mlp.plot_model()

    # Get MLP labels
    # y_mlp_train = mlp.predict(data["x_train_flat"])
    # y_gen = mlp.predict(x_gen)

    # x_gen = [np.reshape(x_gen[i], [28, 28]) for i in range(len(x_gen))]
    #
    # import matplotlib.pyplot as plt
    #
    # for i, img in enumerate(x_gen):
    #     print(y_gen[i])
    #     plt.figure(figsize=(1, 1))
    #     plt.axis('off')
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
    #     plt.close()

    # x_both = join_data([data["x_train_flat"], x_gen])
    # y_both = join_data([y_mlp_train, y_gen])
    # x_both, y_both = shuffle(x_both, y_both)

    # y_both_single = np.argmax(y_both, axis=1)
    #
    # from collections import Counter
    #
    # print(Counter(y_both_single))
    # quit(-1)

    # --------------------------------------------------------------------------------
    # TRAIN A CNN

    cnn = ConvDNN(
        name="cnn",
        img_rows=n_rows,
        img_cols=n_cols,
        num_outputs=n_classes
    )

    cnn.load()

    # cnn.train(data["x_train"], data["y_train_one_hot"], data["x_valid"], data["y_valid_one_hot"])
    #
    # cnn.save()

    cnn.evaluate(data["x_test"], data["y_test_one_hot"])

    cnn.plot_model()
    quit(-1)

    y_mlp_train = cnn.predict(data["x_train"])
    y_gen = cnn.predict(x_gen)

    x_both = join_data([data["x_train"], x_gen])
    x_both = x_both.reshape((x_both.shape[0], -1))

    y_both = join_data([y_mlp_train, y_gen])
    x_both, y_both = shuffle(x_both, y_both)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO FIT THE MAPPING FUNCTION

    # Create SDT
    sdt_raw = SoftBinaryDecisionTree(
        name="sdt_raw",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train SDT RAW
    sdt_raw.load()
    # sdt_raw.train(data["x_train_flat"], data["y_train_one_hot"], data["x_valid_flat"], data["y_valid_one_hot"])

    # Evaluate RAW
    # sdt_raw_results = sdt_raw.evaluate(data["x_test"], data["y_test_one_hot"])
    # save_results(sdt_raw_results, "sdt_raw_results")
    # print(sdt_raw_results)

    # Save SDT RAW
    # sdt_raw.save()

    # sdt_raw.load()

    sdt_raw_results = sdt_raw.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE MULTI-LAYER PERCEPTRON

    # Create SDT MLP
    sdt_mlp = SoftBinaryDecisionTree(
        name="sdt_mlp",
        num_inputs=n_features,
        num_outputs=n_classes
    )
    #
    # # # Train SDT MLP
    # sdt_mlp.train(data["x_train"], y_mlp_train, data["x_valid"], data["y_valid_one_hot"])
    # #
    # # # Save SDT MLP
    # sdt_mlp.save()

    sdt_mlp.load()
    #
    # # Evaluate SDT MLP
    sdt_mlp_results = sdt_mlp.evaluate(data["x_test_flat"], data["y_test_one_hot"])
    # # save_results(sdt_mlp_results, "sdt_mlp_results")
    #
    # quit(-1)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE MULTI-LAYER PERCEPTRON

    # Create SDT MLP
    sdt_cnn = SoftBinaryDecisionTree(
        name="sdt_cnn",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # # Train SDT MLP
    # sdt_cnn.train(data["x_train_flat"], y_mlp_train, data["x_valid_flat"], data["y_valid_one_hot"])
    #
    # # Save SDT MLP
    # sdt_cnn.save()

    sdt_cnn.load()

    # # Evaluate SDT MLP
    sdt_cnn_results = sdt_cnn.evaluate(data["x_test_flat"], data["y_test_one_hot"])
    # # save_results(sdt_mlp_results, "sdt_mlp_results")
    #

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE MULTI-LAYER PERCEPTRON WITH VAE

    # Create SDT VAE
    sdt_vae = SoftBinaryDecisionTree(
        name="sdt_cnn_vae",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train SDT MLP
    sdt_vae.train(x_both, y_both, data["x_valid_flat"], data["y_valid_one_hot"])

    # Save SDT MLP
    # sdt_vae.save()

    # sdt_vae.load()

    # Evaluate SDT MLP
    sdt_vae_results = sdt_vae.evaluate(data["x_test_flat"], data["y_test_one_hot"])
    quit(-1)

    # save_results(sdt_vae_results, "sdt_vae_results")

    # --------------------------------------------------------------------------------

    return vae_results, mlp_results, sdt_raw_results, sdt_mlp_results, sdt_vae_results


if __name__ == '__main__':
    train_model()


