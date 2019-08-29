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
    data = load_data(dataset="MNIST", already_downloaded=True)

    # Get the number of input features
    n_rows, n_cols = np.shape(data["x_train"])[1:]
    n_features = n_rows * n_cols
    n_classes = np.unique(data["y_train"]).shape[0]

    # Print information on data
    for key in data.keys():
        x = data[key]
        print(key, " : ", np.shape(x))

    unique, counts = np.unique(data["y_train"], return_counts=True)
    print(dict(zip(unique, counts)))

    unique, counts = np.unique(data["y_valid"], return_counts=True)
    print(dict(zip(unique, counts)))

    unique, counts = np.unique(data["y_test"], return_counts=True)
    print(dict(zip(unique, counts)))

    quit(-1)

    # --------------------------------------------------------------------------------
    # TRAIN THE VARIATIONAL AUTOENCODER TO FIT THE UNIT FUNCTION

    # Create VAE
    vae = VariationalAutoEncoder(
        name="vae",
        num_inputs=n_features,
        epochs=1000
    )

    # Train VAE
    vae.train(data["x_train_flat"], data["x_valid_flat"], load_model=True)

    # Evaluate VAE
    vae_results = vae.evaluate(data["x_test_flat"])

    # Generate new data
    x_gen_flat = vae.sample(20000)

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

    # --------------------------------------------------------------------------------
    # TRAIN THE MULTI-LAYER PERCEPTRON TO FIT THE MAPPING FUNCTION

    # Create MLP
    mlp = MultiLayerPerceptron(
        name="mlp",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train MLP
    mlp.train(data["x_train_flat"], data["y_train_one_hot"], data["x_valid_flat"], data["y_valid_one_hot"], load_model=True)

    # Evaluate MLP
    mlp_results = mlp.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # Get MLP labels
    # y_mlp_train = mlp.predict(data["x_train_flat"])
    # y_gen = mlp.predict(x_gen)
    #
    # x_both = join_data([data["x_train_flat"], x_gen])
    # y_both = join_data([y_mlp_train, y_gen])
    # x_both, y_both = shuffle(x_both, y_both)

    # --------------------------------------------------------------------------------
    # TRAIN A CNN TO FIT THE MAPPING FUNCTION

    # Create CNN
    cnn = ConvDNN(
        name="cnn",
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

    y_gen_labels = np.argmax(y_gen, axis=1)

    unique, counts = np.unique(y_gen_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    quit(-1)

    x_both = join_data([data["x_train"], x_gen])
    x_both = x_both.reshape((x_both.shape[0], -1))

    y_both = join_data([y_cnn_train, y_gen])
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
    sdt_raw.train(data["x_train_flat"], data["y_train_one_hot"], data["x_valid_flat"], data["y_valid_one_hot"], load_model=True)

    # Evaluate SDT RAW
    sdt_raw_results = sdt_raw.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE CNN

    # Create SDT CNN
    sdt_cnn = SoftBinaryDecisionTree(
        name="sdt_cnn",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train SDT CNN
    sdt_cnn.train(data["x_train_flat"], y_cnn_train, data["x_valid_flat"], data["y_valid_one_hot"], load_model=True)

    # Evaluate SDT CNN
    sdt_cnn_results = sdt_cnn.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # digit = 9
    #
    # sample_index = np.random.choice(np.where(np.argmax(data["y_test_one_hot"], axis=1) == digit)[0])
    # input_img = data["x_test"][sample_index]
    #
    # draw_tree(sdt_cnn, n_rows, n_cols)
    # draw_tree(sdt_cnn, n_rows, n_cols, input_img=input_img)
    # draw_tree(sdt_cnn, n_rows, n_cols, input_img=input_img, show_correlation=True)
    #
    # quit(-1)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE CNN WITH VAE

    # Create SDT VAE
    sdt_vae = SoftBinaryDecisionTree(
        name="sdt_cnn_vae",
        num_inputs=n_features,
        num_outputs=n_classes
    )

    # Train SDT VAE
    sdt_vae.train(x_both, y_both, data["x_valid_flat"], data["y_valid_one_hot"], load_model=True)

    # Evaluate SDT VAE
    sdt_vae_results = sdt_vae.evaluate(data["x_test_flat"], data["y_test_one_hot"])

    # --------------------------------------------------------------------------------

    return vae_results, cnn_results, sdt_raw_results, sdt_cnn_results, sdt_vae_results


if __name__ == '__main__':

    train_models()


