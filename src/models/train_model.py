from .multi_layer_perceptron import MultiLayerPerceptron
from .soft_decision_tree import SoftBinaryDecisionTree
from .variational_autoencoder import VariationalAutoEncoder


def load_data(path: str, split_pcts: tuple):
    # TODO implement
    return 0, 0, 0, 0, 0, 0


def join(*args):
    # TODO implement
    return 0


def save_results(json_file, path):
    pass


def train_model(max_depth: int):

    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        load_data(path="path", split_pcts=(0.7, 0.2, 0.1))

    n_features = 784
    n_classes = 10

    # --------------------------------------------------------------------------------
    # TRAIN THE VARIATIONAL AUTOENCODER TO FIT THE UNIT FUNCTION

    # Create VAE
    vae = VariationalAutoEncoder(
        name="vae",
        latent_dimensions=8,
        epochs=200,
        encoding_layers={

        },
        decoding_layers={

        }
    )

    # Train VAE
    vae.train(x_train, x_valid)

    # Evaluate VAE
    vae_results = vae.evaluate(x_test)
    save_results(vae_results, "vae_results")

    # Save VAE
    vae.save("models")

    # Generate data
    x_gen = vae.sample(1000)

    # --------------------------------------------------------------------------------
    # TRAIN THE MULTI-LAYER PERCEPTRON TO FIT THE MAPPING FUNCTION

    # Create MLP
    mlp = MultiLayerPerceptron(
        name="mlp",
        epochs=200,
        layers={

        }
    )

    # Train MLP
    mlp.train(x_train, y_train, x_valid, y_valid)

    # Evaluate MLP
    mlp_results = mlp.evaluate(x_test, y_test)
    save_results(mlp_results, "mlp_results")

    # Save MLP
    mlp.save("models")

    # Get MLP labels
    y_mlp_train = mlp.predict(x_train)
    y_gen = mlp.predict(x_gen)

    x_both = join(x_train, x_gen)
    y_both = join(y_train, y_gen)

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO FIT THE MAPPING FUNCTION

    # Create SDT
    sdt_raw = SoftBinaryDecisionTree(
        name="sdt_raw",
        max_depth=max_depth,
        n_features=n_features,
        n_classes=n_classes
    )

    # Train SDT RAW
    sdt_raw.train(x_train, y_train)

    # Evaluate RAW
    sdt_raw_results = sdt_raw.evaluate(x_test, y_test)
    save_results(sdt_raw_results, "sdt_raw_results")

    # Save SDT RAW
    sdt_raw.save("models")

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE MULTI-LAYER PERCEPTRON

    # Create SDT MLP
    sdt_mlp = SoftBinaryDecisionTree(
        name="sdt_mlp"
    )

    # Train SDT MLP
    sdt_mlp.train(x_train, y_mlp_train)

    # Evaluate SDT MLP
    sdt_mlp_results = sdt_mlp.evaluate(x_test, y_test)
    save_results(sdt_mlp_results, "sdt_mlp_results")

    # Save SDT MLP
    sdt_mlp.save("models")

    # --------------------------------------------------------------------------------
    # TRAIN A SOFT DECISION TREE TO APPROXIMATE THE MULTI-LAYER PERCEPTRON WITH VAE

    # Create SDT VAE
    sdt_vae = SoftBinaryDecisionTree(
        name="sdt_mlp"
    )

    # Train SDT MLP
    sdt_vae.train(x_both, y_both)

    # Evaluate SDT MLP
    sdt_vae_results = sdt_vae.evaluate(x_test, y_test)
    save_results(sdt_vae_results, "sdt_vae_results")

    # Save SDT MLP
    sdt_vae.save("models")

    # --------------------------------------------------------------------------------

    return vae_results, mlp_results, sdt_raw_results, sdt_mlp_results, sdt_vae_results

