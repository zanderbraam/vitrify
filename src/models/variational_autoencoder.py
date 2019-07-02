import numpy as np
from pathlib import Path
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input, GaussianDropout, Lambda
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

import src.utils as utils


class VariationalAutoEncoder:
    """
    An implementation of a variational autoencoder.
    """
    def __init__(self, name: str, num_inputs: int, *args, **kwargs):
        """
        :param name: the user specified name given for the neural network
        :param num_inputs: the number of inputs that goes into the model
        :param num_outputs: the number of possible classes
        """
        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_inputs

        # If true, training info is outputted to stdout
        self.keras_verbose = True
        # A summary of the NN is printed to stdout
        self.print_model_summary = True

        # Size of latent dimension
        self.latent_dim = 20

        # Specify the encoder layers [units, activation, dropout, l2, bias]
        self.encoder_layers = [
            [512, "relu", 0.0, 0.0, True, "gaussian"],
            [256, "relu", 0.0, 0.0, True, "gaussian"],
            [128, "relu", 0.0, 0.0, True, "gaussian"]
        ]

        # Specify the mean and standard deviation layers respectively [units, mean/std.dev, bias]
        self.latent_layers = [
            [self.latent_dim, 0.0, False],
            [self.latent_dim, 1.0, False]
        ]

        # Specify the decoder layers [units, activation, dropout, l2, bias]
        self.decoder_layers = [
            [128, "relu", 0.0, 0.0, True, "gaussian"],
            [256, "relu", 0.0, 0.0, True, "gaussian"],
            [512, "relu", 0.0, 0.0, True, "gaussian"]
        ]

        # The final output layer's activation function
        self.final_activation = "sigmoid"
        # The maximum number of epochs to run
        self.epochs = 100
        # The batch size to use in the NN
        self.batch_size = 250
        # The learning rate used in optimization
        self.learning_rate = 0.001
        # The optimization algorithm to use
        self.optimizer = Adam(lr=self.learning_rate)
        # If this many stagnant epochs are seen, stop training
        self.stopping_patience = 10

        # --------------------------------------------------------------------------------

        # Create the input
        input = [
            Input(
                shape=(num_inputs,)
            )
        ]

        # Create the layers
        layers = []
        layers.extend(input)

        # Build up the layers
        for j in range(len(self.encoder_layers)):

            # --- Add the Encoding layers ---

            layers.append(
                Dense(
                    units=self.encoder_layers[j][0],
                    activation=self.encoder_layers[j][1],
                    kernel_regularizer=l2(self.encoder_layers[j][2]),
                    use_bias=self.encoder_layers[j][4],
                    name="encoder_" + str(j)
                )(layers[-1])
            )

            # --- Add the Dropout layers (normal or gaussian) to the model ---

            if self.encoder_layers[j][5] == "normal":
                layers.append(
                    Dropout(
                        self.encoder_layers[j][3]
                    )(layers[-1])
                )
            elif self.encoder_layers[j][5] == "gaussian":
                layers.append(
                    GaussianDropout(
                        self.encoder_layers[j][3]
                    )(layers[-1])
                )

        # --- Add the Gaussian sampling layer ---

        # Create mean layer
        mean_layer = Dense(
            units=self.latent_layers[0][0],
            use_bias=self.latent_layers[0][2],
            name="mean_latent_layer",
            activation='linear'
        )(layers[-1])

        # Create standard deviation layer
        std_layer = Dense(
            units=self.latent_layers[1][0],
            use_bias=self.latent_layers[1][2],
            name="std_latent_layer",
            activation='linear'
        )(layers[-1])

        # Create sampling function
        def sampling_1d(iargs):
            mean, std = iargs
            epsilon = K.random_normal(
                shape=(
                    K.shape(mean)[0],
                    self.latent_layers[0][0]
                ),
                mean=self.latent_layers[0][1],
                stddev=self.latent_layers[1][1]
            )
            return mean + K.exp(std / 2.) * epsilon

        # Add the Lambda layer
        layers.append(
            Lambda(
                sampling_1d, name="latent_mixing_layer"
            )([mean_layer, std_layer])
        )

        # Create encoder model
        encoder_model = Model(input, [mean_layer, std_layer, layers[-1]], name="encoder")

        # Add input layer, used to separate the decoder model
        latent_inputs = Input(shape=(self.latent_layers[0][0],), name="z_sampling")
        layers.append(latent_inputs)

        # --- Add the decoding layers - reverse the encoding process ---

        # Add decoder layers
        for j in range(len(self.decoder_layers)):
            # Add the dropout layer to the inputs
            layers.append(
                Dense(
                    units=self.decoder_layers[j][0],
                    activation=self.decoder_layers[j][1],
                    kernel_regularizer=l2(self.decoder_layers[j][2]),
                    use_bias=self.decoder_layers[j][4],
                    name="decoder_" + str(j)
                )(layers[-1])
            )

            # --- Add the Dropout layers (normal or gaussian) to the model ---

            if self.encoder_layers[j][5] == "normal":
                layers.append(
                    Dropout(
                        self.encoder_layers[j][3]
                    )(layers[-1])
                )
            elif self.encoder_layers[j][5] == "gaussian":
                layers.append(
                    GaussianDropout(
                        self.encoder_layers[j][3]
                    )(layers[-1])
                )

        # Add the final dense layer
        output = Dense(
            units=self.num_outputs,
            activation=self.final_activation
        )(layers[-1])

        # Create decoder model
        decoder_model = Model(latent_inputs, output, name="decoder")

        # Create the neural network model and compile it
        output = decoder_model(encoder_model(input)[2])
        vae = Model(inputs=input, outputs=output, name=self.name)

        # Specifying loss
        reconstruction_loss = binary_crossentropy(layers[0], output)
        reconstruction_loss *= num_inputs
        kl_loss = 1 + std_layer - K.square(mean_layer) - K.exp(std_layer)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        # Add the loss to the model
        vae.add_loss(vae_loss)

        # Finally compile the variational autoencoder so we can use it with the weights going forward
        vae.compile(optimizer=self.optimizer, metrics=["acc"])

        # Assign vae to self.model
        self.model = vae
        self.encoder = encoder_model
        self.decoder = decoder_model
        if self.print_model_summary:
            print("")
            self.encoder.summary()
            print("")
            self.decoder.summary()
            print("")
            self.model.summary()

    def train(self, x_train: np.ndarray, x_valid: np.ndarray) -> None:

        # What we want back from Keras
        callbacks_list = []

        # The default patience is stopping_patience
        patience = self.stopping_patience

        # Create an early stopping callback and add it
        callbacks_list.append(
            EarlyStopping(
                verbose=1,
                monitor='val_loss',
                patience=self.stopping_patience))

        # TODO: Double check parameter patience and monitor!

        # Train the model
        training_process = self.model.fit(
            x=x_train,
            epochs=self.epochs,
            shuffle=True,
            batch_size=self.batch_size,
            verbose=self.keras_verbose,
            callbacks=callbacks_list,
            validation_data=(x_valid, None)
        )

    def evaluate(self, x_test: np.ndarray) -> tuple:
        score = self.model.evaluate(x_test, verbose=0)
        print("loss: {}".format(score))
        return score

    def save(self):
        project_dir = Path(__file__).resolve().parents[2]
        models_dir = str(project_dir) + '/models/' + self.name + '/'
        utils.check_folder(models_dir)
        self.model.save_weights(models_dir + self.name + ".h5")
        print("Saved model to disk")

    def load(self):
        try:
            project_dir = Path(__file__).resolve().parents[2]
            models_dir = str(project_dir) + '/models/' + self.name + '/'
            utils.check_folder(models_dir)
            self.model.load_weights(models_dir + self.name + ".h5")
            print("Loaded " + self.name + " model from disk")

        except ValueError as e:
            print("No saved model found. Check file name or train from scratch")

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.encoder.predict(x, batch_size=self.batch_size)
        return prediction

    def sample(self, n: int):
        randoms = [np.random.normal(0, 1, self.latent_dim) for _ in range(n)]
        imgs = self.decoder.predict(np.array([randoms]).squeeze())
        return imgs
