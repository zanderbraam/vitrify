import numpy as np
from pathlib import Path
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input, GaussianDropout, Conv2D, MaxPool2D, Flatten, Reshape
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

import keras.backend as K

import src.utils as utils


class ConvDNN:
    """
    An implementation of a deep neural networks.
    """
    def __init__(self, name: str, img_rows: int, img_cols: int, num_outputs: int, *args, **kwargs):
        """
        :param name: the user specified name given for the neural network
        :param num_inputs: the number of inputs that goes into the model
        :param num_outputs: the number of possible classes
        """
        self.name = name
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_outputs = num_outputs

        # If true, training info is outputted to stdout
        self.keras_verbose = False
        # A summary of the NN is printed to stdout
        self.print_model_summary = False

        # Convolutional layers
        self.conv_layers = [
            # Layer, filters, kernel size, strides, padding, activation, use bias, kernel reg, bias reg, activity reg,
            # dropout (after layers)
            ["conv2d", 32, (3, 3), (1, 1), "valid", "relu", True, 0.0, 0.0, 0.0, 0.0],
            ["conv2d", 64, (3, 3), (1, 1), "valid", "relu", True, 0.0, 0.0, 0.0, 0.0],
            # Layer, pool size, stride, padding, dropout
            ["max_pool2d", (2, 2), None, "valid", 0.25],
        ]

        # ff_layers = [units, activation, regularization, dropout, use_bias]
        self.ff_layers = [
            [128, "relu", 0.0, 0.2, True, "normal"]
        ]

        # The final output layer's activation function
        self.final_activation = "softmax"
        # The objective function for the NN
        self.objective = "categorical_crossentropy"
        # The maximum number of epochs to run
        self.epochs = 20
        # The batch size to use in the NN
        self.batch_size = 128
        # The learning rate used in optimization
        self.learning_rate = 0.001
        # The optimization algorithm to use
        self.optimizer = Adam(lr=self.learning_rate)
        # If this many stagnant epochs are seen, stop training
        self.stopping_patience = 20

        # --------------------------------------------------------------------------------

        # Create the input
        input = [
            Input(
                shape=(self.img_rows, self.img_cols)
            )
        ]

        # Create the layers
        layers = []
        layers.extend(input)

        # Add channel dimension (required by conv layers)
        if K.image_data_format() == 'channels_last':
            layers.append(
                Reshape((self.img_rows, self.img_cols, 1))(layers[-1])
            )
        else:
            layers.append(
                Reshape((1, self.img_rows, self.img_cols))(layers[-1])
            )

        # Build up the layers
        for j in range(len(self.conv_layers)):

            # Get the convolutional layer type
            layer_type = self.conv_layers[j][0]

            if layer_type == "conv2d":
                
                # --- Add a 1 dimensional convolution ---
            
                layers.append(
                    Conv2D(
                        filters=self.conv_layers[j][1],
                        kernel_size=self.conv_layers[j][2],
                        strides=self.conv_layers[j][3],
                        padding=self.conv_layers[j][4],
                        activation=self.conv_layers[j][5],
                        use_bias=self.conv_layers[j][6],
                        kernel_regularizer=l2(self.conv_layers[j][7]),
                        bias_regularizer=l2(self.conv_layers[j][8]),
                        activity_regularizer=l2(self.conv_layers[j][9]),
                        name="convolutional_" + str(j)
                    )(layers[-1])
                )

                layers.append(
                    Dropout(
                        rate=self.conv_layers[j][10]
                    )(layers[-1])
                )

            elif layer_type == "max_pool2d":
                # Add a max pooling layer
                layers.append(
                    MaxPool2D(
                        pool_size=self.conv_layers[j][1],
                        strides=self.conv_layers[j][2],
                        padding=self.conv_layers[j][3],
                        name="max_pooling_" + str(j)
                    )(layers[-1])
                )

                layers.append(
                    Dropout(
                        rate=self.conv_layers[j][4]
                    )(layers[-1])
                )

            else:
                raise AttributeError(
                    "Invalid convolutional layer type")

        # --- Add a flatten Layer so we can build a FFNN ontop of the convolutional layers ---

        layers.append(
            Flatten()(layers[-1])
        )

        for j in range(len(self.ff_layers)):

            # --- Add the Dense layers (fully connected) to the model ---

            layers.append(
                Dense(
                    units=self.ff_layers[j][0],
                    activation=self.ff_layers[j][1],
                    kernel_regularizer=l2(self.ff_layers[j][2]),
                    use_bias=self.ff_layers[j][4]
                )(layers[-1])
            )

            # --- Add the Dropout layers (normal or gaussian) to the model ---

            if self.ff_layers[j][5] == "normal":
                layers.append(
                    Dropout(
                        self.ff_layers[0][3]
                    )(layers[-1])
                )
            elif self.ff_layers[j][5] == "gaussian":
                layers.append(
                    GaussianDropout(
                        self.ff_layers[0][3]
                    )(layers[-1])
                )

        # Add the final dense layer
        output = Dense(
            units=self.num_outputs,
            activation=self.final_activation
        )(layers[-1])

        # Create the neural network model and compile it
        cnn = Model(inputs=input, outputs=output, name=self.name)

        # Finally compile the neural network so we can use it with the weights going forward
        cnn.compile(optimizer=self.optimizer, loss=self.objective, metrics=["accuracy"])

        # Assign cnn to self.model
        self.model = cnn
        if self.print_model_summary:
            print("")
            self.model.summary()

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> None:

        # What we want back from Keras
        callbacks_list = []

        # The default patience is stopping_patience
        patience = self.stopping_patience

        # Create an early stopping callback and add it
        callbacks_list.append(
            EarlyStopping(
                verbose=1,
                monitor="val_acc",
                patience=patience))

        # Train the model
        training_process = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.keras_verbose,
            callbacks=callbacks_list,
            validation_data=(x_valid, y_valid)
        )

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
        score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)
        test_loss = score[0]
        test_accuracy = score[1]
        print("accuracy: {:.2f}% | loss: {}".format(100 * test_accuracy, test_loss))
        return test_loss, test_accuracy

    def save(self):
        project_dir = Path(__file__).resolve().parents[2]
        models_dir = str(project_dir) + "/models/" + self.name + "/"
        utils.check_folder(models_dir)
        self.model.save(models_dir + self.name + ".h5")
        print("Saved model to disk")

    def load(self):
        try:
            project_dir = Path(__file__).resolve().parents[2]
            models_dir = str(project_dir) + "/models/" + self.name + "/"
            utils.check_folder(models_dir)
            self.model = load_model(models_dir + self.name + ".h5")
            print("Loaded " + self.name + " model from disk")

        except ValueError as e:
            print("No saved model found. Check file name or train from scratch")

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(x)
        return prediction
