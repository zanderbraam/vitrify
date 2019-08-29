import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import backend as tfk
from tensorflow.keras.initializers import RandomNormal, TruncatedNormal
from tensorflow.keras.layers import Input, Dense, Activation, Layer, Lambda, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.backend import set_session

import src.utils as utils


class TrainableVar(Layer):
    """
    Creates a variable that's trainable with keras model. Needs to be attached
    to some node that conditions optimizer op.
    """

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, name, shape, **kwargs):
        super(TrainableVar, self).__init__()
        self.kernel = self.add_variable(name=name, shape=shape, **kwargs)

    def call(self, input, *args, **kwargs):
        return self.kernel
    

class LeftBranch(Layer):
    def compute_output_shape(self, input_shape):
        pass

    def call(self, input, *args, **kwargs):
        return input[0] * (1 - input[1])
    

class RightBranch(Layer):
    def compute_output_shape(self, input_shape):
        pass

    def call(self, input, *args, **kwargs):
        return input[0] * input[1]
    

class Scale(Layer):
    def compute_output_shape(self, input_shape):
        pass

    def call(self, input, *args, **kwargs):
        return input[0] * input[1]


class Constant(Layer):
    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, value=1, **kwargs):
        self.value = value
        super(Constant, self).__init__(**kwargs)

    def call(self, input, *args, **kwargs):
        return tfk.constant(self.value, shape=(1,), dtype="float32")


class OutputLayer(Layer):
    def compute_output_shape(self, input_shape):
        pass

    def call(self, input, *args, **kwargs):
        opinions, weights = input
        opinions = Concatenate(axis=0)(opinions)  # shape=(n_bigots, n_classes)
        weights = Concatenate(axis=1)(weights)  # shape=(batch_size, n_bigots)
        elems = tfk.argmax(weights, axis=1)  # shape=(batch_size,)

        def from_keras_tensor(opinions, elems=None):
            return tfk.map_fn(lambda x: opinions[x], elems, dtype=tf.float32)

        outputs = Lambda(
            from_keras_tensor, arguments={"elems": elems})(opinions)

        return outputs  # shape=(batch_size, n_classes)

# --------------------------------------------------------------------------------
    

class Node(object):
    def __init__(self, id, depth, path_prob, tree):
        self.id = id
        self.depth = depth
        self.path_prob = path_prob
        self.is_leaf = self.depth == tree.max_depth
        self.left_child = None
        self.right_child = None

        self.prob = None

        self.dense_scaled = None

        self.ema = None
        self.ema_apply_op = None
        self.ema_P = None
        self.ema_p = None
        self.alpha = None
        self.penalty = None

        self.leaf_loss = None

        if self.is_leaf:
            self.phi = TrainableVar(
                name="phi_" + self.id, shape=(1, tree.n_classes),
                dtype="float32", initializer=TruncatedNormal())(path_prob)
        else:
            self.dense = Dense(
                units=1, name="dense_" + self.id, dtype="float32",
                kernel_initializer=RandomNormal(),
                bias_initializer=TruncatedNormal())(tree.input_layer)

    def build(self, tree):
        """
        Defines the output probability of the node and builds child nodes.
        """
        self.prob = self.forward(tree)
        if not self.is_leaf:
            left_prob = LeftBranch()([self.path_prob, self.prob])
            right_prob = RightBranch()([self.path_prob, self.prob])
            self.left_child = Node(id=self.id + "0", depth=self.depth + 1,
                                   path_prob=left_prob, tree=tree)
            self.right_child = Node(id=self.id + "1", depth=self.depth + 1,
                                    path_prob=right_prob, tree=tree)

    def forward(self, tree):
        """
        Defines the output probability.
        """
        if not self.is_leaf:
            self.dense_scaled = Scale()([tree.inv_temp, self.dense])
            return Activation("sigmoid", name="prob_" + self.id)(
                self.dense_scaled)
        else:
            return Activation("softmax", name="pdist_" + self.id)(self.phi)

    def get_penalty(self, tree):
        """
        From paper: "... we can maintain an exponentially decaying running
        average of the actual probabilities with a time window that is
        exponentially proportional to the depth of the node."
        So here we track EMAs of batches of P^i and p_i and calculate:
            alpha = sum(ema(P^i) * ema(p_i)) / sum(ema(P^i))
            penalty = -0.5 * (log(alpha) + log(1-alpha))
        """
        # Keep track of running average of probabilities (batch-wise)
        # with exponential growth of time window w.r.t. the  depth of the node
        self.ema = tf.train.ExponentialMovingAverage(
            decay=0.9999, num_updates=tree.ema_win_size * 2 ** self.depth)
        self.ema_apply_op = self.ema.apply([self.path_prob, self.prob])
        self.ema_P = self.ema.average(self.path_prob)
        self.ema_p = self.ema.average(self.prob)

        # Calculate alpha by summing probabilities and path probabilities over batch
        self.alpha = (tf.reduce_sum(self.ema_P * self.ema_p) + tree.eps) / (
            tf.reduce_sum(self.ema_P) + tree.eps)

        # Calculate penalty for this node using running average of alpha
        self.penalty = (- 0.5 * tf.log(self.alpha + tree.eps)
                        - 0.5 * tf.log(1. - self.alpha + tree.eps))

        # Replace possible NaN values with zeros
        self.penalty = tf.where(
            tf.is_nan(self.penalty), tf.zeros_like(self.penalty), self.penalty)

        return self.penalty

    def get_loss(self, y, tree):
        if self.is_leaf:
            # Cross-entropies (batch) of soft labels with output of this leaf
            leaf_ce = - tf.reduce_sum(y * tf.log(tree.eps + self.prob), axis=1)
            # Mean of cross-entropies weighted by path probability of this leaf
            self.leaf_loss = tf.reduce_mean(self.path_prob * leaf_ce)
            # Return leaf contribution to the loss
            return self.leaf_loss
        else:
            # Return decayed penalty term of this (inner) node
            return tree.penalty_strength * self.get_penalty(tree) * (
                tree.penalty_decay ** self.depth)


# --------------------------------------------------------------------------------


class SoftBinaryDecisionTree(object):

    def __init__(self, name: str, num_inputs: int, num_outputs: int, *args, **kwargs):
        """
        Initialize model instance by saving parameter values
        as model properties and creating others as placeholders.
        """

        self.name = name

        self.n_features = num_inputs
        self.n_classes = num_outputs

        # If true, training info is outputted to stdout
        self.keras_verbose = False
        # A summary of the NN is printed to stdout
        self.print_model_summary = False

        # Hyperparameters
        self.max_depth = 4
        self.penalty_strength = 1e+1
        self.penalty_decay = 0.25
        self.inv_temp = 0.01
        self.ema_win_size = 1000

        self.learning_rate = 5e-03
        self.epochs = 40
        self.batch_size = 4

        # If this many stagnant epochs are seen, stop training
        self.stopping_patience = 20

        self.__dict__.update(kwargs)

        # --------------------------------------------------------------------------------

        def brand_new_tfsession(sess=None):

            if sess:
                tf.reset_default_graph()
                sess.close()

            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            set_session(sess)

            return sess

        self.sess = brand_new_tfsession()

        # --------------------------------------------------------------------------------

        # Internal variables: Do not alter!
        self.optimizer = Adam(lr=self.learning_rate)
        self.metrics = ["acc"]

        self.nodes = list()
        self.bigot_opinions = list()
        self.bigot_weights = list()
        self.ema_apply_ops = list()

        self.loss = 0.0
        self.loss_leaves = 0.0
        self.loss_penalty = 0.0

        self.eps = tfk.constant(1e-8, shape=(1,), dtype="float32")
        self.initialized = False

        self.input_layer = None
        self.root = None
        self.output_layer = None
        self.model = None
        self.saver = None

    def build_model(self):
        self.input_layer = Input(shape=(self.n_features,), dtype="float32")

        if self.inv_temp:
            self.inv_temp = Constant(value=self.inv_temp)(self.input_layer)
        else:
            self.inv_temp = TrainableVar(
                name="beta", shape=(1,), dtype="float32",
                initializer=RandomNormal())(self.input_layer)

        self.root = Node(id="0", depth=0, path_prob=Constant()(self.input_layer), tree=self)
        self.nodes.append(self.root)

        for node in self.nodes:
            node.build(tree=self)
            if node.is_leaf:
                self.bigot_opinions.append(node.prob)
                self.bigot_weights.append(node.path_prob)
            else:
                self.nodes.append(node.left_child)
                self.nodes.append(node.right_child)

        def tree_loss(y_true, y_pred):
            for node in self.nodes:
                if node.is_leaf:
                    self.loss_leaves += node.get_loss(y=y_true, tree=self)
                else:
                    self.loss_penalty += node.get_loss(y=None, tree=self)
                    self.ema_apply_ops.append(node.ema_apply_op)

            with tf.control_dependencies(self.ema_apply_ops):
                self.loss = tf.log(
                    self.eps + self.loss_leaves) + self.loss_penalty
            return self.loss

        self.output_layer = OutputLayer()([self.bigot_opinions, self.bigot_weights])

        print("Built tree has {} leaves out of {} nodes".format(
            sum([node.is_leaf for node in self.nodes]), len(self.nodes)))

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)

        self.model.compile(optimizer=self.optimizer, loss=tree_loss, metrics=self.metrics)

        if self.print_model_summary:
            print("")
            self.model.summary()

        self.saver = tf.train.Saver()

    def initialize_variables(self, sess, x, batch_size):
        """
        Since tf.ExponentialMovingAverage generates variables that
        depend on other variables being initialized first, we need to
        perform customized, 2-step initialization.
        Importantly, initialization of EMA variables also requires
        a single input batch of size that will be used for evaluation
        of loss, in order to create compatible shapes. Therefore,
        model will be constrained to initial batch size.
        """

        if not self.model:
            print("Missing model instance.")
            return

        ema_vars = [v for v in tf.global_variables() if
                    "ExponentialMovingAverage" in v.name and
                    "Const" not in v.name]
        independent_vars = [v for v in tf.global_variables() if
                            v not in ema_vars]
        feed_dict = {self.input_layer: x[:batch_size]}

        init_indep_vars_op = tf.variables_initializer(independent_vars)
        init_ema_vars_op = tf.variables_initializer(ema_vars)

        sess.run(init_indep_vars_op)
        sess.run(init_ema_vars_op, feed_dict=feed_dict)
        self.initialized = True

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, load_model=True) -> None:

        if load_model:
            self.load()

        else:
            # Build the model
            self.build_model()

            # Initialize variables
            self.initialize_variables(self.sess, x_train, self.batch_size)

            # What we want back from Keras
            callbacks_list = []

            # The default patience is stopping_patience
            patience = self.stopping_patience

            # Create an early stopping callback and add it
            callbacks_list.append(
                EarlyStopping(
                    verbose=1,
                    monitor='val_acc',
                    patience=patience))

            # Train the model
            training_process = self.model.fit(
                x=x_train,
                y=y_train,
                epochs=self.epochs,
                shuffle=True,
                batch_size=self.batch_size,
                callbacks=callbacks_list,
                validation_data=(x_valid, y_valid),
                verbose=self.keras_verbose
            )

            self.save()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
        if self.model and self.initialized:
            score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=0)
            test_loss = score[0]
            test_accuracy = score[1]
            print("accuracy: {:.2f}% | loss: {}".format(100 * test_accuracy, test_loss))
            return test_loss, test_accuracy
        else:
            print("Missing initialized model instance.")

    def save(self):
        """
        Keras saving methods such as model.save() or model.save_weights()
        are not suitable, since Keras won"t serialize tf.Tensor objects which
        get included into saving process as arguments of Lambda layers.
        """
        project_dir = Path(__file__).resolve().parents[2]
        models_dir = str(project_dir) + '/models/' + self.name + '/'
        utils.check_folder(models_dir)
        self.saver.save(self.sess, models_dir + self.name)
        print("Saved model to disk")

    def load(self):
        try:
            # Build the model
            self.build_model()

            project_dir = Path(__file__).resolve().parents[2]
            models_dir = str(project_dir) + '/models/' + self.name + '/'
            utils.check_folder(models_dir)
            self.saver.restore(self.sess, models_dir + self.name)
            self.initialized = True
            print("Loaded " + self.name + " model from disk")

        except ValueError as e:
            print("No saved model found. Check file name or train from scratch")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model and self.initialized:
            return self.model.predict(x, verbose=1)
        else:
            print("Missing initialized model instance.")
