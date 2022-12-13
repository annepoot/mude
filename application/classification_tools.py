from sklearn.utils.extmath import safe_sparse_dot
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.utils.multiclass import _check_partial_fit_first_call, type_of_target
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import numpy as np

class manualMLPClassifier(MLPClassifier):

    def partial_fit(self, X, y, classes=None):
        """Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        classes : array of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        Returns
        -------
        self : object
            Trained MLP model.
        """
        if _check_partial_fit_first_call(self, classes):
            self._label_binarizer = LabelBinarizer()
            if type_of_target(y).startswith("multilabel"):
                self._label_binarizer.fit(y)
            else:
                self._label_binarizer.fit(classes)

        super().partial_fit(X, y)

        return self

    def _forward_pass_fast(self, X):
        """
        The forward pass is overwritten, to have access to the states of neurons during training
        """
        X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)

        # Initialize first layer
        activation = X

        # Forward propagate
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            if i == self.n_layers_ - 2:
                self.lasthidlayer = activation
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)

        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)

        return activation

    def _latent_pass_fast(self, X):
        """
        A forward pass from the final layer to the output is created
        """
        X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)

        # Forward propagate through final layer
        activation = safe_sparse_dot(X, self.coefs_[self.n_layers_ - 2])
        activation += self.intercepts_[self.n_layers_ - 2]

        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)

        return activation

    def predict_raw(self, X):
        y_pred = self._forward_pass_fast(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        return y_pred


# function to train the NN
def NN_train_visualize(NN, X_train, y_train, X_val, y_val, classes, epochs=10000, verbose=True, lr_init=1e-3, latent_states_train=None, latent_states_val=None, latent_state_out=None, latent_grid_points = None, PCA_state_out = None, PCA_grid_points = None, transformed_latent_grid = None):

    validation_score = np.empty(epochs)
    training_score = np.empty(epochs)

    # set learning rate
    lr = lr_init
    NN.learning_rate_init = lr


    # loop over iterations
    for epoch in range(epochs):
        # train for one epoch, compute rmse on validation set
        NN.partial_fit(X_train, y_train, classes=classes)

        # Compute training and prediction scores
        training_score[epoch] = NN.score(X_train, y_train)
        validation_score[epoch] = NN.score(X_val, y_val)

        # print(latent_states)
        train_out = NN.predict(X_train)     # TODO predict raw
        NN.predict_raw(X_train)
        training_score[epoch] = accuracy_score(y_train, train_out)
        latent_states_train[epoch, :, :] = NN.lasthidlayer  # Store latent space last hidden layer

        # Compute boundaries of latent grid
        x_latent_coords = NN.lasthidlayer[:,0]
        y_latent_coords = NN.lasthidlayer[:,1]

        val_out = NN.predict(X_val)       # TODO predict raw
        NN.predict_raw(X_val)
        validation_score[epoch] = accuracy_score(y_val, val_out)

        latent_states_val[epoch, :, :] = NN.lasthidlayer  # Store latent space coordinates last hidden layer

        # Grid in PCA space
        out = NN.predict_raw(PCA_grid_points)   # Non-binarized, score instead
        PCA_state_out[epoch, :] = out  # Store output

        cur_latent_bounds = np.array([[min(x_latent_coords), max(x_latent_coords)],[min(y_latent_coords), max(y_latent_coords)]])

        # Grid in latent space
        scaled_latent_grid = np.empty_like(latent_grid_points)
        scaled_latent_grid[:,0] = latent_grid_points[:,0] * (cur_latent_bounds[0, 1] - cur_latent_bounds[0, 0]) * 1.2 + cur_latent_bounds[0, 0] - (cur_latent_bounds[0, 1] - cur_latent_bounds[0, 0]) * 0.1
        scaled_latent_grid[:,1] = latent_grid_points[:,1] * (cur_latent_bounds[1, 1] - cur_latent_bounds[1, 0]) * 1.2 + cur_latent_bounds[1, 0] - (cur_latent_bounds[1, 1] - cur_latent_bounds[1, 0]) * 0.1

        transformed_latent_grid[epoch, :, :] = scaled_latent_grid

        out = NN._latent_pass_fast(scaled_latent_grid)   # Non-binarized, score instead
        latent_state_out[epoch, :] = out.ravel()  #.reshape(len(latent_state_out[epoch, :]))  # Store output

        # print loss (optional)
        if verbose and epoch%200==0:
            print(f"Iteration {epoch} out of {epochs}")

    if (epoch==epochs-1): print(f"Reachead max_epochs ( {epochs} )")

    # return trained network and last rmse
    return NN, training_score, validation_score

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    # read max layer size and compute linewidth with an offset
    max_layer_size = max(layer_sizes)
    linewidth = np.min(np.exp( - (max_layer_size - 5 ) / 10 ))

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            if n == len(layer_sizes) - 2:
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='yellow', ec='k', zorder=4)
            elif n == 0:
                circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                        color='blue', ec='k', zorder=4)
            else:
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='green', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', linewidth=linewidth)
                ax.add_artist(line)

