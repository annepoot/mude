
import numpy as np
from sklearn.neural_network import MLPRegressor
from mude_tools import neuralnetplotter
from sklearn.preprocessing import StandardScaler

# The true function relating t to x
def f_truth(x, freq=2, **kwargs):

    # Return a sine with a frequency of f
    return np.sin(x * freq)


# The data generation function
def f_data(epsilon=0.7, N=100, **kwargs):

    # Apply a seed if one is given
    if 'seed' in kwargs:
        np.random.seed(kwargs['seed'])

    # Get the minimum and maximum
    xmin = kwargs.get('xmin', 0)
    xmax = kwargs.get('xmax', 2*np.pi)

    # Generate N evenly spaced observation locations
    x = np.linspace(xmin, xmax, N)

    # Generate N noisy observations (1 at each location)
    t = f_truth(x, **kwargs) + np.random.normal(0, epsilon, N)

    # Return both the locations and the observations
    return x, t


# Function that normalizes data
def norm_x(x, **kwargs):
    return (x - kwargs['mean']) / kwargs['std']


# Function that creates a NN
def create_NN(**kwargs):
    return MLPRegressor(solver='sgd', hidden_layer_sizes=(kwargs['neurons'], kwargs['neurons']),
                        activation=kwargs['activation'], batch_size=kwargs['batch_size'])

# Define the prediction locations
# (note that these are different from the locations where we observed our data)
x, t = f_data()
x_pred = np.linspace(-1, 2 * np.pi + 1, 200)

xscaler = StandardScaler()
xscaler.fit(x[:,None])


# Function that trains a given NN for a given number of epochs
def NN_train(x, t, network, **kwargs):

    # Convert the training data to a column vector and normalize it
    X = x.reshape(-1, 1)
    X = xscaler.transform(X)

    # Get the number of epochs per block
    epochs_per_block = int(round(kwargs['epochs'] / kwargs['epoch_blocks'], 0))

    # Run a number of epochs
    for i in range(epochs_per_block):
        network.partial_fit(X, t)

    return network, network.loss_curve_


# Function that returns predictions from a given NN model
def NN_pred(x, t, x_pred, **kwargs):

    # Get the network from the kwargs
    network = kwargs.get('network')
    return_network = kwargs.get('return_network', False)

    if network is None:
        network = create_NN(**kwargs)
        retrain = True
    else:
        retrain = kwargs['train_network']

    # Convert the prediction data to a column vector and normalize it
    X_pred = x_pred.reshape(-1, 1)
    X_pred = xscaler.transform(X_pred)

    if retrain:
        network, train_loss = NN_train(x, t, network, **kwargs)

    # Make a prediction at the locations given by x_pred
    y = network.predict(X_pred)

    if return_network:
        return y, network, train_loss
    else:
        return y


plot1 = neuralnetplotter(f_data, f_truth, create_NN, NN_train, NN_pred, x_pred)  # title=r'Run cell above plot before using!')
plot1.add_sliders('neurons', valmax=20, valinit=3)
plot1.add_buttons('truth', 'seed', 'rerun')
# plot1.add_radiobuttons('activation')
plot1.show()