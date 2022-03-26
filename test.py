###########
# Imports #
###########

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.widgets import Slider, Button
from mude_tools import magicplotter

##############################
# Generate all relevant data #
##############################

# The true function relating t to x
def f (x, l=2*np.pi, **kwargs):
    return np.sin(x*2*np.pi/l)

# Define the number of measurements, and how noisy they are
N = 100
noise = 0.7

def f_data(eps=noise, N=N, **kwargs):
    
    # Use a seed if necessary
    if 'seed' in kwargs:
        np.random.seed(kwargs['seed'])
    
    x = np.linspace(0, 2*np.pi, N)

    return x, f(x) + np.random.normal(0, eps, N)
    
# Use a seed, to ensure that the results are reproducible
seed = 0
np.random.seed(seed)

# Define two linspaces along x
# x_pred --> locations where we want to make predictions, i.e. everywhere
# x      --> locations where we observe data
x_pred = np.linspace(0, 2*np.pi, 1000)

# Generate the observed data
# t = f(x) + np.random.normal(0, noise, N)
x, t = f_data(noise, N)

# Define the prediction locations
# (note that these are different from the locations where we observed our data)
x_pred = np.linspace(0, 2*np.pi, 1000)

# Define a function that makes a KNN prediction at the given locations, based on the given (x,t) data
def KNN(x=x, t=t, x_pred=x_pred, k=1, **kwargs):
    
    # Convert x and x_pred to a column vector in order for KNeighborsRegresser to work
    X = x.reshape(-1,1)
    X_pred = x_pred.reshape(-1,1)
    
    # Train the KNN based on the given (x,t) data
    neigh = KNeighborsRegressor(k)
    neigh.fit(X, t)
    
    # Make a prediction at the locations given by x_pred
    y = neigh.predict(X_pred)
    
    # Return the predicted values
    return y

y_1 = KNN(x, t, x_pred, 1)

plot = magicplotter(f_data, f, KNN, x_pred, x_pred)

plot.add_slider('eps')
plot.add_slider('k')
plot.add_slider('N')
plot.add_slider('l')

plot.add_button('truth')
plot.add_button('reset')
plot.add_button('seed')

plot.add_sidebar()

plot.show()