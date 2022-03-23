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
def f (x, l=2*np.pi):
    return np.sin(x*2*np.pi/l)

# Define the number of measurements, and how noisy they are
N = 100
noise = 0.7

# Use a seed, to ensure that the results are reproducible
seed = 0
np.random.seed(seed)

# Define two linspaces along x
# x_pred --> locations where we want to make predictions, i.e. everywhere
# x      --> locations where we observe data
x_pred = np.linspace(0, 2*np.pi, 1000)
x = np.linspace(0, 2*np.pi, N)

# Generate the observed data
t = f(x) + np.random.normal(0, noise, N)

# Define the prediction locations
# (note that these are different from the locations where we observed our data)
x_pred = np.linspace(0, 2*np.pi, 1000)

# Define a function that makes a KNN prediction at the given locations, based on the given (x,t) data
def KNN(x=x, t=t, x_pred=x_pred, k=1):
    
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

################
# Initial plot #
################

# # Create an initial plot
# fig, ax = plt.subplots(figsize=(8,6))
# truth, = plt.plot(x, f(x), 'k-', label=r'Ground truth $f(x)$')
# data, = plt.plot(x, t, 'x', label=r'Noisy data $(x,t)$')
# pred, = plt.plot(x_pred, y_1, '-', label=r'Prediction $y(x)$, $k=1$')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_ylim((-2.5, 2.5))
# plt.legend(loc='lower left')

plot = magicplotter(x, t, x_pred, f(x_pred), x_pred, y_1)

# Adjust the main plot to make room for the sliders
# plt.subplots_adjust(left=0.25, bottom=0.33)

###########
# Sliders #
###########

# # Make a horizontal slider to control the number of neighbors
# ax_k = plt.axes([0.25, 0.20, 0.65, 0.03])
# k_slider = Slider(
#     ax=ax_k,
#     label=r'Neighbors ($k$)',
#     valmin=1,
#     valmax=100,
#     valinit=1,
#     valfmt='%0.0f'
# )

# # Make a horizontal slider to control the number of measurements
# ax_N = plt.axes([0.25, 0.15, 0.65, 0.03])
# N_slider = Slider(
#     ax=ax_N,
#     label=r'Training size ($N$)',
#     valmin=2,
#     valmax=1000,
#     valinit=N,
#     valfmt='%0.0f'
# )

# # Make a horizontal slider to control the wave length
# ax_l = plt.axes([0.25, 0.10, 0.65, 0.03])
# l_slider = Slider(
#     ax=ax_l,
#     label=r'Wave length ($l$)',
#     valmin=0.5,
#     valmax=8*np.pi,
#     valinit=2*np.pi,
# )

# # Make a vertical slider to control the noise level
# ax_eps = plt.axes([0.10, 0.33, 0.0225, 0.50])
# eps_slider = Slider(
#     ax=ax_eps,
#     label=r'Noise ($\varepsilon$)',
#     valmin=0,
#     valmax=1,
#     valinit=noise,
#     orientation="vertical"
# )

# Define an update function when a slider value changes
def update(event):
    
    global seed

    # Get the slider values
    k = int(min(k_slider.val, N_slider.val))
    N = int(N_slider.val)
    l = l_slider.val
    eps = eps_slider.val
    
    # Generate the noisy data again
    np.random.seed(seed)

    x = np.linspace(0, 2*np.pi, N)
    t = f(x, l) + np.random.normal(0, eps, N)
    
    # Update the ground truth and the data in the plots
    truth.set_data(x_pred, f(x_pred,l))
    data.set_data(x, t)
    
    # Fit the KNN to the updated data
    y_pred = KNN(x=x, t=t, x_pred=x_pred, k=k)
    pred.set_ydata(y_pred)
    
    fig.canvas.draw_idle()

plot.add_slider('eps', update)
plot.add_slider('k', update)
plot.add_slider('N', update)
plot.add_slider('l', update)

# # Connect the update function to each slider
# k_slider.on_changed(update)
# N_slider.on_changed(update)
# l_slider.on_changed(update)
# eps_slider.on_changed(update)

###########
# Buttons #
###########

# # Make a button to update the seed
# ax_seed = plt.axes([0.75, 0.025, 0.15, 0.04])
# seed_button = Button(ax_seed, 'New seed', hovercolor='0.975')

# # Make a button to hide/show the truth
# ax_truth = plt.axes([0.50, 0.025, 0.15, 0.04])
# truth_button = Button(ax_truth, 'Hide truth', hovercolor='0.975')

# # Make a button to go back to the initial settings
# ax_reset = plt.axes([0.25, 0.025, 0.15, 0.04])
# reset_button = Button(ax_reset, 'Reset', hovercolor='0.975')

# # Define a function that changes the seed
# def update_seed(event):
    
#     global seed
    
#     seed += 1
    
#     update(event)

# # Define a function that changes the seed
# def toggle_truth(event):
    
#     if truth.get_alpha() is None:
#         truth.set_alpha(0)
#         truth_button.label.set_text('Show truth')
#     else:
#         truth.set_alpha(None)
#         truth_button.label.set_text('Hide truth')
        
#     update(event)
    
# # Define a function that performs a reset
# def reset_all(event):
    
#     global seed
    
#     seed = 0
    
#     k_slider.reset()
#     N_slider.reset()
#     l_slider.reset()
#     eps_slider.reset()
    
#     if not truth.get_alpha() is None:
#         toggle_truth(event)

#     update(event)

# # Connect the correct function to each button
# seed_button.on_clicked(update_seed)
# truth_button.on_clicked(toggle_truth)
# reset_button.on_clicked(reset_all)

plt.show()