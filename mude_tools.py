# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.utils import shuffle

class magicplotter:

    # Define the default settings for all sliders
    defaults = {
        'epsilon':{
            'valmin':0,
            'valmax':1,
            'valinit':0.7,
            'valfmt':None,
            'orientation':'vertical',
            'label':r'Noise ($\varepsilon$)',
            'update':'data'
        },
        'k':{
            'valmin':1,
            'valmax':100,
            'valinit':1,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'Neighbors ($k$)',
            'update':'pred'
        },
        'N':{
            'valmin':1,
            'valmax':1000,
            'valinit':100,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'Training size ($N$)',
            'update':'data'
        },
        'freq':{
            'valmin':1/8,
            'valmax':8,
            'valinit':1,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Frequency ($freq$)',
            'update':'data'
        },
        'l':{
            'valmin':1/5,
            'valmax':5,
            'valinit':1,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Length scale ($l$)',
            'update':'pred'
        },
        'degree':{
            'valmin':1,
            'valmax':30,
            'valinit':1,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'Degree ($p$)',
            'update':'pred'
        },
        'M_radial':{
            'valmin':1,
            'valmax':30,
            'valinit':5,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'Number of RBFs ($M$)',
            'update':'pred'
        },
        'l_radial':{
            'valmin':1/5,
            'valmax':5,
            'valinit':1,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Length scale ($l$)',
            'update':'pred'
        },
        'probe':{
            'valmin':0,
            'valmax':1,
            'valinit':0.5,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Probe location',
            'update':'probe'
        },
        'val_pct':{
            'valmin':0,
            'valmax':50,
            'valinit':0,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Validation size ($\%$)',
            'update':'data'
        },
        'truth':{
            'index':1,
            'hovercolor':'0.975',
            'label':'Hide truth',
            'update':'truth'
        },
        'seed':{
            'index':2,
            'hovercolor':'0.975',
            'label':'New seed',
            'update':'seed'
        },
        'reset':{
            'index':3,
            'hovercolor':'0.975',
            'label':'Reset',
            'update':'reset'
        }
    }

    # Store the width, height and aspect ratio of the plot
    # The ratio is necessary to account for the fact that the plot is not 1:1
    w = 8
    h = 6
    r = h / w


    # Create the initial plot
    def __init__(self, f_data, f_truth, f_pred, x_pred = None, x_data = None, y_data = None,
                 x_truth = None, **settings):

        # Define a seed to make sure all results are reproducible
        if "seed" in settings:
            self.seed = settings["seed"]
        else:        
            self.seed = 0
            
        # Create an empty dictionary, which will later store all slider values
        self.sliders = {}
        self.buttons = {}

        # Store the additional settings
        self.settings = settings

        # Collect all key word arguments
        kwargs = self.collect_kwargs()
        
        # Store all variables that were passed to the init function
        self.f_data = f_data
        self.f_truth = f_truth
        self.f_pred = f_pred
        
        if (x_data is None) & (y_data is None):
            self.x_data, self.y_data = self.f_data(**kwargs)
            self.keep_data = False
        else:
            self.x_data, self.y_data = x_data, y_data
            self.keep_data = True
        
        # self.x_train = self.x_data if x_train is None else x_train
        # self.y_train = self.y_data if y_train is None else y_train
        
        self.x_train, self.y_train = self.x_data, self.y_data
        self.x_pred = self.x_data if x_pred is None else x_pred
        self.y_pred = self.f_pred(self.x_train, self.y_train, self.x_pred, **kwargs)
        self.x_truth = self.x_pred if x_truth is None else x_truth
        self.y_truth = self.f_truth(self.x_truth, **kwargs)
        
        # Get additional settings like the original plot title and labels
        self.title = settings.get('title', None)
        self.data_label = settings.get('data_label', r'Training data $(x,t)$')
        self.truth_label = settings.get('truth_label', r'Ground truth $f(x)$')
        self.pred_label = settings.get('pred_label', r'Prediction $y(x)$, $k={k}$')
        self.val_label = settings.get('val_label', r'Validation data $(x,t)$')
        self.probe_label = settings.get('probe_label', r'Probe')
        self.hide_legend = settings.get('hide_legend', False)

        # Get the given axes from the settings, or create a new figure
        if 'ax' in settings:
            self.ax = settings['ax']
            self.fig = self.ax.figure
        else:
            self.fig, self.ax = plt.subplots(figsize=(self.w,self.h))

        # Add the truth, data and prediction
        self.plot_truth, = self.ax.plot(self.x_truth, self.y_truth, 'k-', label=self.truth_label.format(**kwargs))
        self.plot_data, = self.ax.plot(self.x_train, self.y_train, 'x', label=self.data_label.format(**kwargs))
        self.plot_pred, = self.ax.plot(self.x_pred, self.y_pred, '-', label=self.pred_label.format(**kwargs))

        # Initialize the validation set, probe, probe set and sidebar (to ensure they exist)
        self.plot_val = None
        self.plot_probe = None
        self.plot_neighbors = None
        self.ax_mse = None

        # Call the show function
        # This also redirects to the update_data function
        self.shown = False
        self.show()


    def update_data(self, event):
        
        # Go through all sliders in the dictionary, and store their values in a kwargs dict
        kwargs = self.collect_kwargs()

        # Recompute the data and the truth
        if not self.keep_data:
            self.x_data, self.y_data = self.f_data(**kwargs)
        self.y_truth = self.f_truth(self.x_truth, **kwargs)

        # Split the data into training and validation
        if int(kwargs['val_pct']) == 0:

            # Use everything for training if the validation percentage is 0
            self.x_train = self.x_data
            self.y_train = self.y_data
            self.x_val = None
            self.y_val = None

        else:

            # Otherwise, split the training and validation data
            x_shuffle, y_shuffle = shuffle(self.x_data, self.y_data)
            N_split = int(kwargs['N'] * (1-kwargs['val_pct']/100))
            self.x_train = x_shuffle[:N_split]
            self.y_train = y_shuffle[:N_split]
            self.x_val = x_shuffle[N_split:]
            self.y_val = y_shuffle[N_split:]

        # Update the data and ground truth in the plots
        self.plot_data.set_data(self.x_train, self.y_train)

        if not self.plot_truth is None:
            self.plot_truth.set_data(self.x_truth, self.y_truth)

        # Update the validation data set if necessary
        if self.x_val is None:
            if not self.plot_val is None:
                self.ax.lines.remove(self.plot_val)
                self.plot_val = None
        else:
            if self.plot_val is None:
                self.plot_val, = self.ax.plot(self.x_val, self.y_val, 'x', color='purple', label=self.val_label.format(**kwargs))
            else:
                self.plot_val.set_data(self.x_val, self.y_val)

        # Update the legend of the data and ground truth
        self.plot_data.set_label(self.data_label.format(**kwargs))

        if not self.plot_truth is None:
            self.plot_truth.set_label(self.truth_label.format(**kwargs))

        if not self.plot_val is None:
            self.plot_val.set_label(self.val_label.format(**kwargs))

        # Allow for automatic updating of the plot
        self.fig.canvas.draw_idle()

        # After updating the data, the prediction should be updated as well
        self.update_pred(event)


    def update_pred(self, event):

        # Go through all sliders in the dictionary, and store their values in a kwargs dict
        kwargs = self.collect_kwargs()

        # Compute the prediction in the prediction locations
        self.y_pred = self.f_pred(self.x_train, self.y_train, self.x_pred, **kwargs)

        # Compute the training and validation errors
        train_pred = self.f_pred(self.x_train, self.y_train, self.x_train, **kwargs)
        self.mse_train = sum((self.y_train - train_pred)**2) / len(self.x_train)

        if int(kwargs['val_pct']) == 0:
            self.mse_validation = 0
        else:
            val_pred = self.f_pred(self.x_train, self.y_train, self.x_val, **kwargs)
            self.mse_validation = sum((self.y_val - val_pred)**2) / len(self.x_val)

        # Add the training / validation errors to the dictionary
        kwargs['mse_train'] = self.mse_train
        kwargs['mse_validation'] = self.mse_validation

        # Update the prediction in the plots
        self.plot_pred.set_data(self.x_pred, self.y_pred)

        # Update the sidebar if necessary
        if not self.ax_mse is None:

            self.plot_mse_train.set_data(0, self.mse_train)

            if self.x_val is None:
                if not self.plot_mse_val is None:
                    self.ax_mse.lines.remove(self.plot_mse_val)
                    self.plot_mse_val = None
            else:
                if self.plot_mse_val is None:
                    self.plot_mse_val, = self.ax_mse.plot(1, self.mse_validation, 'o', color='purple', label='Validation error')
                else:
                    self.plot_mse_val.set_data(1, self.mse_validation)

        # Update the legend
        self.plot_pred.set_label(self.pred_label.format(**kwargs))

        if self.hide_legend:
            self.ax.legend = None
        else:
            self.ax.legend(loc='lower left')

        # Update the title
        if self.title is None:
            self.ax.set_title(None)
        else:
            self.ax.set_title(self.title.format(**kwargs))

        # Allow for automatic updating of the plot
        self.fig.canvas.draw_idle()

        # After updating the data, the probe should be updated as well
        self.update_probe(event)


    def update_probe(self, event):

        # Exit the function if no probe exists
        if self.plot_probe is None:
            return

        # Go through all sliders in the dictionary, and store their values in a kwargs dict
        kwargs = self.collect_kwargs()

        # Compute the x-coordinate of the probe
        self.x_probe = min(self.x_train) + self.sliders['probe'].val * (max(self.x_train) - min(self.x_train))

        # Get the KNN regressor
        neigh = self.f_pred(self.x_train, self.y_train, self.x_pred, return_regressor=True, **kwargs)

        # Get the k nearest neighbors
        kneighbors = neigh.kneighbors(self.x_probe.reshape(-1, 1), return_distance=False)[0,:]

        # Get the corresponding predictor and target values
        xneighbors = self.x_train[kneighbors]
        yneighbors = self.y_train[kneighbors]

        # Update the highlighted data points that correspond to the k nearest neighbors
        self.plot_neighbors.set_data(xneighbors, yneighbors)

        # Get the interval data
        xmin = min(xneighbors)
        xmax = max(xneighbors)
        y = self.y_probe
        d = 0.1

        # Display an interval if more than 1 neighbor is used
        if len(kneighbors) > 1:
            markers_on = np.array([4])
            xinterval = np.array([xmin, xmin, xmin, xmin, self.x_probe, xmax, xmax, xmax, xmax])
            yinterval = np.array([y, y-d, y+d, y, self.y_probe, y, y-d, y+d, y])
        else:
            markers_on = np.array([0])
            xinterval = self.x_probe
            yinterval = self.y_probe

        # Update the probe and corresponding interval
        self.plot_probe.set_data(xinterval, yinterval)
        self.plot_probe.set_markevery(markers_on)

        # Allow for automatic updating of the plot
        self.fig.canvas.draw_idle()


    # Define the function that will be called when the hide/show truth button is called
    def toggle_truth(self, event):

        if self.plot_truth is None:
            self.plot_truth, = self.ax.plot(self.x_truth, self.y_truth, 'k-', label=self.truth_label.format(**self.collect_kwargs()))
            self.buttons['truth'].label.set_text('Hide truth')
        else:
            self.ax.lines.remove(self.plot_truth)
            self.plot_truth = None
            self.buttons['truth'].label.set_text('Show truth')

        # Update the legend
        if self.hide_legend:
            self.ax.legend = None
        else:
            self.ax.legend(loc='lower left')


    # Define a function that changes the seed
    def update_seed(self, event):

        self.seed += 1

        self.update_data(event)


    # Define a function that performs a reset
    def reset_all(self, event):

        # Reset the seed
        self.seed = 0

        # Reset all sliders
        for slider in self.sliders.values():
            slider.reset()

        # Show the truth again if necessary
        if self.plot_truth is None:
            self.toggle_truth(event)

        self.update_data(event)


    # Add a slider to the bottom or left side of the plot
    def add_slider(self, var, **settings):

        # Check if the variable is in defaults
        def_settings = self.defaults.get(var, {})

        # Load all default/given values*
        valmin = settings['valmin'] if 'valmin' in settings else def_settings['valmin']
        valmax = settings['valmax'] if 'valmax' in settings else def_settings['valmax']
        valinit = settings['valinit'] if 'valinit' in settings else def_settings['valinit']
        valfmt = settings['valfmt'] if 'valfmt' in settings else def_settings['valfmt']
        orientation = settings['orientation'] if 'orientation' in settings else def_settings['orientation']
        label = settings['label'] if 'label' in settings else def_settings['label']
        update = settings['update'] if 'update' in settings else def_settings['update']

        # Create the slider
        # Note: it is important that the slider is not created in exactly the same place as before
        # otherwise, matplotlib will reuse the same axis
        ax_slider = self.fig.add_axes([0.5 + 0.1 * len(self.sliders), 0.5, 0.1, 0.1])
        slider = Slider(
            ax=ax_slider,
            label=label,
            valmin=valmin,
            valmax=valmax,
            valinit=valinit,
            valfmt=valfmt,
            orientation=orientation
        )

        # Add the slider to the dictionary that will store the slider values
        self.sliders[var] = slider

        # Get the correct update function
        update_func = self.get_update_func(update)

        # Add an event to the slider
        slider.on_changed(update_func)

        # Adjust the plot to make room for the added slider
        self.adjust_plot()


    # Add a button to the bottom or left side of the plot
    def add_button(self, var, **settings):

        # Check if the variable is in defaults
        def_settings = self.defaults.get(var, {})

        # Load all default/given values
        hovercolor = settings['hovercolor'] if 'hovercolor' in settings else def_settings['hovercolor']
        label = settings['label'] if 'label' in settings else def_settings['label']
        update = settings['update'] if 'update' in settings else def_settings['update']

        # Create the button
        # Note: it is important that the button is not created in exactly the same place as before
        # otherwise, matplotlib will reuse the same axis
        ax_button = self.fig.add_axes([0.1 * len(self.buttons), 0., 0.1, 0.1])
        button = Button(
            ax=ax_button,
            label=label,
            hovercolor=hovercolor
        )

        # Add the slider to the dictionary that will store the slider values
        self.buttons[var] = button

        # Get the correct update function
        update_func = self.get_update_func(update)

        # Add an event to the slider
        button.on_clicked(update_func)

        # Adjust the plot to make room for the added slider
        self.adjust_plot()


    # Add the mse sidebar to the right side of the plot
    def add_sidebar(self):

        # Add 20% validation set size to the settings if it is not already given
        if not 'val_pct' in self.settings:
            self.settings['val_pct'] = 20

        # Make sure that the sidebar doesn't already exist
        if self.ax_mse is None:

            # Create the sidebar axes
            self.ax_mse = self.fig.add_axes([0.8, 0.2, 0.2, 0.8])

            # Plot both mean square errors
            self.plot_mse_train, = self.ax_mse.plot(0, self.mse_train, 'o', label='Training error')

            if self.x_val is None:
                self.plot_mse_val = None
            else:
                self.plot_mse_val, = self.ax_mse.plot(1, self.mse_validation, 'o', color='purple', label='Validation error')

            # Set the layout of the sidebar
            self.ax_mse.yaxis.grid()
            self.ax_mse.set_xlim((-.9, 1.9))
            self.ax_mse.set_ylim((-0.05, 1.5))
            self.ax_mse.set_xticks([0, 1])
            self.ax_mse.set_xticklabels(['Training', 'Validation'], rotation=45)
            self.ax_mse.set_ylabel('Mean Squared Error')

        # Adjust the plot to make room for the added sidebar
        self.adjust_plot()


    # Add a probe to display the effect region as a function of k
    def add_probe(self):

        # Add a slider that controls the probe location
        if not 'probe' in self.sliders:
            self.add_slider('probe')

        # Go through all sliders in the dictionary, and store their values in a kwargs dict
        kwargs = self.collect_kwargs()

        # Compute the x-coordinate of the probe
        self.x_probe = min(self.x_train) + self.sliders['probe'].val * (max(self.x_train) - min(self.x_train))
        self.y_probe = max(self.y_train)

        # Add a probe to the plot
        self.plot_probe, = self.ax.plot(self.x_probe, self.y_probe, '-o', color='C2', label=self.probe_label.format(**kwargs))

        # Highlight the k neighbors that are nearest to the probe
        self.plot_neighbors, = self.ax.plot(self.x_probe, self.y_probe, 'X', color='C2', alpha=0.5)


    # A nice wrapper to add multiple sliders at once
    def add_sliders(self, *var_list, **settings):

        for var in var_list:
            self.add_slider(var, **settings)


    # A nice wrapper to add multiple buttons at once
    def add_buttons(self, *var_list, **settings):

        for var in var_list:
            self.add_button(var, **settings)

    # Adjust the plot to make room for the sliders
    def adjust_plot(self):

        r = self.r
        slider_thick = 0.03
        hor_slider_space = 0.02
        ver_slider_space = 0.07
        hor_label_space = 0.10
        ver_label_space = 0.12
        button_thick = 0.04
        button_length = 0.15
        sidebar_width = 0.1

        hor_sliders = [slider for slider in self.sliders.values() if slider.orientation=='horizontal']
        ver_sliders = [slider for slider in self.sliders.values() if slider.orientation=='vertical']

        # Get all the sizes of the main plot
        bottom = max(hor_label_space + (hor_slider_space + slider_thick) * len(hor_sliders) + (hor_slider_space + button_thick) * (len(self.buttons) > 0) + (hor_slider_space + button_thick) * (len(self.buttons) > 3), 0.1)
        left = max(ver_label_space + (ver_slider_space + slider_thick) * len(ver_sliders), 0.2)
        top = 0.1
        right = 0.1 + (sidebar_width + ver_slider_space) * (not self.ax_mse is None)
        height = 1 - top - bottom
        width = 1 - right - left

        # Set the size of the main plot
        self.ax.set_position([left, bottom, 1-left-right, 1-bottom-top])

        # Set the position of the horizontal sliders one by one
        for i, slider in enumerate(hor_sliders):

            # Set the position of the slider
            slider.ax.set_position(
                [left,
                 bottom - hor_label_space - slider_thick - (hor_slider_space + slider_thick) * i,
                 width,
                 slider_thick])

        # Set the position of the vertical sliders one by one
        for i, slider in enumerate(ver_sliders):

            # Set the position of the slider
            slider.ax.set_position(
                [left - (ver_label_space + slider_thick + (ver_slider_space + slider_thick) * i) * r,
                 bottom,
                 slider_thick * r,
                 height])

        # Calculate the spacing needed for 3 buttons
        n_button = 3
        button_space = (width - n_button * button_length) / (n_button-1)

        # Set the position of the buttons one by one
        for val, button in self.buttons.items():

            # Find the left side of the button
            if val == 'truth':
                i = 0
            elif val == 'seed':
                i = 1
            elif val == 'reset':
                i = 2
            elif val == 'D_small':
                i = 3
            elif val == 'D_medium':
                i = 4
            elif val == 'D_large':
                i = 5

            # Set the position of the button
            button.ax.set_position(
                [left + (button_space + button_length) * (i % n_button),
                 bottom - hor_label_space - (hor_slider_space + slider_thick) * len(hor_sliders) - (hor_slider_space + button_thick) * (i // 3) - button_thick,
                 button_length,
                 button_thick])

        # Set the position of the sidebar if it exists
        if not self.ax_mse is None:

            # Change the size of the main figure
            self.fig.set_figwidth(self.w + 2)

            # Set the position of the sidebar
            self.ax_mse.set_position(
                [1 - right + ver_slider_space,
                 bottom,
                 sidebar_width,
                 height])

        # Set the limits of the main plot x-axis based on the observed data
        xmax = max(self.x_data)
        xmin = min(self.x_data)
        xdif = xmax - xmin
        self.ax.set_xbound((xmin - 0.1 * xdif, xmax + 0.1 * xdif))

        # Set the limits of the main plot y-axis based on the observed data
        ymax = max(self.y_data)
        ymin = min(self.y_data)
        ydif = ymax - ymin
        self.ax.set_ybound((ymin - 0.1 * ydif, ymax + 0.1 * ydif))

        # Disable autoscaling, to make sure the limits remain enforced
        plt.autoscale(False)

    # Define a function that collects all key word arguments
    def collect_kwargs(self):

        # Initialize an empty dictionary
        kwargs = {}

        # First, go through all sliders in the default settings and store their initial values
        for val, val_dict in self.defaults.items():

            # Check if the slider has a initial value
            if 'valinit' in val_dict:

                # If so, store it in the kwargs
                kwargs[val] = val_dict['valinit']

        # Then, go through all custom settings, and store their values
        for val, setting in self.settings.items():

            kwargs[val] = setting

        # Lastly, go through all current sliders, and store their current values
        for val, slider in self.sliders.items():

            # Check if the slider should return an integer
            if slider.valfmt == '%0.0f':
                kwargs[val] = round(slider.val)
            else:
                kwargs[val] = slider.val

        # Add the current seed to the dictionary
        kwargs['seed'] = self.seed

        return kwargs


    # This function takes a string and returns the corresponding update function
    def get_update_func(self, update):

        if update == 'data':
            return self.update_data
        elif update == 'pred':
            return self.update_pred
        elif update == 'seed':
            return self.update_seed
        elif update == 'reset':
            return self.reset_all
        elif update == 'truth':
            return self.toggle_truth
        elif update == 'probe':
            return self.update_probe
        elif update == 'datasets_small':
            return self.update_datasets_small
        elif update == 'datasets_medium':
            return self.update_datasets_medium
        elif update == 'datasets_large':
            return self.update_datasets_large


    # Define a show function, so importing matplotlib is not strictly necessary in the notebooks
    def show(self):

        # Update the plot
        self.update_data(0)

        # Adjust the plot
        self.adjust_plot()

        # Check if show has already been called
        if 'ax' in self.settings:
            self.shown = True

        # Check if the plot has already been shown
        if not self.shown:

            # If not, forward to plt.show()
            # Note that plt.show() should only be called once!
            plt.show()

            # Remember that the plot has now been shown
            self.shown = True


class biasvarianceplotter(magicplotter):

    # Store the sizes of the small, medium and large number of data sets
    D_small = 10
    D_medium = 30
    D_large = 100

    # Define the default settings for all sliders
    defaults = {
        'D':{
            'valmin':1,
            'valmax':100,
            'valinit':25,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'# of datasets ($D$)',
            'update':'pred'
        },
        'D_small':{
            'index':4,
            'hovercolor':'0.975',
            'label':'{} datasets'.format(D_small),
            'update':'datasets_small'
        },
        'D_medium':{
            'index':5,
            'hovercolor':'0.975',
            'label':'{} datasets'.format(D_medium),
            'update':'datasets_medium'
        },
        'D_large':{
            'index':6,
            'hovercolor':'0.975',
            'label':'{} datasets'.format(D_large),
            'update':'datasets_large'
        }
    }

    # Add the defaults settings for the sliders from the magicplotter
    defaults.update(magicplotter.defaults)


    # Create the initial plot
    def __init__(self, f_data, f_truth, f_pred, x_pred = None, x_truth = None, **settings):

        # Get additional settings like the original labels
        self.title = settings.get('title', r'Bias and variance computed over {D} datasets')
        self.bias_label = settings.get('bias_label', r'Bias')
        self.variance_label = settings.get('variance_label', r'Variance (95% confidence)')
        self.datasets = settings.get('D', self.D_medium)

        # Store the title in the settings (to prevent it from being overwritten if the default is used)
        settings['title'] = self.title

        # Perform the initialization of the magicplotter stuff first
        # This also redirects to the update_pred function
        super().__init__(f_data, f_truth, f_pred, x_pred, x_truth, **settings)


        # Delete the data from the plot (this is a remnant from magicplotter)
        self.ax.lines.remove(self.plot_data)

        # Change the prediction to the mean prediction
        self.plot_pred.set_data(self.x_pred, self.y_mean_pred)
        self.plot_pred.set_color('C0')
        self.pred_label = r'$\mathbb{{E}}_{{\mathcal{{D}}}}[y(x)](k={k})$'


    def update_pred(self, event):

        # Go through all sliders in the dictionary, and store their values in a kwargs dict
        kwargs = self.collect_kwargs()

        # Add the dataset size to the kwargs
        kwargs['D'] = self.datasets

        # Recompute the truth
        self.y_truth = self.f_truth(self.x_truth, **kwargs)

        # Compute the predictions of all D models in the prediction locations
        self.make_preds(self.x_train, self.x_pred, **kwargs)

        # Compute the truth in all prediction locations
        pred_truth = self.f_truth(self.x_pred, **kwargs)

        # Compute the bias by taking the difference between the truth and the mean prediction
        bias_bottom = np.minimum(self.y_mean_pred, pred_truth)
        bias_top = np.maximum(self.y_mean_pred, pred_truth)

        # Compute the 95% confidence interval of the variance
        variance_bottom = bias_bottom-np.sqrt(self.variance)*2
        variance_top = bias_top+np.sqrt(self.variance)*2

        # Remove the confidence intervals from the plot
        if 'plot_bias' in vars(self):
            self.plot_bias.remove()
        if 'plot_variance_bottom' in vars(self):
            self.plot_variance_bottom.remove()
        if 'plot_variance_top' in vars(self):
            self.plot_variance_top.remove()

        # Add the bias and variance to the plots again
        self.plot_bias = self.ax.fill_between(self.x_pred, bias_bottom, bias_top, alpha=0.6, color='red', label=self.bias_label.format(**kwargs))
        self.plot_variance_bottom = self.ax.fill_between(self.x_pred, variance_bottom, bias_bottom, color='red', alpha=0.3, label=self.variance_label.format(**kwargs))
        self.plot_variance_top = self.ax.fill_between(self.x_pred, bias_top, variance_top, color='red', alpha=0.3)

        # Update the data and mean prediction in the plots
        self.plot_pred.set_data(self.x_pred, self.y_mean_pred)

        if not self.plot_truth is None:
            self.plot_truth.set_data(self.x_truth, self.y_truth)

        # Update the legend
        self.plot_pred.set_label(self.pred_label.format(**kwargs))

        if not self.plot_truth is None:
            self.plot_truth.set_label(self.truth_label.format(**kwargs))

        if self.hide_legend:
            self.ax.legend = None
        else:
            self.ax.legend(loc='lower left')

        # Update the title
        if self.title is None:
            self.ax.set_title(None)
        else:
            self.ax.set_title(self.title.format(**kwargs))

        # Allow for automatic updating of the plot
        self.fig.canvas.draw_idle()


    def update_datasets_small(self, event):

        self.datasets = self.D_small

        self.update_pred(event)


    def update_datasets_medium(self, event):

        self.datasets = self.D_medium

        self.update_pred(event)


    def update_datasets_large(self, event):

        self.datasets = self.D_large

        self.update_pred(event)


    def make_preds(self, x, x_pred, D, **kwargs):

        # Remember the seed that was passed to the function
        if 'seed' in kwargs:
            seed = kwargs['seed']

        # Get the number of predictions
        N_pred = len(x_pred)

        # Array to store mean prediction in; required for computing the bias
        self.y_mean_pred = np.zeros(N_pred)

        # Store each model output; required for computing the variance
        y_pred_list = []

        # We average over many models with independent datasets
        for i in range(D):

            # Update the seed to ensure different data sets
            # Note that +D is used instead of +1, since +1 is used by 'update_seed', which produces partly the same datasets
            if 'seed' in kwargs:
                kwargs['seed'] += D

            # Generate the next data set
            x, t = self.f_data(**kwargs)

            # Run the prediction model model
            y_pred = self.f_pred(x, t, x_pred, **kwargs)

            # Store predictions
            self.y_mean_pred += y_pred / D

            y_pred_list.append(y_pred)

        # Set the seed in the kwargs back to its original value
        if 'seed' in kwargs:
            kwargs['seed'] = seed

        # Compute the average bias
        self.bias = self.y_mean_pred - self.f_truth(self.x_pred, **kwargs)

        # Compute the variance
        self.variance = np.zeros(N_pred)
        for y_pred in y_pred_list:
            self.variance += (y_pred-self.y_mean_pred)**2
        self.variance /= D
