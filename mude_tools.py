# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
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
    def __init__(self, f_data, f_truth, f_pred, x_pred = None, x_truth = None, **settings):

        # Define a seed to make sure all results are reproducible
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
        self.x_data, self.y_data = self.f_data(**kwargs)
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
            N_split = int(np.ceil(kwargs['N'] * (1-kwargs['val_pct']/100)))
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

        F = self.r
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


class neuralnetplotter(magicplotter):

    # Define the default settings for all sliders
    defaults = {
        'neurons': {
            'valmin': 1,
            'valmax': 35,
            'valinit': 20,
            'valfmt': '%0.0f',
            'orientation': 'horizontal',
            'label': r'Neurons / layer',
            'update': 'change_neurons',
            'position': 'Left'
        },
        'epochs': {
            'valmin': 1000,
            'valmax': 10000,
            'valinit': 10000,
            'valfmt': '%0.0f',
            'orientation': 'horizontal',
            'label': r'Training epochs',
            'update': 'passive',
            'position': 'Left',
            'valstep': np.arange(1000, 10050, 50)
        },
        'cur_model': {
            'valmin': 1,
            'valmax': 200,
            'valinit': 200,
            'valfmt': '%0.0f',
            'orientation': 'horizontal',
            'label': 'Selected model',
            'update': 'change model',
            'position': 'Right'
            # 'valstep': np.concatenate([[1], np.arange(10, 810, 10)])
        },
        'batch_size': {
            'valmin': 1,
            'valmax': 100,
            'valinit': 5,
            'valfmt': '%0.0f',
            'orientation': 'horizontal',
            'label': r'Samples per batch',
            'update': 'passive',
            'position': 'Left'
        },
        'rerun': {
            'index': 3,
            'hovercolor': '0.975',
            'label': 'Run',
            'update': 'train'
        },
        'activation': {
            'index': 4,
            'active': 2,
            'activecolor': 'black',
            'valinit': 'tanh',
            'labels': ['identity', 'relu', 'tanh'],
            'update': 'update_activation'
        },
    }

    defaults.update(magicplotter.defaults)

    # Modify the preexisting defaults appropriately
    defaults['epsilon'].update(
        {
            'valinit': 0.4,
            'orientation': 'horizontal',
            'update': 'passive',
            'position': 'Left'
        })
    defaults['k'].update(
        {
            'update': 'passive',
            'position': 'Left'
        })
    defaults['N'].update(
        {
            'valmin': 2,
            'valmax': 200,
            'valinit': 30,
            'update': 'passive',
            'position': 'Left'
        })
    defaults['freq'].update(
        {
            'valmax': 5,
            'valinit': 3.4,
            'update': 'passive',
            'position': 'Left'
        })
    defaults['val_pct'].update(
        {
            'valmax': 60,
            'valinit': 30,
            'update': 'passive',
            'position': 'Left',
            'valstep': np.arange(0, 65, 5)
        })
    defaults['reset'].update(
        {
            'update': 'passive'
        })

    # Create the initial plot
    def __init__(self, f_data, f_truth, f_create, f_train, f_pred, x_pred=None, x_truth=None, network=None, **settings):

        # Define a seed to make sure all results are reproducible
        self.seed = 2

        # Create an empty dictionary, which will later store all slider values
        self.sliders = {}
        self.sliders_right = {}
        self.buttons = {}
        self.radio_buttons = {}
        # self.trained_models = []

        # Store the additional settings
        self.settings = settings

        # Collect all key word arguments
        kwargs = self.collect_kwargs()

        # Store all variables that were passed to the init function
        self.f_data = f_data
        self.f_truth = f_truth
        self.f_create = f_create
        self.f_train = f_train
        self.f_pred = f_pred
        self.x_data, self.y_data = self.f_data(**kwargs)
        self.x_train, self.y_train = self.x_data, self.y_data
        self.x_pred = self.x_data if x_pred is None else x_pred
        self.x_truth = self.x_pred if x_truth is None else x_truth
        self.y_truth = self.f_truth(self.x_truth, **kwargs)

        # Data for truth function
        self.x_dense = np.linspace(0, 2 * np.pi, 1500)
        self.y_dense = f_truth(self.x_dense, **kwargs) + np.random.normal(0, kwargs['epsilon'], len(self.x_dense))

        # Get additional settings like the original plot title and labels
        self.title = settings.get('title', None)
        self.data_label = settings.get('data_label', r'Training data $(x,t)$')
        self.truth_label = settings.get('truth_label', r'Ground truth $f(x)$')
        self.pred_label = settings.get('pred_label', r'Prediction $y(x)$, $k={k}$')
        self.val_label = settings.get('val_label', r'Validation data $(x,t)$')
        self.probe_label = settings.get('probe_label', r'Probe')
        self.hide_legend = settings.get('hide_legend', False)

        # self.vline_label = settings.get('vline_label', 'Selected model')
        self.train_loss_label = settings.get('tr_label', 'Training error')
        self.val_loss_label = settings.get('val_label', 'Validation error')
        self.true_loss_label = settings.get('true_label', 'True error')

        self.fig = plt.figure(figsize=(self.w, self.h))
        self.ax = self.fig.add_subplot(121)
        self.ax.set_title(self.title)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('t')

        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlim(0, 10000)
        self.ax2.set_ylim(0, 4)
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('RMSE')

        # Add the truth, data and prediction
        self.plot_truth, = self.ax.plot(self.x_truth, self.y_truth, 'k-', label=self.truth_label.format(**kwargs))
        self.plot_data, = self.ax.plot(self.x_train, self.y_train, 'x', label=self.data_label.format(**kwargs))
        self.plot_pred, = self.ax.plot([], [], '-', label=self.pred_label.format(**kwargs))
        self.plot_tr_loss, = self.ax2.plot([], [], '-', label=self.train_loss_label.format(**kwargs))
        self.plot_true_loss, = self.ax2.plot([], [], 'k-', label=self.true_loss_label.format(**kwargs))
        self.plot_val_loss, = self.ax2.plot([], [], '-', color='purple', label=self.val_loss_label.format(**kwargs))

        self.plot_cur_epoch = self.ax2.axvline(kwargs['epochs'], 0, 1, linestyle='--', color='green')

        # Initialize the validation set, probe, probe set and sidebar (to ensure they exist)
        self.plot_val = None
        self.plot_probe = None
        self.plot_neighbors = None

        # Draw image of the activation function
        self.activation_ax = self.fig.add_axes([0.85, 0.02, 0.10, 0.10], anchor='SW', zorder=1)
        self.activation_ax.set_xlim(-4, 4)
        self.activation_ax.set_ylim(-2, 2)
        self.plot_act, = self.activation_ax.plot([], [])
        self.activation_ax.axis('off')

        # Call the show function
        # This also redirects to the update_data function
        self.shown = False
        self.show()

    # When a slider is changed, don't do anything except changing the 'Run' button appearance
    def passive(self, event):
        self.buttons['rerun'].color = 'red'
        self.buttons['rerun'].label.set_text('Re-run')
        self.buttons['rerun'].hovercolor = 'green'

    # Function to change the image & buttons when the number of neurons in the model changes
    def change_neurons(self, event):
        kwargs = self.collect_kwargs()

        # Update network image
        self.network_ax.cla()  # delete previous image
        self.network_ax.axis('off')
        draw_neural_net(self.network_ax, .1, .9, .1, .9, [1, kwargs['neurons'], kwargs['neurons'], 1])
        self.fig.canvas.draw()

        # Do standard slider updating
        self.passive(event)

    # Function that plots the activation function
    def plot_activation(self):
        x_act = np.linspace(-4, 4, 100)
        kwargs = self.collect_kwargs()
        activation = kwargs['activation']

        if activation == 'identity':
            y_act = x_act
        elif activation == 'relu':
            y_act = [max(0, xi) for xi in x_act]
        elif activation == 'tanh':
            y_act = np.tanh(x_act)

        self.plot_act.set_data(x_act, y_act)
        self.fig.canvas.draw()

    # A function that changes the appearances when changing the activation
    def update_activation(self, event):
        self.plot_activation()
        self.passive(event)

    # Function that loads a previously computed model
    def change_model(self, event):
        kwargs = self.collect_kwargs()
        epochs = kwargs['epochs']  # 1- epochs
        cur_epoch = int(kwargs['cur_model'] / 200 * epochs)

        # Change vertical line in loss plot (can't adjust x-position only; so re-draw with extreme bounds)
        self.plot_cur_epoch.set_data([[cur_epoch, cur_epoch], [-1, 9e9]])

        if len(self.trained_models) != 0:
            y_pred = self.trained_models[kwargs['cur_model']]
            self.plot_pred.set_data(self.x_pred, y_pred)

        # Allow for automatic updating of the plot
        self.fig.canvas.draw_idle()

    def trainloop(self, epoch_blocks, **kwargs):
        mse_validation_ar = []
        self.mse_true_ar = []
        self.rmse_true_ar = []

        # Loop over epoch nums. A block here contains multiple epochs
        for i in range(epoch_blocks):
            # Train for a fixed number of epochs
            self.network, mse_train_ar = self.f_train(self.network, self.x_train, self.y_train, **kwargs)

            rmse_train_ar = np.sqrt(np.array(mse_train_ar) * 2)     # Automatically computes (1/(2N) SUM ||..||2 ), so multiply by 2 first

            y_pred = self.f_pred(self.network, self.x_pred, **kwargs)
            self.plot_pred.set_data(self.x_pred, y_pred)

            # Compute the validation error
            if self.x_val is not None:
                val_pred = self.f_pred(self.network, self.x_val, **kwargs)

                mse_validation_ar.append(sum((val_pred - self.y_val) ** 2) / len(self.x_val))
                rmse_validation_ar = np.sqrt(np.array(mse_validation_ar))

                self.plot_val_loss.set_data(np.arange(len(rmse_validation_ar)) * self.epochs_per_block, rmse_validation_ar)

                # Compute boundary of the plot
                upper_bound = max(min(rmse_train_ar) * 2, min(rmse_validation_ar) * 1.5, sum(rmse_validation_ar[-5:]) / 5 * 1.2, sum(self.rmse_true_ar[-5:]) / 5 * 1.2)
            else:
                upper_bound = max(min(rmse_train_ar) * 2, sum(self.rmse_true_ar[-5:]) / 5 * 1.2)

            self.rmse_true_ar.append(np.sqrt(self.dense_sampling_error(**kwargs)))

            self.ax2.set_ybound((0, upper_bound))

            self.plot_tr_loss.set_data(np.arange(len(rmse_train_ar)), rmse_train_ar)
            self.plot_true_loss.set_data(np.arange(len(self.rmse_true_ar)) * self.epochs_per_block, self.rmse_true_ar)
            self.fig.canvas.draw()

            # Store prediction to select back after training
            self.trained_models.append(y_pred)

    # The train function
    def train(self, event):
        kwargs = self.collect_kwargs()

        # Update the legend
        self.plot_pred.set_label(self.pred_label.format(**kwargs))

        self.plot_val_loss.set_data([], []) # Empty validation loss, to reset it in case of %=0.

        self.trained_models = []

        # Change colors of the Rerun button while training
        self.buttons['rerun'].color = 'green'
        self.buttons['rerun'].label.set_text('Running')
        self.buttons['rerun'].hovercolor = 'green'

        # Create initial network
        self.network = self.f_create(**kwargs)

        epochs = kwargs['epochs']
        # A block here is multiple epochs with a single validation update
        epoch_blocks = 200   #  Matches number of 'selected models'
        self.epochs_per_block = int(round(epochs / epoch_blocks, 0)) # This can give a mismatch between shown and selected model
        kwargs['epochs_per_block'] = self.epochs_per_block

        # Set limit of loss function plot
        self.ax2.set_xbound((0, epochs))
        # Set 'selected model' to end
        if 'cur_model' in self.sliders_right:
            self.sliders_right['cur_model'].set_val(epoch_blocks)

        # Data
        self.update_data(0)

        # Add normalization bounds to kwargs based on provided data
        kwargs['mean'] = np.mean(self.x_data)
        kwargs['std'] = np.std(self.x_data)

        # Update legend
        if self.hide_legend:
            self.ax.legend = None
            self.ax2.legend = None
        else:
            self.ax.legend(loc='lower left')
            self.ax2.legend(loc='lower left', bbox_to_anchor=(0.0, 0.95))

        self.trainloop(epoch_blocks, **kwargs)

        # Change colors of the Rerun button back
        self.buttons['rerun'].color = 'gray'
        self.buttons['rerun'].label.set_text('Finished')
        self.buttons['rerun'].hovercolor = 'gray'

        self.fig.canvas.draw_idle()

    # Approximate the 'true error' with a dense sampling of the space & network predictions
    def dense_sampling_error(self, **kwargs):
        return sum((self.y_dense - self.f_pred(self.network, self.x_dense, **kwargs)) ** 2) / len(self.x_dense)

    # Add a slider below the plot; with minimal spacing
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
        position = settings['position'] if 'position' in settings else def_settings['position']
        if 'valstep' in settings:
            valstep = settings['valstep']
        elif 'valstep' in def_settings:
            valstep = def_settings['valstep']
        else:
            valstep = None

        # Create the slider
        # Note: it is important that the slider is not created in exactly the same place as before
        # otherwise, matplotlib will reuse the same axis
        ax_slider = self.fig.add_axes([0.8 + 0.1 * (len(self.sliders) + len(self.sliders_right)), 0.5, 0.1, 0.1])
        slider = Slider(
            ax=ax_slider,
            label=label,
            valmin=valmin,
            valmax=valmax,
            valinit=valinit,
            valfmt=valfmt,
            valstep=valstep,
            orientation=orientation,
        )

        # Add the slider to the dictionary that will store the slider values
        if position == 'Left':
            self.sliders[var] = slider
        elif position == 'Right':
            self.sliders_right[var] = slider

        # Get the correct update function
        update_func = self.get_update_func(update)

        # Add an event to the slider
        slider.on_changed(update_func)

        # Hide the value of the model selector
        if update == 'change model':
            slider.valtext.set_visible(False)

        # Draw image of the neural network if the number of neurons are optional
        if update == 'change_neurons':
            self.network_ax = self.fig.add_axes([0.72, 0.11, 0.27, 0.27], anchor='SW', zorder=0)
            draw_neural_net(self.network_ax, .1, .9, .1, .9, [1, valinit, valinit, 1])
            self.network_ax.axis('off')
            self.fig.canvas.draw()

        # Adjust the plot to make room for the added slider
        self.adjust_plot()


    # Add a radiobutton to the plot
    def add_radiobutton(self, var, **settings):

        # Check if the variable is in defaults
        def_settings = self.defaults.get(var, {})

        # Load all default/given values
        activecolor = settings['activecolor'] if 'activecolor' in settings else def_settings['activecolor']
        labels = settings['labels'] if 'labels' in settings else def_settings['labels']
        update = settings['update'] if 'update' in settings else def_settings['update']
        active = settings['active'] if 'active' in settings else def_settings['active']

        # Create the button
        # Note: it is important that the radiobutton is not created in exactly the same place as before
        # otherwise, matplotlib will reuse the same axis
        ax_button = self.fig.add_axes([0.1 * len(self.radio_buttons), 0., 0.1, 0.1])
        radiobutton = RadioButtons(
            ax=ax_button,
            labels=labels,
            active=active,
            activecolor=activecolor
        )

        # Add the radiobutton to the dictionary that will store the radiobutton values
        self.radio_buttons[var] = radiobutton

        # Get the correct update function
        update_func = self.get_update_func(update)

        # Add an event to the radiobutton
        radiobutton.on_clicked(update_func)

        # Draw image of the activation function if it is optional
        if update == 'update_activation':
            self.plot_activation()

        # Adjust the plot to make room for the added radiobutton
        self.adjust_plot()

    # A nice wrapper to add multiple radio_buttons at once
    def add_radiobuttons(self, *var_list, **settings):

        for var in var_list:
            self.add_radiobutton(var, **settings)

    # This function takes a string and returns the corresponding update function
    def get_update_func(self, update):
        if update == 'passive':
            return self.passive
        if update == 'update_activation':
            return self.update_activation
        if update == 'change model':
            return self.change_model
        elif update == 'train':
            return self.train
        elif update == 'change_neurons':
            return self.change_neurons
        elif update == 'seed':
            return self.update_seed
        elif update == 'reset':
            return self.reset_all
        elif update == 'truth':
            return self.toggle_truth
        elif update == 'probe':
            return self.update_probe

    # Define a function that changes the seed
    def update_seed(self, event):
        self.seed += 1
        self.passive(event)

    # Adjust the plot to make room for the sliders
    def adjust_plot(self):

        slider_thick = 0.03
        hor_slider_space = 0.02
        start_slider_shift = 0.12
        hor_label_space = 0.10
        hor_label_space_end = 0.09
        button_thick = 0.04
        button_length = 0.1
        sideplot_fraction = 0.25  # 0.5 = equal: [(1-x), x] * w
        hor_plot_space = 0.1

        hor_sliders = [slider for slider in self.sliders.values() if slider.orientation == 'horizontal']
        hor_sliders_right = [slider for slider in self.sliders_right.values() if slider.orientation == 'horizontal']

        # Even with few sliders, make space for the NN visualization
        bottom_space_sliders = (hor_slider_space + slider_thick) * max(len(hor_sliders), 7 )

        # Get all the sizes of the main plot
        bottom = max(hor_label_space + bottom_space_sliders + (
                    hor_slider_space + button_thick) * (len(self.buttons) > 0) + (hor_slider_space + button_thick) * (
                                 len(self.buttons) > 3), 0.1)
        left = 0.04
        top = 0.1
        right_space = 0.05
        height = 1 - top - bottom
        totwidth = 1 - right_space - left - hor_plot_space
        mainplot_width = totwidth * (1 - sideplot_fraction)
        sideplot_width = totwidth * sideplot_fraction

        # Set the size of the main plot
        self.ax.set_position([left, bottom, mainplot_width, height])

        # Set the position of the sideplot if it exists
        if not self.ax2 is None:
            # Change the size of the main figure
            self.fig.set_figwidth(self.w + hor_plot_space + sideplot_width + 1.0)

            # Set the position of the sideplot
            self.ax2.set_position(
                # [1 - right + ver_slider_space,
                [1 - right_space - sideplot_width,
                 bottom,
                 sideplot_width,
                 height])

        # Set the position of the horizontal sliders one by one
        for i, slider in enumerate(hor_sliders):
            slider.ax.set_position(
                [left + start_slider_shift,
                 bottom - hor_label_space - slider_thick - (hor_slider_space + slider_thick) * i,
                 mainplot_width - start_slider_shift - hor_label_space_end,
                 slider_thick])
        # Set the position of the horizontal sliders on the right
        for i, slider in enumerate(hor_sliders_right):
            slider.ax.set_position(
                [1 - right_space - sideplot_width,
                 bottom - hor_label_space - slider_thick - (hor_slider_space + slider_thick) * i,
                 sideplot_width,
                 slider_thick])

        # Calculate the spacing needed for 3 buttons
        n_button = 3
        button_space = (mainplot_width - n_button * button_length) / (n_button - 1)

        # Set the position of the buttons one by one
        for val, button in self.buttons.items():

            # Find the left side of the button
            if val == 'truth':
                i = 1
            elif val == 'seed':
                i = 2
            elif val == 'rerun':
                i = 0
            # Set the position of the button
            button.ax.set_position(
                [left + (button_space + button_length) * (i % n_button),
                 bottom - hor_label_space - bottom_space_sliders - (
                             hor_slider_space + button_thick) * (i // 3) - button_thick,
                 button_length,
                 button_thick])

        # Set the position of the radiobuttons one by one
        tot_height = 0
        for val, radiobutton in self.radio_buttons.items():
            # Get automatic size of button
            ll, bb, ww, hh = radiobutton.ax.get_position().bounds
            posy = bottom - hor_label_space - bottom_space_sliders - button_thick + tot_height  # Adjust starting height height

            if val == 'activation':
                i = 0

            radiobutton.ax.set_position(
                # [1 - right + ver_slider_space,  # posx
                [1 - right_space - sideplot_width + (button_space + button_length) * (i % n_button),  # posx
                 posy,  # posy
                 ww,  # width
                 hh])  # height
            tot_height += hh

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

        # Go through all current sliders, and store their current values
        for val, slider in self.sliders.items():

            # Check if the slider should return an integer
            if slider.valfmt == '%0.0f':
                kwargs[val] = round(slider.val)
            else:
                kwargs[val] = slider.val

        # Go through all current sliders, and store their current values
        for val, slider in self.sliders_right.items():

            # Check if the slider should return an integer
            if slider.valfmt == '%0.0f':
                kwargs[val] = round(slider.val)
            else:
                kwargs[val] = slider.val

        # Repeat for radio_buttons:
        for val, radiobutton in self.radio_buttons.items():
            kwargs[val] = radiobutton.value_selected

        # Add the current seed to the dictionary
        kwargs['seed'] = self.seed

        return kwargs

        # Define the function that will be called when the hide/show truth button is called

    def toggle_truth(self, event):
        kwargs = self.collect_kwargs()
        if self.plot_truth is None:
            self.plot_truth, = self.ax.plot(self.x_truth, self.y_truth, 'k-', label=self.truth_label.format(**kwargs))
            self.plot_true_loss, = self.ax2.plot(np.arange(len(self.rmse_true_ar)) * self.epochs_per_block,
                                                 self.rmse_true_ar, 'k-', label=self.true_loss_label.format(**kwargs))

            self.buttons['truth'].label.set_text('Hide truth')
        else:
            self.ax.lines.remove(self.plot_truth)
            self.ax2.lines.remove(self.plot_true_loss)
            self.plot_truth = None
            self.buttons['truth'].label.set_text('Show truth')

        # Update the legend
        if self.hide_legend:
            self.ax.legend = None
        else:
            self.ax.legend(loc='lower left')
            # self.ax2.legend(loc='upper right')
            self.ax2.legend(loc='lower left', bbox_to_anchor=(0.0, 0.95))
        # Disable autoscaling, to make sure the limits remain enforced
        plt.autoscale(False)

    def update_data(self, event):

        # Go through all sliders in the dictionary, and store their values in a kwargs dict
        kwargs = self.collect_kwargs()

        # Recompute the data and the truth
        self.x_data, self.y_data = self.f_data(**kwargs)
        self.y_truth = self.f_truth(self.x_truth, **kwargs)

        # Recompute dense
        self.y_dense = self.f_truth(self.x_dense, **kwargs) + np.random.normal(0, kwargs['epsilon'], len(self.x_dense))

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
            N_split = int(np.ceil(kwargs['N'] * (1 - kwargs['val_pct'] / 100))) # Round up, having 1 sample will guarentee 1 in training set
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
                self.plot_val, = self.ax.plot(self.x_val, self.y_val, 'x', color='purple',
                                              label=self.val_label.format(**kwargs))
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

    # Define a show function, so importing matplotlib is not strictly necessary in the notebooks
    # Shows plot without updating data
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


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.

    :usage:
        # >>> fig = plt.figure(figsize=(12, 12))
        # >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
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