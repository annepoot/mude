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
            'label':r'Noise ($\varepsilon$)'
        },
        'k':{
            'valmin':1,
            'valmax':100,
            'valinit':1,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'Neighbors ($k$)'
        },
        'N':{
            'valmin':1,
            'valmax':1000,
            'valinit':100,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'Training size ($N$)'
        },
        'freq':{
            'valmin':1/8,
            'valmax':8,
            'valinit':1,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Frequency ($freq$)'
        },
        'l':{
            'valmin':1/5,
            'valmax':5,
            'valinit':1,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Length scale ($l$)'
        },
        'degree':{
            'valmin':1,
            'valmax':30,
            'valinit':1,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'Degree ($k$)'
        },
        'N_radial':{
            'valmin':1,
            'valmax':30,
            'valinit':5,
            'valfmt':'%0.0f',
            'orientation':'horizontal',
            'label':r'Number of RBFs ($N$)'
        },
        'l_radial':{
            'valmin':1/5,
            'valmax':5,
            'valinit':1,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Length scale ($l$)'
        },
        'val_pct':{
            'valmin':0,
            'valmax':50,
            'valinit':0,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Validation size ($\%$)'
        },
        'truth':{
            'index':1,
            'hovercolor':'0.975',
            'label':'Hide truth'
        },
        'seed':{
            'index':2,
            'hovercolor':'0.975',
            'label':'New seed'
        },
        'reset':{
            'index':3,
            'hovercolor':'0.975',
            'label':'Reset'
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
        self.x_pred = self.x_data if x_pred is None else x_pred
        self.x_truth = self.x_pred if x_truth is None else x_truth
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
            x_shuffle, y_shuffle = shuffle(self.x_data, self.y_data, random_state=self.seed)
            N_split = int(kwargs['N'] * (1-kwargs['val_pct']/100))
            self.x_train = x_shuffle[:N_split]
            self.y_train = y_shuffle[:N_split]
            self.x_val = x_shuffle[N_split:]
            self.y_val = y_shuffle[N_split:]
    
        # Compute the prediction in the prediction locations
        self.y_pred = self.f_pred(self.x_train, self.y_train, self.x_pred, **kwargs)
        
        # Compute the training and validation errors
        train_pred = self.f_pred(self.x_train, self.y_train, self.x_train, **kwargs)
        self.mse_train = sum((self.y_train - train_pred)**2) / len(self.x_train)

        if self.x_val is None:
            self.mse_validation = 0
        else:
            val_pred = self.f_pred(self.x_train, self.y_train, self.x_val, **kwargs)
            self.mse_validation = sum((self.y_val - val_pred)**2) / len(self.x_val)

        # Add the training / validation errors to the dictionary
        kwargs['mse_train'] = self.mse_train
        kwargs['mse_validation'] = self.mse_validation

        # Get additional settings like the original plot title and labels
        self.title = settings.get('title', None)
        self.data_label = settings.get('data_label', r'Training data $(x,t)$')
        self.truth_label = settings.get('truth_label', r'Ground truth $f(x)$')
        self.pred_label = settings.get('pred_label', r'Prediction $y(x)$, $k={k}$')
        self.val_label = settings.get('val_label', r'Validation data $(x,t)$')
        
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
        
        if self.x_val is None:
            self.plot_val = None
        else:
            self.plot_val, = self.ax.plot(self.x_val, self.y_val, 'x', color='purple', label=self.val_label.format(**kwargs))

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('t')
        self.ax.set_ylim((-2.5, 2.5))
        
        # Check if the legend should be shown, and plot it if so
        self.hide_legend = settings.get('hide_legend', False)
        
        if self.hide_legend:
            self.ax.legend = None
        else:
            self.ax.legend(loc='lower left')

        # Update the title
        if self.title is None:
            self.ax.set_title(None)
        else:
            self.ax.set_title(self.title.format(**kwargs))
            
        # Initialize the sidebar axes as None (to make sure that it is defined)
        self.ax_mse = None

    # Define the update function that will be called when a slider is changed
    def update_plot(self, event):

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
            N_split = int(kwargs['N'] * (1-kwargs['val_pct']/100))
            self.x_train = x_shuffle[:N_split]
            self.y_train = y_shuffle[:N_split]
            self.x_val = x_shuffle[N_split:]
            self.y_val = y_shuffle[N_split:]
    
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

        # Update the ground truth and the data in the plots
        self.plot_data.set_data(self.x_train, self.y_train)
        self.plot_pred.set_data(self.x_pred, self.y_pred)

        if not self.plot_truth is None:
            self.plot_truth.set_data(self.x_truth, self.y_truth)

        if self.x_val is None:
            if not self.plot_val is None:
                self.ax.lines.remove(self.plot_val)
                self.plot_val = None
        else:
            if self.plot_val is None:
                self.plot_val, = self.ax.plot(self.x_val, self.y_val, 'x', color='purple', label=self.val_label.format(**kwargs))
            else:
                self.plot_val.set_data(self.x_val, self.y_val)
            
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
        self.plot_data.set_label(self.data_label.format(**kwargs))
        self.plot_pred.set_label(self.pred_label.format(**kwargs))
        
        if not self.plot_truth is None:
            self.plot_truth.set_label(self.truth_label.format(**kwargs))

        if not self.plot_val is None:
            self.plot_val.set_label(self.val_label.format(**kwargs))
        
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
    
    # Define the function that will be called when the hide/show truth button is called
    def toggle_truth(self, event):
        
        if self.plot_truth is None:
            self.plot_truth, = self.ax.plot(self.x_truth, self.y_truth, 'k-', label=self.truth_label.format(**self.collect_kwargs()))
            self.buttons['truth'].label.set_text('Hide truth')
        else:
            self.ax.lines.remove(self.plot_truth)
            self.plot_truth = None
            self.buttons['truth'].label.set_text('Show truth')

        self.update_plot(event)        
    
    # Define a function that changes the seed
    def update_seed(self, event):
        
        self.seed += 1
        
        self.update_plot(event)
    
    # Define a function that performs a reset
    def reset_all(self, event):
        
        # Reset the seed
        self.seed = 0
        
        # Reset all sliders
        for slider in self.sliders.values():
            slider.reset()

        # Show the truth again if necessary
        if not self.truth.get_alpha() is None:
            self.toggle_truth(event)
    
        self.update_plot(event)

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

        # Add an event to the slider
        slider.on_changed(self.update_plot)

        # Adjust the plot to make room for the added slider
        self.adjust_plot()
        
    # Add a button to the bottom or left side of the plot
    def add_button(self, var, **settings):

        # Check if the variable is in defaults
        def_settings = self.defaults.get(var, {})

        # Load all default/given values
        hovercolor = settings['hovercolor'] if 'hovercolor' in settings else def_settings['hovercolor']
        label = settings['label'] if 'label' in settings else def_settings['label']

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

        # Add an event to the slider
        if var == 'truth':
            button.on_clicked(self.toggle_truth)
        elif var == 'seed':
            button.on_clicked(self.update_seed)
        elif var == 'reset':
            button.on_clicked(self.reset_all)

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
        bottom = max(hor_label_space + (hor_slider_space + slider_thick) * len(hor_sliders) + (hor_slider_space + button_thick) * (len(self.buttons) > 0), 0.1)
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

            # Set the position of the button
            button.ax.set_position(
                [left + (button_space + button_length) * i,
                 bottom - hor_label_space - (hor_slider_space + slider_thick) * len(hor_sliders) - button_thick,
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

    # Define a show function, so importing matplotlib is not strictly necessary in the notebooks
    def show(self):
        
        # Update the plot
        self.update_plot(0)

        # Forward to plt.show()
        plt.show()