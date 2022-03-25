# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class magicplotter:
    
    # Define the default settings for all sliders
    defaults = {
        'eps':{
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
        'l':{
            'valmin':np.pi/8,
            'valmax':np.pi*8,
            'valinit':np.pi*2,
            'valfmt':None,
            'orientation':'horizontal',
            'label':r'Wave length ($l$)'
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
    def __init__(self, f_data, f_truth, f_pred, x_truth = None, x_pred = None, **settings):

        # Define a seed to make sure all results are reproducible
        self.seed = 0

        # Create an empty dictionary, which will later store all slider values
        self.sliders = {}
        self.buttons = {}
        
        # Collect all key word arguments
        kwargs = self.collect_kwargs()

        # Store all variables that were passed to the init function
        self.f_data = f_data
        self.f_truth = f_truth
        self.f_pred = f_pred
        self.x_data, self.y_data = self.f_data(**kwargs)
        self.x_truth = self.x_data if x_truth is None else x_truth
        self.x_pred = self.x_truth if x_pred is None else x_pred
        self.y_truth = self.f_truth(self.x_truth, **kwargs)
        
        # Compute the prediction and training / validation errors
        pred = self.f_pred(self.x_data, self.y_data, self.x_pred, **kwargs)
        if hasattr(pred[0], "__len__"):
            self.y_pred = pred[0]
            self.train_mse = None if len(pred) <= 1 else pred[1]
            self.val_mse = None if len(pred) <= 2 else pred[2]
        else:
            self.y_pred = pred
            self.train_mse = None
            self.val_mse = None

        # Add the training / validation errors to the dictionary
        if not self.train_mse is None:
            kwargs['train_mse'] = self.train_mse
        if not self.val_mse is None:
            kwargs['val_mse'] = self.val_mse

        # Get additional settings like the original plot title and labels
        self.title = settings.get('title', None)
        self.data_label = settings.get('data_label', r'Noisy data $(x,t)$')
        self.truth_label = settings.get('truth_label', r'Ground truth $f(x)$')
        self.pred_label = settings.get('pred_label', r'Prediction $y(x)$, $k={k}$')
        
        # Create a figure and add the data, truth, and prediction
        self.fig, self.ax = plt.subplots(figsize=(self.w,self.h))
        self.data, = plt.plot(self.x_data, self.y_data, 'x', label=self.data_label.format(**kwargs))
        self.truth, = plt.plot(self.x_truth, self.y_truth, 'k-', label=self.truth_label.format(**kwargs))
        self.pred, = plt.plot(self.x_pred, self.y_pred, '-', label=self.pred_label.format(**kwargs))
        
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('t')
        self.ax.set_ylim((-2.5, 2.5))
        self.ax.legend(loc='lower left')

        # Update the title
        if self.title is None:
            self.ax.set_title(None)
        else:
            self.ax.set_title(self.title.format(**kwargs))

    # Define the update function that will be called when a slider is changed
    def update_plot(self, event):

        # Get the slider values
        # k = int(min(k_slider.val, N_slider.val))
        # N = int(N_slider.val)
        # l = l_slider.val
        # eps = eps_slider.val
        
        # Go through all sliders in the dictionary, and store their values in a kwargs dict
        kwargs = self.collect_kwargs()
        
        # for val, slider in self.sliders.items():
            
        #     # Check if the slider should return an integer
        #     if slider.valfmt == '%0.0f':
        #         kwargs[val] = int(slider.val)
        #     else:
        #         kwargs[val] = slider.val
        
        # # Add the current seed to the dictionary
        # kwargs['seed'] = self.seed

        # Recompute the data and the truth
        self.x_data, self.y_data = self.f_data(**kwargs)
        self.y_truth = self.f_truth(self.x_truth, **kwargs)
        
        # Recompute the prediction and training / validation errors
        pred = self.f_pred(self.x_data, self.y_data, self.x_pred, **kwargs)
        
        print(len(pred))
        
        if hasattr(pred[0], "__len__"):
            self.y_pred = pred[0]
            self.train_mse = None if len(pred) <= 1 else pred[1]
            self.val_mse = None if len(pred) <= 2 else pred[2]
        else:
            self.y_pred = pred
            self.train_mse = None
            self.val_mse = None

        # Add the training / validation errors to the dictionary
        if not self.train_mse is None:
            kwargs['train_mse'] = self.train_mse
        if not self.val_mse is None:
            kwargs['val_mse'] = self.val_mse

        # Update the ground truth and the data in the plots
        self.data.set_data(self.x_data, self.y_data)
        self.truth.set_data(self.x_truth, self.y_truth)
        self.pred.set_data(self.x_pred, self.y_pred)
        
        # Update the legend
        self.data.set_label(self.data_label.format(**kwargs))
        self.truth.set_label(self.truth_label.format(**kwargs))
        self.pred.set_label(self.pred_label.format(**kwargs))
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
        
        if self.truth.get_alpha() is None:
            self.truth.set_alpha(0)
            self.buttons['truth'].label.set_text('Show truth')
        else:
            self.truth.set_alpha(None)
            self.buttons['truth'].label.set_text('Hide truth')

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

        hor_sliders = [slider for slider in self.sliders.values() if slider.orientation=='horizontal']
        ver_sliders = [slider for slider in self.sliders.values() if slider.orientation=='vertical']
        
        # Get all the sizes of the main plot
        bottom = max(hor_label_space + (hor_slider_space + slider_thick) * len(hor_sliders) + (hor_slider_space + button_thick) * np.sign(len(self.buttons)), 0.1)
        left = max(ver_label_space + (ver_slider_space + slider_thick) * len(ver_sliders), 0.2)
        top = 0.1
        right = 0.1
        height = 1 - top - bottom
        width = 1 - right - left

        # self.fig.subplots_adjust(left=left, bottom=bottom)
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
        
        # Then, go through all current sliders, and store their current values
        for val, slider in self.sliders.items():
            
            # Check if the slider should return an integer
            if slider.valfmt == '%0.0f':
                kwargs[val] = int(slider.val)
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