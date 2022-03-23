# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def base_plot(x, t, x_truth, y_truth, x_pred, y_pred):
    
    # Create a figure and add the data, truth, and prediction
    fig, ax = plt.subplots(figsize=(8,6))
    data, = plt.plot(x, t, 'x', label=r'Noisy data $(x,t)$')
    truth, = plt.plot(x_truth, y_truth, 'k-', label=r'Ground truth $f(x)$')
    pred, = plt.plot(x_pred, y_pred, '-', label=r'Prediction $y(x)$, $k=1$')
        
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_ylim((-2.5, 2.5))
    fig.legend(loc='lower left')
    
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
        }
    }
    
    # Store the width, height and aspect ratio of the plot
    # The ratio is necessary to account for the fact that the plot is not 1:1
    w = 8
    h = 6
    r = h / w

    def __init__(self, x_data, y_data, x_truth, y_truth, x_pred, y_pred):

        # Create a figure and add the data, truth, and prediction
        self.fig, self.ax = plt.subplots(figsize=(self.w,self.h))
        self.data, = plt.plot(x_data, y_data, 'x', label=r'Noisy data $(x,t)$')
        self.truth, = plt.plot(x_truth, y_truth, 'k-', label=r'Ground truth $f(x)$')
        self.pred, = plt.plot(x_pred, y_pred, '-', label=r'Prediction $y(x)$, $k=1$')
            
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('t')
        self.ax.set_ylim((-2.5, 2.5))
        plt.legend(loc='lower left')
            
        # Store the sliders and buttons in lists
        self.hor_sliders = []
        self.ver_sliders = []
        self.buttons = []
        
    def add_slider(self, var, update, **settings):
        
        # Check if the variable is in defaults
        def_settings = self.defaults.get(var, {})

        # Load all default/given values
        valmin = settings['valmin'] if 'valmin' in settings else def_settings['valmin']
        valmax = settings['valmax'] if 'valmax' in settings else def_settings['valmax']
        valinit = settings['valinit'] if 'valinit' in settings else def_settings['valinit']
        valfmt = settings['valfmt'] if 'valfmt' in settings else def_settings['valfmt']
        orientation = settings['orientation'] if 'orientation' in settings else def_settings['orientation']
        label = settings['label'] if 'label' in settings else def_settings['label']
        
        # Create the slider
        ax_slider = self.fig.add_axes([0., 0., 0.1, 0.1])
        slider = Slider(
            ax=ax_slider,
            label=label,
            valmin=valmin,
            valmax=valmax,
            valinit=valinit,
            valfmt=valfmt,
            orientation=orientation
        )

        # Store the slider
        if orientation == 'horizontal':
            self.hor_sliders.append(slider)
        elif orientation == 'vertical':
            self.ver_sliders.append(slider)

        # Adjust the plot to make room for the added slider
        self.adjust_plot()
        
        # Add an event to the slider
        slider.on_changed(update)

    # Adjust the plot to make room for the sliders
    def adjust_plot(self):
        
        r = self.r
        hor_sliders = self.hor_sliders
        ver_sliders = self.ver_sliders

        # Make room for the sliders
        bottom = 0.1 + 0.05 * len(hor_sliders) if len(hor_sliders) > 0 else 0
        left = (0.15 + 0.125 * len(ver_sliders)) * r if len(ver_sliders) > 0 else 0
        
        self.fig.subplots_adjust(left=left, bottom=bottom)

        # Set the position of the horizontal sliders one by one
        for i, slider in enumerate(hor_sliders):
            
            # Set the position of the slider
            slider.ax.set_position([left, bottom - 0.15 - 0.05 * i, 0.65, 0.03])
    
        # Set the position of the vertical sliders one by one
        for i, slider in enumerate(ver_sliders):

            # Set the position of the slider
            slider.ax.set_position([left - (0.15 + 0.125 * i) * r, bottom, 0.03 * r, 1 - bottom - 0.15])
    
    
    
    