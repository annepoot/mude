import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import plotly.graph_objects as go

def plotly_plot(df, total_idx=None, measure_locs=None, defect_loc_true=None, defect_loc_pred=None, defect_loc_pred_PCA=None):
    # Simply use the first beam if no index is given
    if total_idx is None:
        total_idx = 0

    # Extract data corresponding to one single bridge, and interpolate the displacements on a grid for plotting
    bar = df[df['sample'] == total_idx]
    grid_x, grid_y = np.mgrid[0.02:9.98:250j, 0.02:1.98:50j]
    grid_z = griddata(bar[['x','y']].to_numpy(), np.sqrt(bar['dx']**2 + bar['dy']**2), (grid_x, grid_y))

    # Plot displacement
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=grid_z.transpose(),
                             x=grid_x[:,0],
                             y=grid_y[0],
                             hoverinfo='skip',
                             name='heatmap'))

    # Compute the x and y coordinates in the measurement locations (or all locations, if measure_locs is not specified)
    if measure_locs is None:
        # Plot nodes
        fig.add_trace(go.Scatter(x=bar['x'],
                                 y=bar['y'],
                                 mode='markers',
                                 marker_color='black',
                                 hovertemplate='<b>Node</b>: %{text}',
                                 text=bar['node'],
                                 name=''))
    else:
        measure_coords = np.array([bar[bar['node'] == loc][['x','y']].to_numpy() for loc in measure_locs]).squeeze(1)
        fig.add_trace(go.Scatter(x = measure_coords[:,0],
                                 y = measure_coords[:,1],
                                 mode='markers',
                                 marker=dict(size=10, color='DarkSlateGrey', line=dict(width=2, color='white')),
                                 hovertemplate='<b>Node</b>: %{text}',
                                 text=measure_locs,
                                 name=''))

    # Add buttons to display different displacement fields
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=['z', [griddata(bar[['x','y']].to_numpy(), np.sqrt(bar['dx']**2 + bar['dy']**2), (grid_x, grid_y)).transpose()]],
                        label='magnitude', method='restyle'),
                    dict(
                        args=['z', [griddata(bar[['x','y']].to_numpy(), bar['dx'], (grid_x, grid_y)).transpose()]],
                        label='x',
                        method='restyle'),
                    dict(
                        args=['z', [griddata(bar[['x','y']].to_numpy(), bar['dy'], (grid_x, grid_y)).transpose()]],
                        label='y',
                        method='restyle')
                ]),
                direction='right', pad={'r': 10, 't': 10}, showactive=True, x=0.5, xanchor='left', y=1.2,
                yanchor='bottom', type='buttons', font=dict(size=13)
            ),
        ]
    )

    # Add annotation for button
    fig.add_annotation(dict(font=dict(size=13), x=0.5, y=1.23, showarrow=False,
                       xref='paper', yref='paper', xanchor='right', yanchor='bottom', text='Displacement: '))

    # Plot a vertical line at the true location, if specified
    if defect_loc_true is not None:
        fig.add_vline(x=defect_loc_true, name='truth', line=dict(color='Black'))
        fig.add_annotation(dict(font=dict(size=13), x=defect_loc_true, y=-.22, showarrow=False,
                           xref='x', yref='paper', text='truth: {:.2f}'.format(defect_loc_true)))

    # Plot a vertical line at the prediction location, if specified
    if defect_loc_pred is not None:
        if defect_loc_pred_PCA is None:
            text = 'prediction: {:.2f}'.format(defect_loc_pred)
        else:
            text = '5 sensor prediction: {:.2f}'.format(defect_loc_pred)

        fig.add_vline(x=defect_loc_pred, name='pred', line=dict(color='LightSlateGrey'))
        fig.add_annotation(dict(font=dict(size=13), x=defect_loc_pred, y=1.15, showarrow=False,
                           xref='x', yref='paper', text=text))

    # Plot a vertical line at the PCA prediction location, if specified
    if defect_loc_pred_PCA is not None:
        fig.add_vline(x=defect_loc_pred_PCA, name='pred_PCA', line=dict(color='LightSlateGrey'), line_dash='dot')
        fig.add_annotation(dict(font=dict(size=13), x=defect_loc_pred_PCA, y=1.25, showarrow=False,
                           xref='x', yref='paper', text='PCA prediction: {:.2f}'.format(defect_loc_pred_PCA)))

    # Update xaxis range and show figure
    fig.update_xaxes(range=(-0.2,10.2), constrain='domain')
    fig.update_yaxes(range=(-0.2,2.2), constrain='domain', scaleanchor='x', scaleratio=1)
    fig.show()


def format_colorbar_plot(fig, ax, plot, idcs):

    if hasattr(ax, '__iter__'):
        for i, axs in enumerate(ax):
            format_colorbar_plot(fig, axs, plot[i], idcs)

            if i > 0:
                axs.set_ylabel(None)

            titles = [r'true $y$', r'prediction $\hat y$', r'$|y - \hat y|$']
            if i < 3:
                axs.set_title(titles[i])

    else:
        lbound = np.min(plot.get_offsets().data, axis=0)
        ubound = np.max(plot.get_offsets().data, axis=0)
        xticks = np.linspace(lbound[0], ubound[0], 4)

        [ax.ticklabel_format(style='sci', axis=axis, scilimits=(0,0)) for axis in ['x','y']]
        ax.set_xlabel(rf"$u_{{{int(idcs[0]/2)+1}, {'x' if idcs[0]%2 == 0 else 'y'}}}$", fontsize=12)
        ax.set_ylabel(rf"$u_{{{int(idcs[1]/2)+1}, {'x' if idcs[1]%2 == 0 else 'y'}}}$", fontsize=12)
        ax.set_xticks(xticks)
        fig.colorbar(plot, ax=ax)
        ax.set_title("Defect location (color) as a function \n of two measurements")
