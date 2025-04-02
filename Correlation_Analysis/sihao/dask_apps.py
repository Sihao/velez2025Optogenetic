from dash import dcc
from dash import html
from dash import Input, Output, callback
from dash import Dash
import plotly.graph_objs as go


def show_responsive_cells(centroids, responsive_cells):
    """Dash app to show the positions of all the cells in 3D and highlight the responsive cells for a given stimulus.

    Parameters
    ----------
    centroids : array
        Array with shape (n_neurons, 3). The positions of each neuron.
    responsive_cells : dict
        Dictionary with keys the stimulus ids and values the indices of the responsive cells for that stimulus.

    Returns
    -------
    dash.Dash
        Dash app object

    """
    # Create app object
    app = Dash(__name__)

    # Define layout



