#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:31:22 2024

@author: lorenzo piu
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os

def f(x, noise=0.2):
    return 0.5*np.sin(12*x) - 1*np.cos(4*x) + 0.2 * x * np.sin(9*x) + 0.1*np.cos(24*x+1) + noise*torch.randn(*x.shape)

def scale_vector(v, a, b):
    # Build x, y, x_train, y_train, x_test, y_test
    # Convert the input to a NumPy array
    v = np.array(v)
    
    # Find the minimum and maximum of the original vector
    x_min = np.min(v)
    x_max = np.max(v)
    
    # Scale the vector
    scaled_v = a + (v - x_min) * (b - a) / (x_max - x_min)
    
    return scaled_v

def load_training_data(folder='training_data', verbose=False):
    """
    Loads training data and parameters from the specified folder.
    
    Args:
        folder (str): The folder where the training data is saved.
    
    Returns:
        dict: A dictionary containing training data, test data, and parameters.
    """
    import torch
    import pickle
    import os

    # Initialize a dictionary to hold all the data and parameters
    data = {}
    
    try:
        # Load parameters
        with open(os.path.join(folder, 'params.pkl'), 'rb') as file:
            data['params'] = pickle.load(file)

        # Load tensors
        data['x']        = torch.load(os.path.join(folder, 'x.pt'), weights_only=True)
        data['y']        = torch.load(os.path.join(folder, 'y.pt'), weights_only=True)
        data['x_train']  = torch.load(os.path.join(folder, 'x_train.pt'), weights_only=True)
        data['x_val']    = torch.load(os.path.join(folder, 'x_val.pt'), weights_only=True)
        data['x_extr']   = torch.load(os.path.join(folder, 'x_extr.pt'), weights_only=True)
        data['x_interp'] = torch.load(os.path.join(folder, 'x_interp.pt'), weights_only=True)
        data['y_train']  = torch.load(os.path.join(folder, 'y_train.pt'), weights_only=True)
        data['y_val']    = torch.load(os.path.join(folder, 'y_val.pt'), weights_only=True)
        data['y_extr']   = torch.load(os.path.join(folder, 'y_extr.pt'), weights_only=True)
        data['y_interp'] = torch.load(os.path.join(folder, 'y_interp.pt'), weights_only=True)
        
        if verbose:
            print("Training data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")

    return data
    
def ordered_indices_generator(total_size, batch_size):
    """Yields ordered batch indices from 0 to total_size."""
    for start_idx in range(0, total_size, batch_size):
        end_idx = min(start_idx + batch_size, total_size)
        yield list(range(start_idx, end_idx))

def plot_loss(train_loss, test_loss=None, return_fig=False, log=True):
    """
    Plots the training and optional test loss over epochs.
    
    Parameters:
    - train_loss (list or array-like): The loss values for the training dataset over epochs.
    - test_loss (list or array-like, optional): The loss values for the test dataset over epochs.
    
    Returns:
    - None: Displays the plot.
    """
    set_plotting_preferences()
    
    epochs = range(1, len(train_loss) + 1)  # Epoch numbers
    
    plt.figure(figsize=(7, 5), dpi=600)
    
    # Plot training loss
    plt.plot(epochs, train_loss, label="Training Loss", linestyle='-', linewidth=2)
    
    # Plot test loss if provided
    if test_loss is not None:
        plt.plot(epochs, test_loss, label="Test Loss", linestyle='--', linewidth=2)
    
    # Add titles and labels
    # plt.title("Loss Over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    if log:
        plt.yscale('log')
    plt.tight_layout()
    
    fig = plt.gcf()
    
    # Show the plot
    plt.show()
    
    if return_fig:
        return fig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_plotting_preferences():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],  # or use ["DejaVu Serif"] if Times New Roman is unavailable
        "font.size": 20,                    # General font size
        "axes.titlesize": 26,               # Title font size
        "axes.labelsize": 25,               # Label font size
        "legend.fontsize": 14,              # Legend font size
        "xtick.labelsize": 20,              # X-tick label size
        "ytick.labelsize": 20,              # Y-tick label size
    })

def generate_n_layers(input_dim, n_layers, n_neurons, output_dim, tr_layers=0):
    # Define the architecture based on the initialisation logic
    tr_layers = tr_layers + 1
    n = [input_dim]
    for i in range(tr_layers - 1):
        n.append(input_dim + (n_neurons - input_dim) * (i + 1) // tr_layers)
    for _ in range(n_layers):
        n.append(n_neurons)
    for i in range(tr_layers - 1):
        n.append(output_dim + (n_neurons - output_dim) * (tr_layers - i - 1) // tr_layers)
    n.append(output_dim)
    
    return n

# Define a dictionary that maps extended parameter names to their shortened versions.
# Used to build the model folder
param_map = {
    "input_dim"         : "i",
    "n_neurons"         : "nn",
    "n_layers"          : "nl",
    "output_dim"        : "o",
    "tr_layers"         : "tr",
    "n_epochs"          : "ep",
    "n_epochs_pretrain" : "epr",
    "batch_size"        : "b",
    "n_ensembles"       : "ne",
    "dropout_prob"      : "dr",
    "prior_std"         : "pr",
    "kl_weight"         : "kl",
    "n_samples"         : "ns",
    "seed"              : "s"
}

def build_model_folder(models_folder='models', architecture=None, mkdir=True, **params):
    # Default folder structure base
    models_path = 'models'
    if architecture is not None:
        model_path = os.path.join(models_path, architecture)
    
    # Initialize a list to hold parts of the folder name
    folder_parts = []
    
    # This loop ensures the order in which the parameters are passed is maintained
    # Convert extended names to shortened names using the param_map
    # for extended_name, value in params.items():
    #     if value is not None:  # Ignore parameters with None value
    #         # Find the shortened name for this extended name
    #         shortened_name = param_map.get(extended_name)
    #         if shortened_name is not None:
    #             folder_parts.append(f"{shortened_name}{value}")
    #         else:
    #             raise ValueError(f'{extended_name} is not a valid variable. Change the valid parameters in _utils.param_map')
    
    # This loop follows the order in which the parameters appear in the param_map dictionary.
    # In this case we need to check that the parameters passed are valid entries
    for extended_name, value in params.items():
        if extended_name not in param_map:
            raise ValueError(f'{extended_name} is not a valid variable. Change the valid parameters in _utils.param_map')
    for extended_name in param_map:
        if extended_name in params and params[extended_name] is not None:
            value = params[extended_name]
            shortened_name = param_map[extended_name]
            folder_parts.append(f"{shortened_name}{value}")
        else:
            # If the parameter is missing or has value None, skip it
            continue
    
    # Combine the parts to create the model folder name
    model_folder = os.path.join(model_path, "_".join(folder_parts))
    
    if mkdir:
        # Create the folder structure if it doesn't already exist
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        

    return model_folder
