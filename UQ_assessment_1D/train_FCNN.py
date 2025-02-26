#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:19:28 2024

@author: lorenzo piu
"""
# %% Imports
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os

from architectures import BayesianNN, FCNN
from _utils import load_training_data, f, plot_loss, build_model_folder

# %% Input parameters
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Process some input parameters for model configuration.")

# Define command-line arguments
parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
parser.add_argument('--n_neurons', type=int, default=128, help='Number of neurons')
parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
parser.add_argument('--tr_layers', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=100000, help='Batch size')
parser.add_argument('--seed', type=int, default=54, help='Random seed')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
input_dim  = args.input_dim
n_neurons  = args.n_neurons
n_layers   = args.n_layers
output_dim = args.output_dim
tr_layers  = args.tr_layers
n_epochs   = args.n_epochs
batch_size = args.batch_size
seed = args.seed

# %% Make model folder
model_folder = build_model_folder(architecture='FCNN', 
                                   input_dim=input_dim, 
                                   n_neurons=n_neurons,
                                   n_layers=n_layers,
                                   output_dim=output_dim,
                                   tr_layers=tr_layers,
                                   n_epochs=n_epochs,
                                   batch_size=batch_size,
                                   seed=seed
                                   )

# %% Load training data and initialize model
# Load training and testing data
data = load_training_data(verbose=True)

# Model, optimizer
model = FCNN(input_dim, n_layers, n_neurons, output_dim, tr_layers)
model._initialize_weights(seed=seed)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.fit(data, epochs=n_epochs, batch_size=batch_size)

# plot_loss(model.train_loss, model.test_loss)

torch.save(model, os.path.join(model_folder, 'model.pth'))

# %% Plots
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],  # or use ["DejaVu Serif"] if Times New Roman is unavailable
    "font.size": 16,                    # General font size
    "axes.titlesize": 20,               # Title font size
    "axes.labelsize": 18,               # Label font size
    "legend.fontsize": 14,              # Legend font size
    "xtick.labelsize": 16,              # X-tick label size
    "ytick.labelsize": 16,              # Y-tick label size
})

plt.figure(figsize=(8, 5), dpi=600)
plt.scatter(data['x_train'].numpy().flatten()        , data['y_train'].numpy().flatten()       , c='#440154', marker='.', s=7, alpha=0.01)
plt.scatter([]                         , []                        , c='#440154', marker='.', s=70, alpha=1  , label='Training/Validation points')
plt.scatter(data['x_extr'].numpy().flatten()   , data['y_extr'].numpy().flatten()  , c='#35B779', marker='.', s=7, alpha=0.01)
plt.scatter([]                         , []                        , c='#35B779', marker='.', s=70, alpha=1  , label='Extrapolation points')
plt.scatter(data['x_interp'].numpy().flatten() , data['y_interp'].numpy().flatten(), c='#FDE725', marker='.', s=7, alpha=0.01)
plt.scatter([]                         , []                        , c='#FDE725', marker='.', s=70, alpha=1  , label='Interpolation points')

# Line plot of the function
steps       = 1000
x_ptp       = 2
x_lin       = torch.linspace(-1-x_ptp/4, 1+x_ptp/4, steps)
x_lin       = torch.reshape(x_lin, [len(x_lin), 1])
y_lin       = f(x_lin, noise=0)
plt.plot(x_lin, y_lin, c='red', linewidth=2.5, label='y = f(x)')

# Line plot of the model
model.to('cpu')
with torch.no_grad():
    y_model = model.forward(x_lin)
plt.plot(x_lin, y_model, c='cyan', linewidth=2.5, label='FCNN')

# Labels, legend, and title
plt.xlabel('x')
plt.ylabel('y')
# plt.title('2D Histogram with Function Overlay')
plt.legend(loc='lower right')
plt.grid(visible=True, linestyle='--', alpha=0.6)

plt.xlim(x_lin.min(), x_lin.max())

# Tight layout for a cleaner look
plt.tight_layout()

plt.savefig(os.path.join(model_folder, 'model_output.png'))

# Show plot
# plt.show()


