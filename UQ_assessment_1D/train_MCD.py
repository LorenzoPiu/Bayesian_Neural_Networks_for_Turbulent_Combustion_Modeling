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
from matplotlib import gridspec
import seaborn as sns
import os
import time
from architectures import BayesianNN, FCNN, MC_Dropout
import argparse

from _utils import load_training_data, f, plot_loss, set_seed, build_model_folder
from my_plot_utils import plot_model_output
from scipy.stats import gaussian_kde

# %% Input parameters
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Neural Network architecture and training parameters.")

# Define command-line arguments
parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
parser.add_argument('--n_neurons', type=int, default=128, help='Number of neurons')
parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
parser.add_argument('--tr_layers', type=int, default=2, help='Number of transition layers')
parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
parser.add_argument('--dropout_prob', type=float, default=0.02, help='Dropout probability')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

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
dropout_prob = args.dropout_prob
seed = args.seed

# %% Initialize model folder
model_folder = build_model_folder(architecture='MCD', 
                   input_dim=input_dim, 
                   n_neurons=n_neurons,
                   n_layers=n_layers,
                   output_dim=output_dim,
                   tr_layers=tr_layers,
                   n_epochs=n_epochs,
                   batch_size=batch_size,
                   dropout_prob=dropout_prob,
                   seed=seed
                   )

# %% Load training data and initialize model
# Load training and testing data
set_seed(seed)

data = load_training_data(verbose=True)

# Model, optimizer
model = MC_Dropout(input_dim, n_layers, n_neurons, output_dim, tr_layers, dropout_prob=dropout_prob)
model._initialize_weights(seed=seed)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

t = time.time()
model.fit(data, epochs=n_epochs, batch_size=batch_size)
t = time.time() - t
model.training_time = t

#fig = plot_loss(model.train_loss, model.test_loss, return_fig=True)
#fig.savefig(os.path.join(model_folder, 'training.png'))

torch.save(model, os.path.join(model_folder, 'model.pth'))

# %% Plots
# Evaluate the model
set_seed(seed)  # reproducibility
steps = 5000
x_ptp = 2
x_lin = torch.linspace(-1 - x_ptp / 4, 1 + x_ptp / 4, steps)
x_lin = torch.reshape(x_lin, [len(x_lin), 1])
y_lin = f(x_lin, noise=0)
model = model.to('cpu')
with torch.no_grad():
    y_model_mean, y_model_sigma = model.inference(x_lin, n_samples=2000)

plot_model_output(y_model_mean, y_model_sigma, data, model_folder, x_ptp=x_ptp, steps=steps)
