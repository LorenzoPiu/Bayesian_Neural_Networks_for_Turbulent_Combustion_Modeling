#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:48:03 2024

@author: lorenzo piu
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import argparse

from architectures import BayesianNN, FCNN, DeepEnsemble, inference
from _utils import load_training_data, f, plot_loss, build_model_folder
from my_plot_utils import plot_model_output

# %% Input paramzeters
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
parser.add_argument('--n_ensembles', type=int, default=2, help='Number of ensembles')
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
n_ensembles = args.n_ensembles
seed = args.seed

# %% Build model folder
model_folder = build_model_folder(architecture='DEN', 
                   input_dim=input_dim, 
                   n_neurons=n_neurons,
                   n_layers=n_layers,
                   output_dim=output_dim,
                   tr_layers=tr_layers,
                   n_epochs=n_epochs,
                   batch_size=batch_size,
                   n_ensembles=n_ensembles,
                   seed=seed
                   )

# %% Load training data and initialize model
# Load training and testing data
data = load_training_data(verbose=True)

# Model, optimizer
model = DeepEnsemble(FCNN, n_ensembles, input_dim, n_layers, n_neurons, output_dim, tr_layers=tr_layers)

t = time.time()
model.fit(data, epochs=n_epochs, batch_size=batch_size)
t = time.time()-t
model.training_time = t

torch.save(model, os.path.join(model_folder, 'model.pth'))

# %% Plots
# Line plot of the function
steps       = 1000
x_ptp       = 2
x_lin       = torch.linspace(-1-x_ptp/4, 1+x_ptp/4, steps)
x_lin       = torch.reshape(x_lin, [len(x_lin), 1])
y_lin       = f(x_lin, noise=0)
plt.plot(x_lin, y_lin, c='red', linewidth=2.5, label='y = f(x)')

# Plot mean and uncertainty
for NN in model.models:
    NN.to(torch.device('cpu'))
y_model_mean, y_model_sigma = model.predict(x_lin)

plot_model_output(y_model_mean, y_model_sigma, data, model_folder)
