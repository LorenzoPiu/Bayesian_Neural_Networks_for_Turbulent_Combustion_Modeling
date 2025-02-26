#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:02:28 2025

@author: lpiu
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import argparse
from architectures import BayesianNN, FCNN, DeepEnsemble, MC_Dropout
from _utils import load_training_data, f, plot_loss, build_model_folder
from my_plot_utils import plot_model_output

# %% Input parameters
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Neural Network architecture and training parameters.")

# Define command-line arguments
parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
parser.add_argument('--n_neurons', type=int, default=64, help='Number of neurons')
parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
parser.add_argument('--tr_layers', type=int, default=1, help='Number of transition layers')
parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=50000, help='Batch size')
parser.add_argument('--n_ensembles', type=int, default=100, help='Number of ensembles')
parser.add_argument('--n_samples', type=int, default=100, help='Number of samples')
parser.add_argument('--dropout_prob', type=float, default=0.01, help='Dropout probability')
parser.add_argument('--seed', type=int, default=36, help='Random seed')

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
n_samples  = args.n_samples
dropout_prob = args.dropout_prob
kl_weight = 0.000
prior_std = 0.0003
seed = args.seed

# %% Build model folder
model_folder = build_model_folder(architecture='BNN', mkdir=False,
                   input_dim=input_dim, 
                   n_neurons=n_neurons,
                   n_layers=n_layers,
                   output_dim=output_dim,
                   tr_layers=tr_layers,
                   n_epochs=n_epochs,
                   n_epochs_pretrain=400,
                   batch_size=batch_size,
                   # dropout_prob=dropout_prob,
                   kl_weight=kl_weight,
                   prior_std=prior_std,
                   # n_samples=n_samples,
                   # n_ensembles=n_ensembles,
                   seed=seed
                   )

# %% Load training data and initialize model
# Load training and testing data
data = load_training_data(verbose=True)

# Model
model = torch.load(os.path.join(model_folder, 'model.pth'))

# %% Plot 2
from my_plot_utils import set_plotting_preferences
from matplotlib import gridspec
from scipy.stats import gaussian_kde

x_ptp = 2
steps = 1000

x_lin = torch.linspace(-1 - x_ptp / 4, 1 + x_ptp / 4, steps)
x_lin = torch.reshape(x_lin, [len(x_lin), 1])
y_lin = f(x_lin, noise=0)

from architectures import inference
y_model_mean, y_model_sigma = inference(model, x_lin, gpu=False, n_samples=100)

plot_model_output(y_model_mean, y_model_sigma, data, model_folder+'_prova')
