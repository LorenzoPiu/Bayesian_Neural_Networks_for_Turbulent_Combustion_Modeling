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
from architectures import BayesianNN, FCNN, DeepEnsemble, MC_Dropout
from _utils import load_training_data, f, plot_loss, build_model_folder

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
parser.add_argument('--n_ensembles', type=int, default=10, help='Number of ensembles')
parser.add_argument('--n_samples', type=int, default=100, help='Number of samples')
parser.add_argument('--dropout_prob', type=float, default=0.02, help='Dropout probability')
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
n_samples  = args.n_samples
dropout_prob = args.dropout_prob
seed = args.seed

# %% Build model folder
model_folder = build_model_folder(architecture='DEN_MCD', 
                   input_dim=input_dim, 
                   n_neurons=n_neurons,
                   n_layers=n_layers,
                   output_dim=output_dim,
                   tr_layers=tr_layers,
                   n_epochs=n_epochs,
                   batch_size=batch_size,
                   dropout_prob=dropout_prob,
                   n_samples=n_samples,
                   n_ensembles=n_ensembles,
                   seed=seed
                   )

# %% Load training data and initialize model
# Load training and testing data
data = load_training_data(verbose=True)

# Model
model = DeepEnsemble(MC_Dropout, n_ensembles, input_dim, n_layers, n_neurons, output_dim, tr_layers=tr_layers, dropout_prob=dropout_prob)

# %% Train
t = time.time()
for i ,m in enumerate(model.models):
    print(f"\nTraining model {i}")
    m.fit_bayes(data, 
                epochs=n_epochs, 
                n_samples=n_samples, 
                batch_size=batch_size
                )

t = time.time()-t
model.training_time = t

# Save model
torch.save(model, os.path.join(model_folder, 'model.pth'))


# %% Plot 2
from my_plot_utils import set_plotting_preferences
from matplotlib import gridspec
from scipy.stats import gaussian_kde

x_ptp = 2
steps = 1000

x_lin = torch.linspace(-1 - x_ptp / 4, 1 + x_ptp / 4, steps)
x_lin = torch.reshape(x_lin, [len(x_lin), 1])
y_lin = f(x_lin, noise=0)

# Evaluate the model on the x points
y_model_mean        = 0.0
mean_square_output = 0.0
for NN in model.models:
    NN.to('cpu')
    for _ in range(n_samples):
        NN.eval()
        with torch.no_grad():
            out = NN.forward(x_lin, smooth=True)
        y_model_mean += out
        mean_square_output += out ** 2 # needed for the incremental computation of std.
        # math behind the formula: http://datagenetics.com/blog/november22017/index.html
    
y_model_mean /= (n_samples*n_ensembles)
mean_square_output /= (n_samples*n_ensembles)
# Calculate standard deviation
y_model_sigma = torch.sqrt(mean_square_output - y_model_mean ** 2)

set_plotting_preferences()

# Set up the grid for two plots (one for the KDE plot above, and the main plot below)
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[0.3, 1])  # 0.3 for KDE plot, 1 for main plot

# First axis (for the KDE plot)
ax0 = plt.subplot(gs[0])

# KDE plot with normalized density and custom color
#sns.kdeplot(data['x_train'].numpy(), color='#FFC867', fill=True, alpha=0.3, ax=ax0)
kde = gaussian_kde(data['x_train'].numpy().flatten(), bw_method='silverman')  # You can adjust bw_method as needed
kde_values = kde(x_lin.flatten())
kde_values /= kde_values.max()
ax0.fill_between(x_lin.numpy().flatten(), 0, kde_values, color='#FFC867', alpha=0.6)
ax0.plot(x_lin.numpy().flatten(), kde_values, color='#FFC867', alpha=0.4)

# fill plot for model sigma
ax0.fill_between(x_lin.flatten(), 0, y_model_sigma.flatten()/y_model_sigma.max(), color='blue', alpha=0.25)
ax0.plot(x_lin.flatten(), y_model_sigma.flatten()/y_model_sigma.max(), color='blue', alpha=0.4)

# Hide y-axis and ticks for the KDE plot
ax0.get_yaxis().set_visible(False)  # Hide the y-axis
ax0.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
ax0.set_xticklabels([])  # Remove the x-ticks from the top plot
ax0.set_yticks([])  # Remove the y-ticks from the top plot
ax0.legend().remove()
ax0.set_ylim(bottom=0)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['bottom'].set_visible(True)  # Ensure the bottom spine (x-axis) is visible

ax0.set_xlim(x_lin.min(), x_lin.max())

# Second axis (for the main plot)
ax1 = plt.subplot(gs[1])

# Scatter plot of training data
ax1.scatter(data['x_train'].numpy().flatten(), data['y_train'].numpy().flatten(), c='#f0f0f0', marker='o', s=7, alpha=0.08, edgecolors='#FFC867')
ax1.scatter([], [], c='#FFC867', edgecolors='#FFC867', marker='.', s=90, alpha=1, label='Training/Validation points')

# Line plot of the function
ax1.plot(x_lin, y_lin, c='#da0003', linewidth=2.5, label='y = f(x)')

# Line plot of the model
ax1.plot(x_lin, y_model_mean, c='blue', linestyle='--', linewidth=2, label='mean prediction')

# Confidence interval
ax1.fill_between(
    x_lin.flatten(),
    y_model_mean.flatten() - 2 * y_model_sigma.flatten(),  # Lower bound
    y_model_mean.flatten() + 2 * y_model_sigma.flatten(),  # Upper bound
    color="blue",
    alpha=0.25,
    label="95% confidence interval"
)

# make only the left and bottom lines visible to simplify the plot
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(True)
ax1.spines['bottom'].set_visible(True)  # Ensure the bottom spine (x-axis) is visible

# Labels, legend, and title for the main plot
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(loc='upper center')
# ax1.grid(visible=True, linestyle='--', alpha=0.6)

ax1.set_xlim(x_lin.min(), x_lin.max())
ax1.set_ylim(-2, 4.7)
ax1.set_yticks([0, 2, 4])

# Tight layout for a cleaner look
plt.tight_layout()

# Save the figure with both plots
plt.savefig(os.path.join(model_folder, 'model_output.png'), dpi=600)

# Show the plot
plt.show()