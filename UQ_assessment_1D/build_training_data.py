#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:02:23 2024

@author: lorenzo piu
"""
# %% Import
import numpy as np
import torch
import matplotlib.pyplot as plt
from _utils import scale_vector, f
import pickle
from sklearn.model_selection import train_test_split
import os
import random

# %% Input parameters
training_folder = 'training_data'
input_dim   = 1
n_samples   = 1000000 + 1
train_split = 0.5
noise       = 0.1
x_ratio     = 0.8
x_lim       = [-1, 1]
x_rem       = [0.2, 0.45] # x coordinates of the data to remove
extr_limit  = 1.5 # extrapolates until 1.5 the x_lim extrema

# Set random seed for reproducibility
random_seed = 42  # You can choose any integer here
np.random.seed(random_seed)  # Set seed for NumPy
torch.manual_seed(random_seed)  # Set seed for PyTorch

# %% Build training and validation tensor
normal_x_1  = torch.randn(int(n_samples*x_ratio), input_dim)
x_1         = torch.tensor(scale_vector(normal_x_1, x_lim[0], x_rem[0]))
normal_x_2  = torch.randn(int(n_samples*(1-x_ratio)), input_dim)
x_2         = torch.tensor(scale_vector(normal_x_2, x_lim[1], x_rem[1]))
x           = torch.vstack([x_1, x_2])
x           = random.shuffle(x)
y           = f(x, noise)

x_train, x_val, y_train, y_val = train_test_split(x, 
                                                    y, 
                                                    train_size=train_split, 
                                                    random_state=random_seed, 
                                                    )

# %% Build interpolation and extrapolaiton tensors
# interpolation
normal_x_interp  = torch.randn(int(n_samples/3), input_dim)
x_interp         = torch.tensor(scale_vector(normal_x_interp, x_rem[0], x_rem[1]))
y_interp         = f(x_interp, noise)

normal_x_1_extr  = torch.randn(int(n_samples/3), input_dim)
x_1_extr         = torch.tensor(scale_vector(normal_x_1_extr, extr_limit*x_lim[0], x_lim[0]))
normal_x_2_extr  = torch.randn(int(n_samples//3), input_dim)
x_2_extr         = torch.tensor(scale_vector(normal_x_2_extr, x_lim[1], extr_limit*x_lim[1]))
x_extr           = torch.vstack([x_1_extr, x_2_extr])
y_extr           = f(x_extr, noise)


# %% Plots
# Update plotting parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # or use ["DejaVu Serif"] if Times New Roman is unavailable
    "font.size": 16,                    # General font size
    "axes.titlesize": 20,               # Title font size
    "axes.labelsize": 18,               # Label font size
    "legend.fontsize": 14,              # Legend font size
    "xtick.labelsize": 16,              # X-tick label size
    "ytick.labelsize": 16,              # Y-tick label size
})

# Define plotting details
steps       = 1000
x_ptp       = x_lim[1] - x_lim[0]
x_lin       = torch.linspace(x_lim[0]-x_ptp/4, x_lim[1]+x_ptp/4, steps)
y_lin       = f(x_lin, noise=0)

# Plot #1: training data with density
plt.figure(figsize=(8, 5), dpi=600)
plt.hist2d(x.numpy().flatten(), y.numpy().flatten(), bins=300, cmin=1, cmap='viridis')
plt.colorbar(label='Count')
plt.scatter([], [], cmap='viridis', label='Training/Validation points', marker='s', s=5) # Only used for the legend

# Line plot of the function
plt.plot(x_lin, y_lin, c='red', linewidth=2.5, label='y = f(x)')

# Labels, legend, and title
plt.xlabel('x')
plt.ylabel('y')
# plt.title('2D Histogram with Function Overlay')
plt.legend(loc='lower right')
plt.grid(visible=True, linestyle='--', alpha=0.6)

plt.xlim(x_lin.min(), x_lin.max())

# Tight layout for a cleaner look
plt.tight_layout()

plt.savefig(os.path.join(training_folder, 'training_data.png'))

# Show plot
plt.show()


# Plot #2: Training and testing data
plt.figure(figsize=(8, 5), dpi=600)
plt.scatter(x.numpy().flatten()        , y.numpy().flatten()       , c='#440154', marker='.', s=7, alpha=0.01)
plt.scatter([]                         , []                        , c='#440154', marker='.', s=70, alpha=1  , label='Training/Validation points')
plt.scatter(x_extr.numpy().flatten()   , y_extr.numpy().flatten()  , c='#35B779', marker='.', s=7, alpha=0.01)
plt.scatter([]                         , []                        , c='#35B779', marker='.', s=70, alpha=1  , label='Extrapolation points')
plt.scatter(x_interp.numpy().flatten() , y_interp.numpy().flatten(), c='#FDE725', marker='.', s=7, alpha=0.01)
plt.scatter([]                         , []                        , c='#FDE725', marker='.', s=70, alpha=1  , label='Interpolation points')

# Line plot of the function
plt.plot(x_lin, y_lin, c='red', linewidth=2.5, label='y = f(x)')

# Labels, legend, and title
plt.xlabel('x')
plt.ylabel('y')
# plt.title('2D Histogram with Function Overlay')
plt.legend(loc='lower right')
plt.grid(visible=True, linestyle='--', alpha=0.6)

plt.xlim(x_lin.min(), x_lin.max())

# Tight layout for a cleaner look
plt.tight_layout()

plt.savefig(os.path.join(training_folder, 'training_testing_data.png'))

# Show plot
plt.show()

# %% Plots (proportioned)
# Plot #1: Training data with density
fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=600)
im1 = ax1.hist2d(x.numpy().flatten(), y.numpy().flatten(), bins=300, cmin=1, cmap='viridis')
cbar = fig1.colorbar(im1[3], ax=ax1, label='Count')
ax1.scatter([], [], cmap='viridis', label='Training/Validation points', marker='s', s=10) # Only used for the legend


# Line plot of the function
ax1.plot(x_lin, y_lin, c='red', linewidth=2.5, label='y = f(x)')

# Labels, legend, and title
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(loc='upper center')
ax1.grid(visible=True, linestyle='--', alpha=0.6)
ax1.set_xlim(x_lin.min(), x_lin.max())
ax1.set_ylim(-2,2.5)

# Save and show the plot
plt.savefig(os.path.join(training_folder, 'training_data.png'))
plt.show()

# Plot #2: Training and testing data
fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=600)
ax2.scatter(x.numpy().flatten(), y.numpy().flatten(), c='#440154', marker='.', s=7, alpha=0.01)
ax2.scatter([], [], c='#440154', marker='.', s=70, alpha=1, label='Training/Validation points')
ax2.scatter(x_extr.numpy().flatten(), y_extr.numpy().flatten(), c='#35B779', marker='.', s=7, alpha=0.01)
ax2.scatter([], [], c='#35B779', marker='.', s=70, alpha=1, label='Extrapolation points')
ax2.scatter(x_interp.numpy().flatten(), y_interp.numpy().flatten(), c='#FDE725', marker='.', s=7, alpha=0.01)
ax2.scatter([], [], c='#FDE725', marker='.', s=70, alpha=1, label='Interpolation points')

# Line plot of the function
ax2.plot(x_lin, y_lin, c='red', linewidth=2.5, label='y = f(x)')

# Labels, legend, and title
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend(loc='upper center')
ax2.grid(visible=True, linestyle='--', alpha=0.6)
ax2.set_xlim(x_lin.min(), x_lin.max())
ax2.set_ylim(-2,2.5)

# Match the axis sizes of the two plots
pos1 = ax1.get_position()  # Get the position of the first plot
ax2.set_position([pos1.x0, pos1.y0, pos1.width, pos1.height])  # Apply the same position to the second plot

# Save and show the plot
plt.savefig(os.path.join(training_folder, 'training_testing_data.png'))
plt.show()

# %% Save training data, parameters, and plots
params = {
    "input_dim": input_dim,
    "n_samples": n_samples,
    "train_split": train_split,
    "noise": noise,
    "x_ratio": x_ratio,
    "x_lim": x_lim,
    "x_rem": x_rem,
    "random_seed":random_seed,
}

with open(os.path.join(training_folder, 'params.pkl'), 'wb') as file:
    pickle.dump(params, file)
    
torch.save(x, os.path.join(training_folder, 'x.pt'))
torch.save(y, os.path.join(training_folder, 'y.pt'))
torch.save(x_train, os.path.join(training_folder, 'x_train.pt'))
torch.save(x_val, os.path.join(training_folder, 'x_val.pt'))
torch.save(x_interp, os.path.join(training_folder, 'x_interp.pt'))
torch.save(x_extr, os.path.join(training_folder, 'x_extr.pt'))
torch.save(y_train, os.path.join(training_folder, 'y_train.pt'))
torch.save(y_val, os.path.join(training_folder, 'y_val.pt'))
torch.save(y_interp, os.path.join(training_folder, 'y_interp.pt'))
torch.save(y_extr, os.path.join(training_folder, 'y_extr.pt'))