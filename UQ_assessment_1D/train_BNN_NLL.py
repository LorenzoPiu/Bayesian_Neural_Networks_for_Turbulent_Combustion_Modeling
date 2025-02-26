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
import argparse
import subprocess

from architectures import BayesianNN, FCNN, fcnn_to_bnn
from _utils import load_training_data, f, plot_loss, set_seed, ordered_indices_generator, build_model_folder
from my_plot_utils import plot_training_data, fig_style, plot_model_output
from evaluation_metrics import gaussian_NLL

# %% Input parameters
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Neural Network architecture and training parameters.")

# Define command-line arguments
parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
parser.add_argument('--n_neurons', type=int, default=128, help='Number of neurons')
parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
parser.add_argument('--tr_layers', type=int, default=2, help='Number of transition layers')
parser.add_argument('--n_epochs', type=int, default=400, help='Number of training epochs')
parser.add_argument('--n_epochs_pretrain', type=int, default=400, help='Number of pretraining epochs')
parser.add_argument('--batch_size', type=int, default=50000, help='Batch size for training')
parser.add_argument('--kl_weight', type=float, default=0.00000, help='Weight for KL-divergence regularization')
parser.add_argument('--prior_std', type=float, default=0.0003, help='Prior standard deviation')
parser.add_argument('--n_samples', type=int, default=10, help='Number of forward passes for every training step')
parser.add_argument('--seed', type=int, default=52, help='Random seed for reproducibility')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
input_dim = args.input_dim
n_neurons = args.n_neurons
n_layers = args.n_layers
output_dim = args.output_dim
tr_layers = args.tr_layers
n_epochs = args.n_epochs
n_epochs_pretrain = args.n_epochs_pretrain
batch_size = args.batch_size
kl_weight = args.kl_weight
prior_std = args.prior_std
n_samples = args.n_samples
seed = args.seed

# %% Initialize model folder
model_folder = build_model_folder(architecture='BNN_NLL',
                                  input_dim=input_dim,
                                  n_neurons=n_neurons,
                                  n_layers=n_layers,
                                  output_dim=output_dim,
                                  tr_layers=tr_layers,
                                  batch_size=batch_size,
                                  n_epochs=n_epochs,
                                  n_epochs_pretrain=n_epochs_pretrain,
                                  kl_weight=kl_weight,
                                  prior_std=prior_std,
                                  seed=seed,
                                  n_samples=n_samples
                                  )

# %% Load training data and pretrain FCNN model
# Load training and testing data
data = load_training_data(verbose=True)

FCNN_path = build_model_folder(architecture='FCNN', 
                               mkdir=False,
                               input_dim=input_dim, 
                               n_neurons=n_neurons,
                               n_layers=n_layers,
                               output_dim=output_dim,
                               tr_layers=tr_layers,
                               n_epochs=n_epochs_pretrain,
                               batch_size=batch_size,
                               seed=seed,
                               )

if not os.path.exists(FCNN_path):
    # Call the script to train the FCNN
    command = ['python', 'train_FCNN.py', 
               '--input_dim', f'{input_dim}', 
               '--n_neurons', f'{n_neurons}',
               '--n_layers', f'{n_layers}',
               '--output_dim', f'{output_dim}',
               '--tr_layers', f'{tr_layers}',
               '--n_epochs', f'{n_epochs_pretrain}',
               '--batch_size', f'{batch_size}',
               '--seed', f'{seed}']
    
    subprocess.run(command)

model = torch.load(os.path.join(FCNN_path, 'model.pth'))
    
# %% Train the BNN
# Use GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Only train if the model does not exist yet
if os.path.exists(os.path.join(model_folder, 'model.pth')):
    bnn = torch.load(os.path.join(model_folder, 'model.pth'))
else:
    # Train the variational network
    set_seed(seed)
    
    # Model, optimizer
    bnn = fcnn_to_bnn(model, prior_std=prior_std)
    # bnn.train_sigma(False)
    optimizer = torch.optim.Adam(bnn.parameters(), lr=0.001)
    
    # Loss function
    def loss_function(y_true, y_pred, sigma_pred, kl_divergence, kl_weight=1.0):
        # Negative log likelihood 
        log_likelihood = gaussian_NLL(y_true, y_pred, sigma_pred)
        
        # Total loss: log likelihood + KL divergence
        return log_likelihood + kl_weight * kl_divergence
    
    bnn.train()
    loss_list = []
    loss_test_list = []
    bnn.to(device)
    
    t = time.time()
    
    for epoch in range(n_epochs):
        running_loss = 0.0
        for idx in ordered_indices_generator(len(data['x_train']), batch_size):
            X_batch = data['x_train'][idx, :].to(device)
            Y_batch = data['y_train'][idx, :].to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
    
            # Forward pass
            y_mean, y_sigma = bnn.forward_MC(X_batch, n_samples=n_samples)
            
            # Compute KL divergence
            kl_div = bnn.kl_divergence()
            
            loss = loss_function(Y_batch, y_mean, y_sigma, kl_div, kl_weight=kl_weight)
            
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        
        # Save training loss
        loss_list.append(running_loss)
        
        # Release memory on gpu
        del X_batch
        del Y_batch
        
        # Testing loop
        if (('x_val' in data) and ('y_val' in data)):
            running_test_loss = 0.0
            for idx in ordered_indices_generator(len(data['x_val']), batch_size):
                X_batch = data['x_val'][idx, :].to(device)
                Y_batch = data['y_val'][idx, :].to(device)
                
                with torch.no_grad():
                    # Forward pass + loss
                    y_mean, y_sigma = bnn.forward_MC(X_batch, n_samples=n_samples)
                    
                    # Compute KL divergence for regularization
                    kl_div = bnn.kl_divergence()
                    
                    test_loss = loss_function(Y_batch, y_mean, y_sigma, kl_div, kl_weight=kl_weight)
                    
                running_test_loss += loss.item()
            
            loss_test_list.append(running_test_loss)
    
        if (epoch + 1) % (n_epochs//20) == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Training loss: {running_loss / len(data['x_train'])}")
            
        # Release memory on gpu
        del X_batch
        del Y_batch
    
    bnn.training_time = time.time() - t
    
    bnn.train_loss = loss_list
    bnn.test_loss  = loss_test_list
    
    #fig=plot_loss(bnn.train_loss, bnn.test_loss, return_fig=True)
    #fig.savefig(os.path.join(model_folder, 'training_BNN.png'))
    
    torch.save(bnn, os.path.join(model_folder, 'model.pth'))

# %% Plots
# plot_training_data()

# Define sampling function
steps       = 1000
x_ptp       = 2
x_lin       = torch.linspace(-1-x_ptp/4, 1+x_ptp/4, steps)
x_lin       = torch.reshape(x_lin, [len(x_lin), 1])
y_lin       = f(x_lin, noise=0)

# Plot mean and uncertainty
bnn.to('cpu')
bnn.eval()
with torch.no_grad():
    y_model_mean, y_model_sigma = bnn.forward_MC(x_lin, n_samples=1000)
    
plot_model_output(y_model_mean, y_model_sigma, data, model_folder, x_ptp=x_ptp, steps=1000)

