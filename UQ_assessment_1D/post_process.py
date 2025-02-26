#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:27:14 2025

@author: lpiu
"""

# %% Import libraries
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
import warnings

from architectures import BayesianNN, FCNN, fcnn_to_bnn, MC_Dropout, DeepEnsemble, inference
from _utils import load_training_data, f, plot_loss, set_seed, ordered_indices_generator, build_model_folder
from my_plot_utils import plot_training_data, fig_style, plot_model_output
from evaluation_metrics import MSE, MPIW, PICP, gaussian_NLL

# %% Inputs
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Neural Network architecture and training parameters.")

# Define command-line arguments
parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
parser.add_argument('--n_neurons', type=int, default=64, help='Number of neurons')
parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
parser.add_argument('--tr_layers', type=int, default=1, help='Number of transition layers')
parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs')
parser.add_argument('--n_epochs_pretrain', type=int, default=400, help='Number of epochs for the pretrained network')
parser.add_argument('--batch_size', type=int, default=50000, help='Batch size')
parser.add_argument('--n_ensembles', type=int, default=10, help='Number of ensembles')
parser.add_argument('--n_samples', type=int, default=100, help='Number of samples per forward pass')
parser.add_argument('--dropout_prob', type=float, default=0.01, help='Dropout probability')
parser.add_argument('--kl_weight', type=float, default=0.00000, help='Weight for KL-divergence regularization')
parser.add_argument('--prior_std', type=float, default=0.0003, help='Prior standard deviation')
parser.add_argument('--seed', type=int, default=36, help='Random seed')
parser.add_argument('--architecture', type=str, default='DEN_MCD', help='architecture type. Options: BNN, BNN_NLL, DEN, DEN_MCD, MCD, MCD_NLL')
parser.add_argument('--n_samples_test', type=int, default=10, help='Number of samples per forward pass in the test phase')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
input_dim         = args.input_dim
n_neurons         = args.n_neurons
n_layers          = args.n_layers
output_dim        = args.output_dim
tr_layers         = args.tr_layers
n_epochs          = args.n_epochs
n_epochs_pretrain =  args.n_epochs_pretrain
batch_size        = args.batch_size
n_ensembles       = args.n_ensembles
n_samples         = args.n_samples
dropout_prob      = args.dropout_prob
kl_weight         = args.kl_weight
prior_std         = args.prior_std
seed              = args.seed
architecture      = args.architecture
n_samples_test    = args.n_samples_test

# %% Read Model
params_dict = {
    'input_dim': args.input_dim,
    'n_neurons': args.n_neurons,
    'n_layers': args.n_layers,
    'output_dim': args.output_dim,
    'tr_layers': args.tr_layers,
    'n_epochs': args.n_epochs,
    'n_epochs_pretrain': args.n_epochs_pretrain,
    'batch_size': args.batch_size,
    'n_ensembles': args.n_ensembles,
    'n_samples': args.n_samples,
    'dropout_prob': args.dropout_prob,
    'kl_weight': args.kl_weight,
    'prior_std': args.prior_std,
    'seed': args.seed
}

# Remove entries based on the model
if architecture == 'BNN':
    del params_dict['n_samples']
    del params_dict['n_ensembles']
    del params_dict['dropout_prob']
elif architecture == 'BNN_NLL':
    del params_dict['n_ensembles']
    del params_dict['dropout_prob']
elif architecture == 'DEN':
    del params_dict['n_samples']
    del params_dict['dropout_prob']
    del params_dict['kl_weight']
    del params_dict['prior_std']
    del params_dict['n_epochs_pretrain']
elif architecture == 'DEN_MCD':
    del params_dict['kl_weight']
    del params_dict['prior_std']
    del params_dict['n_epochs_pretrain']
elif architecture == 'MCD':
    del params_dict['n_ensembles']
    del params_dict['n_samples']
    del params_dict['kl_weight']
    del params_dict['prior_std']
    del params_dict['n_epochs_pretrain']
elif architecture == 'MCD_NLL':
    del params_dict['n_ensembles']
    del params_dict['kl_weight']
    del params_dict['prior_std']
    del params_dict['n_epochs_pretrain']


model_folder = build_model_folder(architecture=architecture, mkdir=False, **params_dict)
if not os.path.exists(model_folder):
    raise ValueError(f"The folder {model_folder} does not exist. Please train the model before using the post processing utilities")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.load(os.path.join(model_folder, 'model.pth'))
data = load_training_data()

# %% Test model
# Set seed for reproducibility
set_seed(seed)

# Check that the model is on the cpu
# model.to('cpu')
# model.eval()

x_time_test = data['x_val'][0:1000]
t = time.time()
with torch.no_grad():
    _, _ = inference(model, x_time_test, n_samples=n_samples_test, gpu=False)
time_test = time.time()-t
time_test /= len(x_time_test)

mean = {}
sigma = {}

# test on all the points (interpolation, validation, extr...)
for case in ['train', 'val', 'extr', 'interp']:  
    with torch.no_grad():
        mean[f'{case}'], sigma[f'{case}'] = inference(model, data[f'x_{case}'], n_samples=n_samples_test, gpu=False)

# %% Compute evaluation metrics
MSE_dict = {}
NLL_dict = {}
MPIW_dict = {}
PICP_dict = {}

for case in ['train', 'val', 'extr', 'interp']:  
    y_true = data[f'y_{case}']
    y_mean_pred = mean[f'{case}']
    y_sigma_pred = sigma[f'{case}']
    y_high = y_mean_pred+2*y_sigma_pred
    y_low = y_mean_pred-2*y_sigma_pred
    
    MSE_dict[f'{case}']  = str(MSE(y_true, y_mean_pred))
    NLL_dict[f'{case}']  = str(float(gaussian_NLL(y_true, y_mean_pred, y_sigma_pred)))
    MPIW_dict[f'{case}'] = str(float(MPIW(y_low, y_high)))
    PICP_dict[f'{case}'] = str(float(PICP(y_true, y_low, y_high)))

import json
with open(os.path.join(model_folder, "MSE.json"), "w") as json_file:
    json.dump(MSE_dict, json_file, indent=4) 
with open(os.path.join(model_folder, "NLL.json"), "w") as json_file:
    json.dump(NLL_dict, json_file, indent=4)  
with open(os.path.join(model_folder, "MPIW.json"), "w") as json_file:
    json.dump(MPIW_dict, json_file, indent=4)  
with open(os.path.join(model_folder, "PICP.json"), "w") as json_file:
    json.dump(PICP_dict, json_file, indent=4)  

with open(os.path.join(model_folder, "time.txt"), "w") as file:
    # Write each variable on a new line
    file.write(f"training time : {model.training_time} s\n")
    file.write(f"testing time  : {time_test} s\n")

