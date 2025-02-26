#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:47:22 2024

@author: lpiu
"""
# %% Import libraries
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import os
import sys

from architectures import BayesianNN
from architectures import loss_function
from architectures import FCNN
from architectures import FCNN_2
from _utils import check_x, r2_score, nmse, set_seed

# %% Riproducibility
set_seed(42)

# %% Load Training data
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:47:22 2024

@author: lpiu
"""
# %% Import libraries
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import os
import sys

from architectures import BayesianNN
from architectures import loss_function
from architectures import FCNN
from architectures import FCNN_2
from _utils import check_x, r2_score, nmse

# %% Load Training data
training_folder = "training_data"
n_training      = '12'
formato         = ".npy"

# Load the JSON data from a file
with open(f'{training_folder}/info.json', 'r') as file:
    info = json.load(file)

print("\n----------------------------------------------------------------------"
      "\nLoading training data..."
      )
filter_size = info[n_training]["filter_size"]
variables   = info[n_training]["variables"]
tts         = info[n_training]["train_test_split"]
print(
      f"\nLoaded dataset 01:\n - variables        : {variables}"
                          f"\n - Train-test split : {tts}"
                          f"\n - Filter size      : {filter_size}"                          
      )

# Load the training data
print("\nFiles read:")
path = f"{training_folder}/{n_training}"
variables_to_load = ["X", "X_train", "X_test","HRR_DNS_train", "HRR_DNS_test", "HRR_LFR_train", "HRR_LFR_test"]
data = dict()
for var in variables_to_load:
    data[var] = np.load(f"{path}/{var}{formato}")
    print(f" - {path}/{var}{formato}")
    
check_x(data["X"])

# %% Create Model folder and initialize log
models_folder = 'Models'
model_folder  = models_folder + f'/{n_training}'
optim_folder  = model_folder  + '/optim'
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
if not os.path.exists(optim_folder):
    os.mkdir(optim_folder)

# Open a file in write mode
# log_file = open(f"{optim_folder}/log.txt", "w")

# Define a class to redirect output to both console and file
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure the content is written immediately

    def flush(self):
        for f in self.files:
            f.flush()

# Redirect sys.stdout to both console and file
# sys.stdout = Tee(sys.stdout, log_file)

# %% Initialize Neural Network's hyperparameters
input_dim   = data["X_train"].shape[1]
n_layers    = 12
n_neurons   = 64
output_dim  = 1 
tr_layers   = 3 

# %% Prepare data for Pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert the variables to tensors and move to GPU
X_train = torch.tensor(data["X_train"], dtype=torch.float32).to(device)
X_test = torch.tensor(data["X_test"], dtype=torch.float32) # .to(device)
HRR_DNS_train = torch.tensor(data["HRR_DNS_train"], dtype=torch.float32).to(device)
HRR_DNS_test = torch.tensor(data["HRR_DNS_test"], dtype=torch.float32) # .to(device)
HRR_LFR_train = torch.tensor(data["HRR_LFR_train"], dtype=torch.float32).to(device)
HRR_LFR_test = torch.tensor(data["HRR_LFR_test"], dtype=torch.float32) # .to(device)

# %% Train FCNN
import optuna

def objective(trial):
    # Suggest values for hyperparameters
    n_layers = trial.suggest_int("n_layers", 3, 12)
    n_neurons = trial.suggest_int("n_neurons", 32, 128)
    tr_layers = trial.suggest_int("tr_layers", 0, 3)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    
    batch_size = 10000
    
    # Initialize the model with these hyperparameters
    fcnn = FCNN_2(input_dim, n_layers, n_neurons, output_dim, tr_layers).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(fcnn.parameters(), lr=lr)
    
    # Training loop
    num_epochs = 1000  # Can be adjusted or even set as a hyperparameter
    for epoch in range(num_epochs):
        fcnn.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = fcnn(X_train)
        
        # Loss calculation
        loss = F.mse_loss(output * HRR_LFR_train, HRR_DNS_train) / torch.max(HRR_DNS_train)**2
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Validation loss
        fcnn.eval()
        with torch.no_grad():
            # Randomly select a batch of test data
            random_indices = random.sample(range(X_test.shape[0]), batch_size)
            
            # Extract the random batch
            batch_X_test = X_test[random_indices].to(device)
            batch_HRR_DNS_test = HRR_DNS_test[random_indices].to(device)
            batch_HRR_LFR_test = HRR_LFR_test[random_indices].to(device)
            
            # Forward pass on the random test batch
            test_output = fcnn(batch_X_test)
            
            # Compute test loss for this batch
            test_loss = F.mse_loss(test_output*batch_HRR_LFR_test, batch_HRR_DNS_test) / torch.max(HRR_DNS_train)**2
        
        # Track the validation loss after each epoch
        trial.report(test_loss.item(), epoch)
        
        # Release GPU memory
        del batch_X_test, batch_HRR_LFR_test, batch_HRR_DNS_test
        
        # # Optuna Pruning: stop early if no improvement
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
    
    return test_loss.item()

# Create an Optuna study and run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Print the best parameters
print("Best parameters found: ", study.best_params)
print("Best test loss achieved: ", study.best_value)

# log_file.close()

# %% Save files
if not os.path.exists(f"{optim_folder}/lr"):
    np.save(f"{optim_folder}/lr", study.best_params["lr"])
    np.save(f"{optim_folder}/n_layers", study.best_params["n_layers"])
    np.save(f"{optim_folder}/n_neurons", study.best_params["n_neurons"])
    np.save(f"{optim_folder}/tr_layers", study.best_params["tr_layers"])

# np.save(f"{optim_folder}/best_loss.txt")



# %% Create Model folder and initialize log
models_folder = 'Models'
model_folder  = models_folder + f'/{n_training}'
optim_folder  = model_folder  + '/optim'
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
if not os.path.exists(optim_folder):
    os.mkdir(optim_folder)
    
# Open a file in write mode
# log_file = open(f"{optim_folder}/log.txt", "w")

# Define a class to redirect output to both console and file
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure the content is written immediately

    def flush(self):
        for f in self.files:
            f.flush()

# Redirect sys.stdout to both console and file
# sys.stdout = Tee(sys.stdout, log_file)

# %% Initialize Neural Network's hyperparameters
input_dim   = data["X_train"].shape[1]
n_layers    = 12
n_neurons   = 64
output_dim  = 1 
tr_layers   = 3 

# %% Prepare data for Pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert the variables to tensors and move to GPU
X_train = torch.tensor(data["X_train"], dtype=torch.float32).to(device)
X_test = torch.tensor(data["X_test"], dtype=torch.float32) # .to(device)
HRR_DNS_train = torch.tensor(data["HRR_DNS_train"], dtype=torch.float32).to(device)
HRR_DNS_test = torch.tensor(data["HRR_DNS_test"], dtype=torch.float32) # .to(device)
HRR_LFR_train = torch.tensor(data["HRR_LFR_train"], dtype=torch.float32).to(device)
HRR_LFR_test = torch.tensor(data["HRR_LFR_test"], dtype=torch.float32) # .to(device)

# %% Train FCNN
import optuna

def objective(trial):
    # Suggest values for hyperparameters
    n_layers = trial.suggest_int("n_layers", 3, 12)
    n_neurons = trial.suggest_int("n_neurons", 32, 128)
    tr_layers = trial.suggest_int("tr_layers", 0, 3)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    
    batch_size = 10000
    
    # Initialize the model with these hyperparameters
    fcnn = FCNN_2(input_dim, n_layers, n_neurons, output_dim, tr_layers).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(fcnn.parameters(), lr=lr)
    
    # Training loop
    num_epochs = 1000  # Can be adjusted or even set as a hyperparameter
    for epoch in range(num_epochs):
        fcnn.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = fcnn(X_train)
        
        # Loss calculation
        loss = F.mse_loss(output * HRR_LFR_train, HRR_DNS_train) / torch.max(HRR_DNS_train)**2
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Validation loss
        fcnn.eval()
        with torch.no_grad():
            # Randomly select a batch of test data
            random_indices = random.sample(range(X_test.shape[0]), batch_size)
            
            # Extract the random batch
            batch_X_test = X_test[random_indices].to(device)
            batch_HRR_DNS_test = HRR_DNS_test[random_indices].to(device)
            batch_HRR_LFR_test = HRR_LFR_test[random_indices].to(device)
            
            # Forward pass on the random test batch
            test_output = fcnn(batch_X_test)
            
            # Compute test loss for this batch
            test_loss = F.mse_loss(test_output*batch_HRR_LFR_test, batch_HRR_DNS_test) / torch.max(HRR_DNS_train)**2
        
        # Track the validation loss after each epoch
        trial.report(test_loss.item(), epoch)
        
        # Release GPU memory
        del batch_X_test, batch_HRR_LFR_test, batch_HRR_DNS_test
        
        # # Optuna Pruning: stop early if no improvement
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
    
    return test_loss.item()

# Create an Optuna study and run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Print the best parameters
print("Best parameters found: ", study.best_params)
print("Best test loss achieved: ", study.best_value)

# log_file.close()

# %% Save files
if not os.path.exists(f"{optim_folder}/lr"):
    np.save(f"{optim_folder}/lr", study.best_params["lr"])
    np.save(f"{optim_folder}/n_layers", study.best_params["n_layers"])
    np.save(f"{optim_folder}/n_neurons", study.best_params["n_neurons"])
    np.save(f"{optim_folder}/tr_layers", study.best_params["tr_layers"])

# np.save(f"{optim_folder}/best_loss.txt")

