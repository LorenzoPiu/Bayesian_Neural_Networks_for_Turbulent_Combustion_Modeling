#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:50:07 2024

@author: lorenzo piu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from _utils import ordered_indices_generator
from evaluation_metrics import gaussian_NLL


# -----------------------------------------------------------------------------
#                               FCNN
# -----------------------------------------------------------------------------
class FCNN(nn.Module):
    def __init__(self, input_dim, n_layers, n_neurons, output_dim, tr_layers=0):
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList() # initialize the layers list as an empty list using nn.ModuleList()
        
        self.init_params = [input_dim, n_layers, n_neurons, output_dim, tr_layers]
        
        in_features = input_dim # input dimension for the layer
        
        # define the number of neurons in each layer
        tr_layers = tr_layers+1  # something about the math after is a bit confusing but somehow with this +1 it works
        n = list()
        n.append(input_dim)
        for i in range(tr_layers-1):   # -1 because if not it starts creating the following one
            n.append( input_dim +(n_neurons-input_dim)*(i+1)//tr_layers) 
        for _ in range(n_layers):
            n.append(n_neurons)
        for i in range(tr_layers-1):
            n.append( output_dim +(n_neurons-output_dim)*(tr_layers-i-1)//tr_layers) 
        n.append(output_dim)

        # Hidden layers
        for i in range(len(n) - 1):
            # Add layer
            self.layers.append(nn.Linear(n[i], n[i+1]))
        
    def _initialize_weights(self, seed=None):
            if seed is not None:
                torch.manual_seed(seed)  # Set the seed for reproducibility
                np.random.seed(seed)  # Set the seed for numpy (if needed)
            
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    # Initialize weights and biases
                    nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))  # He initialization
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / np.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)  # Uniform initialization for biases

    def forward(self, x):    # Function to perform forward propagation
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
    def fit(self, data, 
            epochs=1000, 
            criterion=None, 
            optimizer=None, 
            weight_decay=0.0,
            batch_size=None,
            learning_rate=0.01,
            verbose=True,
            n_prints=20, # if verbose, how many times should plot the loss in the console
            use_gpu=True,
            plot_loss=True
            ):
        
        """Trains the model"""
        
        if optimizer is None:
            optimizer=optim.Adam(self.parameters(), lr=learning_rate)
        
        # initialize batch size for both training and testing datasets
        if batch_size is None:
            batch_size = len(data['x_train'])
        if 'x_val' in data:
            batch_size_test = batch_size
            if len(data['x_val']) < batch_size_test:
                batch_size_test = len(data['x_val'])
            
        # initialize the loss function
        if criterion is None:
            criterion = nn.MSELoss()
            
        if use_gpu is True:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        self.train()
        loss_list = []
        loss_test_list = []
        self.to(device)
        for epoch in range(epochs):
            running_loss = 0.0
            for idx in ordered_indices_generator(len(data['x_train']), batch_size):
                X_batch = data['x_train'][idx, :].to(device)
                Y_batch = data['y_train'][idx, :].to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = self.forward(X_batch)
                loss = criterion(outputs, Y_batch)

                # Add L1 regularization
                l1_reg = 0.0
                for param in self.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += weight_decay * l1_reg

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
                for idx in ordered_indices_generator(len(data['x_val']), batch_size_test):
                    X_batch = data['x_val'][idx, :].to(device)
                    Y_batch = data['y_val'][idx, :].to(device)
                    
                    with torch.no_grad():
                        # Forward pass + loss
                        outputs = self.forward(X_batch)
                        test_loss = criterion(outputs, Y_batch)
                        
                    # Add L1 regularization
                    test_loss += weight_decay * l1_reg # was computed previously

                    running_test_loss += loss.item()
                
                loss_test_list.append(running_test_loss)

            if (epoch + 1) % (epochs//n_prints) == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Training loss: {running_loss / len(data['x_train'])}")
        self.train_loss = loss_list
        self.test_loss  = loss_test_list


# -----------------------------------------------------------------------------
#                               BNN
# -----------------------------------------------------------------------------
# Define the Variational Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.001):
        super(BayesianLinear, self).__init__()
        # Variational parameters for the weight distribution (mean and log of standard deviation)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.001))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(np.log10(prior_std)))
        
        # Variational parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.001))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-7))
        
        # Prior standard deviation
        self.prior_std = prior_std

    def forward(self, x):
        # Sample weights and biases using reparameterization trick
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        
        # Sampling weights
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias)
    

    def kl_divergence(self):
        # KL divergence between learned distribution and the prior
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        
        # KL for weights
        weight_kl = torch.sum(
            torch.log(self.prior_std / weight_sigma) + 
            (weight_sigma ** 2 + self.weight_mu ** 2) / (2 * self.prior_std ** 2) - 0.5
        )
        
        # KL for biases
        bias_kl = torch.sum(
            torch.log(self.prior_std / bias_sigma) + 
            (bias_sigma ** 2 + self.bias_mu ** 2) / (2 * self.prior_std ** 2) - 0.5
        )
        
        return weight_kl + bias_kl
    
# Define the Bayesian NN
class BayesianNN(nn.Module):
    def __init__(self, input_dim, n_layers, n_neurons, output_dim, tr_layers=0, prior_std=0.001):
        super(BayesianNN, self).__init__()
        
        # Check that the inputs are integers
        if (
                (not isinstance(input_dim, int))
                or (not isinstance(n_layers, int)) 
                or (not isinstance(n_neurons, int)) 
                or (not isinstance(input_dim, int)) 
                or (not isinstance(output_dim, int))
                or (not isinstance(tr_layers, int))
            ):
            raise ValueError("The input must be an integer")
        
        # Create a list to hold the layers
        self.layers = nn.ModuleList()
        
        in_features = input_dim # input dimension for the layer
        
        # define the number of neurons in each layer
        tr_layers = tr_layers+1  # something about the math after is a bit confusing but somehow with this +1 it works
        n = list()
        n.append(input_dim)
        for i in range(tr_layers-1):   # -1 because if not it starts creating the following one
            n.append( input_dim +(n_neurons-input_dim)*(i+1)//tr_layers) 
        for _ in range(n_layers):
            n.append(n_neurons)
        for i in range(tr_layers-1):
            n.append( output_dim +(n_neurons-output_dim)*(tr_layers-i-1)//tr_layers) 
        n.append(output_dim)
        
        # Hidden layers
        for i in range(len(n) - 1):
            # Add layer
            self.layers.append(BayesianLinear(n[i], n[i+1], prior_std=prior_std)) # add a Bayesian Linear layer
            init.normal_(self.layers[-1].weight_mu, mean=0.0, std=1)
        
        # # Randomly initialize weights
        # init.normal(self.weights)
    
    def forward(self, x):
        # Pass input through all layers except the last (activation applied only to hidden layers)
        count = 0
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # Output layer (no activation function)
        return self.layers[-1](x)
    
    def forward_MC (self, x, n_samples=1000):
        self.eval()
        mean_output = 0.0
        mean_square_output = 0.0
        
        for _ in range(n_samples):
            out = self.forward(x)
            mean_output += out
            mean_square_output += out ** 2 # needed for the incremental computation of std.
            # math behind the formula: http://datagenetics.com/blog/november22017/index.html
        
        mean_output /= n_samples
        mean_square_output /= n_samples
        
        # Calculate standard deviation
        std_output = torch.sqrt(mean_square_output - mean_output ** 2)
        
        return mean_output, std_output
        
    
    def kl_divergence(self):
        # Sum KL divergence across all layers
        return sum(layer.kl_divergence() for layer in self.layers)
    
    def train_mu(self, train_mu=True):
        """
        Dynamically sets the trainability of mu and sigma parameters.
        
        Args:
            train_mu (bool, optional): If True, enable training for mu. If False, disable it.
            train_sigma (bool, optional): If True, enable training for sigma. If False, disable it.
        """
        for layer in self.layers:
            if (train_mu is True) or (train_mu is False):
                layer.weight_mu.requires_grad = train_mu
                layer.bias_mu.requires_grad = train_mu
            # if train_sigma is not None:
            #     layer.weight_log_sigma.requires_grad = train_sigma
            #     layer.bias_log_sigma.requires_grad = train_sigma
            
    def train_sigma(self, train_sigma=True):
        """
        Dynamically sets the trainability of mu and sigma parameters.
        
        Args:
            train_mu (bool, optional): If True, enable training for mu. If False, disable it.
            train_sigma (bool, optional): If True, enable training for sigma. If False, disable it.
        """
        for layer in self.layers:
            if (train_sigma is True) or (train_sigma is False):
                layer.weight_log_sigma.requires_grad = train_sigma
                layer.bias_log_sigma.requires_grad = train_sigma

# # Loss function
# def loss_function_BNN(output, target, kl_divergence, kl_weight=1.0):
#     # Negative log likelihood (mean squared error as an example)
#     log_likelihood = F.mse_loss(output, target, reduction='sum')
    
#     # Total loss: log likelihood + KL divergence
#     return log_likelihood + kl_weight * kl_divergence

def predict_with_uncertainty(model, x, n_samples=100):
    model.eval()
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(n_samples)])
    mean_pred = preds.mean(0)
    uncertainty = preds.std(0)# /torch.abs(mean_pred)  # Variance across the samples divided by the output value
    # I use the square root because it should be similar to how you compute the turbulence intensity
    return mean_pred, uncertainty


# -----------------------------------------------------------------------------
#                               MC Dropout
# -----------------------------------------------------------------------------
class MC_Dropout(nn.Module):
    def __init__(self, input_dim, n_layers, n_neurons, output_dim, tr_layers, dropout_prob=0.5):
        super(MC_Dropout, self).__init__()
        self.layers = nn.ModuleList()  # Initialize the layers list
        self.dropout_prob = dropout_prob  # Dropout probability for MC Dropout
        self.mc_dropout = True  # Flag to control dropout during inference

        # Define the architecture of neurons per layer (similar to FCNN_2)
        tr_layers += 1
        n = [input_dim]
        for i in range(tr_layers - 1):
            n.append(input_dim + (n_neurons - input_dim) * (i + 1) // tr_layers)
        for _ in range(n_layers):
            n.append(n_neurons)
        for i in range(tr_layers - 1):
            n.append(output_dim + (n_neurons - output_dim) * (tr_layers - i - 1) // tr_layers)
        n.append(output_dim)

        # Create layers and add dropout after each hidden layer
        for i in range(len(n) - 1):
            self.layers.append(nn.Linear(n[i], n[i + 1]))
            # if i < len(n) - 2:  # Add dropout only after hidden layers, not after output layer
              #   self.layers.append(nn.Dropout(p=self.dropout_prob))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)  # Set the seed for reproducibility
            np.random.seed(seed)  # Set the seed for numpy (if needed)
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)

    def enable_mc_dropout(self, mc_dropout=True):
        """Enables or disables Monte Carlo dropout during inference."""
        self.mc_dropout = mc_dropout

    def forward(self, x, smooth=False):
        dropout_mask = None
        for layer in self.layers[:-1]:
            # Forward pass in the linear layer
            x = layer(x)
            # Applies dropout after the 
            if self.mc_dropout is True:#isinstance(layer, nn.Dropout):
                if smooth is False:
                    # This method applies dropout to every layer and every sample of the network. This option improves training making the process more stochastic
                    x = torch.dropout(x, self.dropout_prob, train=self.mc_dropout)  # Force dropout if mc_dropout=True
                else:
                # This method instead generates a dropout mask only once per forward pass in every layer, and applies the same to all samples. Better for obtaining smooth outputs during inference
                    if dropout_mask is None or dropout_mask.shape != x.shape:
                        dropout_mask = torch.bernoulli(torch.full((x.shape[1],), 1 - self.dropout_prob, device=x.device)).unsqueeze(0)
                        dropout_mask = dropout_mask.expand(x.shape[0], -1)  # Expand it across all samples in the batch
    
                    x = x * dropout_mask  # Apply the same dropout mask across the batch
            
            x = torch.relu(x)  # Apply ReLU activation

        x = self.layers[-1](x)  # Output layer
        return x
    
    def inference(self, x, n_samples=100):
        """Performs inference with MC Dropout, computing the mean and standard deviation incrementally."""
        self.enable_mc_dropout()
        mean_output = 0.0
        mean_square_output = 0.0
        
        for _ in range(n_samples):
            out = self.forward(x, smooth=True)
            mean_output += out
            mean_square_output += out ** 2 # needed for the incremental computation of std.
            # math behind the formula: http://datagenetics.com/blog/november22017/index.html
        
        mean_output /= n_samples
        mean_square_output /= n_samples
        
        # Calculate standard deviation
        std_output = torch.sqrt(mean_square_output - mean_output ** 2)
        
        return mean_output, std_output
        
    def fit(self, data, 
              epochs=1000, 
              criterion=None, 
              optimizer=None, 
              weight_decay=0.0,
              batch_size=None,
              learning_rate=0.01,
              verbose=True,
              n_prints=20, # if verbose, how many times should plot the loss in the console
              use_gpu=True,
              plot_loss=True
              ):
          
          """Trains the model"""
          
          if optimizer is None:
              optimizer=optim.Adam(self.parameters(), lr=learning_rate)
          
          # initialize batch size for both training and testing datasets
          if batch_size is None:
              batch_size = len(data['x_train'])
          if 'x_val' in data:
              batch_size_test = batch_size
              if len(data['x_val']) < batch_size_test:
                  batch_size_test = len(data['x_val'])
              
          # initialize the loss function
          if criterion is None:
              criterion = nn.MSELoss()
              
          if use_gpu is True:
              if torch.backends.mps.is_available():
                  device = torch.device("mps")
              elif torch.cuda.is_available():
                  device = torch.device("cuda")
          else:
              device = torch.device("cpu")
          
          self.train()
          loss_list = []
          loss_test_list = []
          self.to(device)
          for epoch in range(epochs):
              running_loss = 0.0
              for idx in ordered_indices_generator(len(data['x_train']), batch_size):
                  X_batch = data['x_train'][idx, :].to(device)
                  Y_batch = data['y_train'][idx, :].to(device)
                  # Zero the parameter gradients
                  optimizer.zero_grad()
      
                  # Forward + Backward + Optimize
                  outputs = self.forward(X_batch)
                  loss = criterion(outputs, Y_batch)
      
                  # Add L1 regularization
                  l1_reg = 0.0
                  for param in self.parameters():
                      l1_reg += torch.norm(param, 1)
                  loss += weight_decay * l1_reg
      
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
                  for idx in ordered_indices_generator(len(data['x_val']), batch_size_test):
                      X_batch = data['x_val'][idx, :].to(device)
                      Y_batch = data['y_val'][idx, :].to(device)
                      
                      with torch.no_grad():
                          # Forward pass + loss
                          outputs = self.forward(X_batch)
                          test_loss = criterion(outputs, Y_batch)
                          
                      # Add L1 regularization
                      test_loss += weight_decay * l1_reg # was computed previously
      
                      running_test_loss += loss.item()
                  
                  loss_test_list.append(running_test_loss)
      
              if (epoch + 1) % (epochs//n_prints) == 0:
                  print(f"Epoch [{epoch + 1}/{epochs}], Training loss: {running_loss / len(data['x_train'])}")
          self.train_loss = loss_list
          self.test_loss  = loss_test_list

    def fit_bayes(self, data, 
                epochs=1000, 
                criterion=None, 
                optimizer=None, 
                weight_decay=0.0,
                batch_size=None,
                learning_rate=0.01,
                verbose=True,
                n_prints=20, # if verbose, how many times should plot the loss in the console
                use_gpu=True,
                plot_loss=True,
                n_samples=100
                ):
        
        """Trains the model"""
        
        if optimizer is None:
            optimizer=optim.Adam(self.parameters(), lr=learning_rate)
        
        # initialize batch size for both training and testing datasets
        if batch_size is None:
            batch_size = len(data['x_train'])
        if 'x_val' in data:
            batch_size_test = batch_size
            if len(data['x_val']) < batch_size_test:
                batch_size_test = len(data['x_val'])
            
        # initialize the loss function
        if criterion is None:
            criterion = gaussian_NLL
            
        if use_gpu is True:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        self.train()
        loss_list = []
        loss_test_list = []
        self.to(device)
        for epoch in range(epochs):
            running_loss = 0.0
            for idx in ordered_indices_generator(len(data['x_train']), batch_size):
                X_batch = data['x_train'][idx, :].to(device)
                Y_batch = data['y_train'][idx, :].to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs_mean, outputs_sigma = self.inference(X_batch, n_samples=n_samples)
                loss = criterion(Y_batch, outputs_mean, outputs_sigma)

                # Add L1 regularization
                l1_reg = 0.0
                for param in self.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += weight_decay * l1_reg

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
                for idx in ordered_indices_generator(len(data['x_val']), batch_size_test):
                    X_batch = data['x_val'][idx, :].to(device)
                    Y_batch = data['y_val'][idx, :].to(device)
                    
                    with torch.no_grad():
                        # Forward pass + loss
                        outputs_mean, outputs_sigma = self.inference(X_batch, n_samples=n_samples)
                        test_loss = criterion(Y_batch, outputs_mean, outputs_sigma)
                        
                    # Add L1 regularization
                    test_loss += weight_decay * l1_reg # was computed previously

                    running_test_loss += loss.item()
                
                loss_test_list.append(running_test_loss)

            if (epoch + 1) % (epochs//n_prints) == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Training loss: {running_loss / len(data['x_train'])}")
        self.train_loss = loss_list
        self.test_loss  = loss_test_list

# -----------------------------------------------------------------------------
#                               Deep Ensemble
# -----------------------------------------------------------------------------
class DeepEnsemble:
    def __init__(self, model_class, n_ensembles, *model_args, **model_kwargs):
        """
        Initializes the Deep Ensemble Neural Networks.
        
        Args:
            model_class: The model class to instantiate.
            n_ensembles: Number of models in the ensemble.
            *model_args: Positional arguments for model initialization.
            **model_kwargs: Keyword arguments for model initialization.
        """
        self.models = []
        self.n_ensembles = n_ensembles
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        
        for i in range(n_ensembles):
            model = model_class(*model_args, **model_kwargs)
            model._initialize_weights(seed=i)  # Use different seeds for initialization
            self.models.append(model)
    
    def fit(self, data, **kwargs):
        """
        Trains all models in the ensemble.
        
        Args:
            data: Dataset containing 'x_train', 'y_train', etc.
            **kwargs: Additional keyword arguments for the model's fit method.
        """
        for i, model in enumerate(self.models):
            print(f"Training model {i + 1}/{self.n_ensembles}")
            model.fit(data, **kwargs)
            
    def fit_bayes(self, data,
                    epochs=1000, 
                    criterion=None, 
                    optimizer=None, 
                    weight_decay=0.0,
                    batch_size=None,
                    learning_rate=0.01,
                    verbose=True,
                    n_prints=20, # if verbose, how many times should plot the loss in the console
                    use_gpu=True,
                    plot_loss=True,
                    ):
        
        """Trains the model"""
        
        # Create a generator to run backprop on all the parameters of all the NNs together.
        all_params = (p for model in self.models for p in model.parameters())
        
        if optimizer is None:
            optimizer=optim.Adam(all_params, lr=learning_rate)
        
        # initialize batch size for both training and testing datasets
        if batch_size is None:
            batch_size = len(data['x_train'])
        if 'x_val' in data:
            batch_size_test = batch_size
            if len(data['x_val']) < batch_size_test:
                batch_size_test = len(data['x_val'])
        
        # initialize the loss function
        if criterion is None:
            criterion = gaussian_NLL
            
        if use_gpu is True:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # training mode, and move the ensembles to gpu if needed
        for model in self.models:
            model.train()
            model.to(device)
            
        n_samples = len(self.models)
        loss_list = []
        loss_test_list = []
        # Start training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for idx in ordered_indices_generator(len(data['x_train']), batch_size):
                X_batch = data['x_train'][idx, :].to(device)
                Y_batch = data['y_train'][idx, :].to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize (using incremental average and std)
                mean_output = 0.0
                mean_square_output = 0.0
                for model in self.models:

                    out = model.forward(X_batch)
                    mean_output += out
                    mean_square_output += out ** 2 # needed for the incremental computation of std.
                    # math behind the formula: http://datagenetics.com/blog/november22017/index.html
                
                mean_output /= n_samples
                mean_square_output /= n_samples
                std_output = torch.sqrt(mean_square_output - mean_output ** 2)
                
                loss = criterion(Y_batch, mean_output, std_output)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            # Save training loss
            loss_list.append(running_loss)
            
    def predict(self, x):
        """
        Makes predictions using the ensemble.
        
        Args:
            x: Input tensor.
        
        Returns:
            A tuple (mean_prediction, uncertainty), where:
            - mean_prediction: Mean prediction across the ensemble.
            - uncertainty: std of the predictions across the ensemble.
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions.append(model(x).cpu().numpy())
        
        predictions = np.array(predictions)  # Shape: (n_ensembles, batch_size, output_dim)
        mean_prediction = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)  # std as uncertainty
        return mean_prediction, uncertainty
    
# -----------------------------------------------------------------------------
#                   Probabilistic BackPropagation (PBP)
# -----------------------------------------------------------------------------
class PBPLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.1):
        super(PBPLayer, self).__init__()
        # Variational parameters for the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, prior_std))
        self.weight_var = nn.Parameter(torch.Tensor(out_features, in_features).fill_(prior_std**2))
        
        # Variational parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, prior_std))
        self.bias_var = nn.Parameter(torch.Tensor(out_features).fill_(prior_std**2))

    def forward(self, x):
        # Forward propagation with uncertainty
        # Compute the mean of the output
        mean = F.linear(x, self.weight_mu, self.bias_mu)
        
        # Compute the variance of the output
        weight_var = self.weight_var.abs()
        bias_var = self.bias_var.abs()
        x_var = x.pow(2).matmul(weight_var.T)
        var = x_var + bias_var

        return mean, var

    def update_posterior(self, x, y, noise_var):
        """
        Updates the posterior distribution parameters using moment matching.
        """
        weight_var = self.weight_var.abs()
        bias_var = self.bias_var.abs()

        # Predictive mean and variance for the current input
        mean, var = self.forward(x)
        var += noise_var  # Add noise variance

        # Compute precision (inverse variance)
        precision = 1 / var

        # Update rules for weight_mu and weight_var
        gradient_mu = precision * (y - mean)
        self.weight_mu.data += (x.T @ gradient_mu).T
        self.weight_var.data = 1 / (1 / weight_var + x.pow(2).sum(dim=0).T * precision)

        # Update rules for bias_mu and bias_var
        self.bias_mu.data += gradient_mu.sum(dim=0)
        self.bias_var.data = 1 / (1 / bias_var + precision.sum(dim=0))

class PBPBayesianNN(nn.Module):
    def __init__(self, input_dim, n_layers, n_neurons, output_dim, tr_layers=0, prior_std=0.1):
        super(PBPBayesianNN, self).__init__()
        
        # Create a list to hold the layers
        self.layers = nn.ModuleList()

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

        # Hidden layers
        for i in range(len(n) - 1):
            self.layers.append(PBPLayer(n[i], n[i + 1], prior_std=prior_std))
    
    def forward(self, x):
        mean, var = x, torch.zeros_like(x)
        for layer in self.layers:
            mean, var = layer(mean)
        return mean, var

    def train_step(self, x, y, noise_var):
        """
        Perform a training step using PBP updates.
        """
        for layer in self.layers:
            layer.update_posterior(x, y, noise_var)

    def predict(self, x):
        """
        Predict the output distribution given an input.
        """
        self.eval()
        mean, var = self.forward(x)
        return mean, torch.sqrt(var)

def forward_MC(self, x, n_samples=1000):
    """Performs inference with a MC simulation for every sample, computing the mean and standard deviation incrementally."""
    mean_output = 0.0
    mean_square_output = 0.0
    
    for _ in range(n_samples):
        out = self.forward(x)
        mean_output += out
        mean_square_output += out ** 2 # needed for the incremental computation of std.
        # math behind the formula: http://datagenetics.com/blog/november22017/index.html
    
    mean_output /= n_samples
    mean_square_output /= n_samples
    
    # Calculate standard deviation
    std_output = torch.sqrt(mean_square_output - mean_output ** 2)
    
    return mean_output, std_output

def inference(model, x, n_samples=100, train=False, gpu=True):
    def test(model, x, n_samples, train=False, device=None):
        if isinstance(model, MC_Dropout):
            model.enable_mc_dropout()
            
        x.to(device)
        model.to(device)
        
        if train:
            model.train()
        else:
            model.eval()
            
        mean_output = 0.0
        mean_square_output = 0.0
        for _ in range(n_samples):
            if train is False:
                with torch.no_grad():
                    if isinstance(model, MC_Dropout):
                        out = model.forward(x, smooth=True)
                    else:
                        out = model.forward(x)
            else:
                if isinstance(model, MC_Dropout):
                    out = model.forward(x, smooth=True)
                else:
                    out = model.forward(x)
                    
            mean_output += out
            mean_square_output += out ** 2 # needed for the incremental computation of std.
            # math behind the formula: http://datagenetics.com/blog/november22017/index.html
        return mean_output, mean_square_output

    if gpu:
        # Use GPU acceleration
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")    

    if isinstance(model, DeepEnsemble):
        mean_output = 0.0
        mean_square_output = 0.0
        for NN in model.models:
            if isinstance(NN, FCNN):
                n_samples=1
                NN.to(device)
                with torch.no_grad():
                    output = NN.forward(x)
                mean_output += output
                mean_square_output += output ** 2
            else:
                output, square = test(NN, x, n_samples, device=device)
                mean_output += output
                mean_square_output += square
    else:
        mean_output, mean_square_output = test(model, x, n_samples, device=device)
        
    if isinstance(model, DeepEnsemble):
        n_ensembles = model.n_ensembles
    else:
        n_ensembles = 1
        
    mean_output /= n_samples*n_ensembles
    mean_square_output /= n_samples*n_ensembles
    
    # Calculate standard deviation
    std_output = torch.sqrt(mean_square_output - mean_output ** 2)
    
    return mean_output, std_output

# %% Conversion Utilities
def fcnn_to_bnn(fcnn, prior_std=0.001, verbose=False):
    bnn = BayesianNN(*fcnn.init_params, prior_std=prior_std) # initialize the BNN with the same shape of the FCNN
    
    if verbose:
        print("\n----------------------------------------------------------------------"
              "\n Loading FCNN weights...")
    
    # Initialize BayesianLinear layers with FCNN weights
    with torch.no_grad():  # Disable gradient tracking
        for fcnn_layer, bnn_layer in zip(fcnn.layers, bnn.layers):
            # Get the weights and biases from FCNN
            bnn_layer.weight_mu.copy_(fcnn_layer.weight)  # Set mean weights
            if bnn_layer.bias_mu is not None:
                bnn_layer.bias_mu.copy_(fcnn_layer.bias)  # Set mean biases (if they exist)
    
    if verbose:
        print("\nInitialized Bayesian NN with FCNN weights.")
    
    return bnn


