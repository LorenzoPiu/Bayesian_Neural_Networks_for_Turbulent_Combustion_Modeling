#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:18:45 2024

@author: lpiu
"""

import torch
import numpy as np

def check_is_tensor(x):
    if not isinstance(x, torch.tensor()):
        raise TypeError('Input must be tensor type')
    

def gaussian_NLL(y_true, y_pred, sigma):
    """
    Computes the Negative Log-Likelihood (NLL) assuming a Gaussian distribution.

    Parameters:
    - y_true: Torch tensor of true values (N, ).
    - y_pred: Torch tensor of predicted values (N, ).
    - sigma: Torch tensor of standard deviations for predictions (N, ).

    Returns:
    - nll: Negative log-likelihood value (scalar).
    """
    # Input validation
    if not isinstance(y_true, torch.Tensor):
        raise TypeError("y_true must be a PyTorch tensor.")
    if not isinstance(y_pred, torch.Tensor):
        raise TypeError("y_pred must be a PyTorch tensor.")
    if not isinstance(sigma, torch.Tensor):
        raise TypeError("sigma must be a PyTorch tensor.")
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true and y_pred must have the same shape. Got {y_true.shape} and {y_pred.shape}.")
    if y_true.shape != sigma.shape:
        raise ValueError(f"Shape mismatch: y_true and sigma must have the same shape. Got {y_true.shape} and {sigma.shape}.")

    # Clipping sigma to avoid log(0) or division by zero
    epsilon = 1e-15
    sigma = torch.clamp(sigma, min=epsilon)

    # Compute NLL
    nll = (
        0.5 * torch.sum(((y_true - y_pred) ** 2) / (sigma ** 2)) + 
        torch.sum(torch.log(sigma * torch.sqrt(torch.tensor(2 * torch.pi))))
    ) / y_true.size(0)

    return nll

def PICP(y_true, y_low, y_high):
    """
    Computes the Prediction Interval Coverage Probability (PICP).
    
    Parameters:
    - y_true: Torch tensor of true values (N, ).
    - y_low: Torch tensor of lower bounds of prediction intervals (N, ).
    - y_high: Torch tensor of upper bounds of prediction intervals (N, ).
    
    Returns:
    - picp: PICP value (scalar).
    """
    # Input validation
    if not (y_true.shape == y_low.shape == y_high.shape):
        raise ValueError("y_true, y_low, and y_high must have the same shape.")
    
    # Compute the indicator for whether true values fall within prediction intervals
    within_bounds = (y_true >= y_low) & (y_true <= y_high)
    
    # Compute PICP as the mean of the indicators
    picp = within_bounds.float().mean()
    
    return picp

def MPIW(y_low, y_high):
    """
    Computes the Mean Prediction Interval Width (MPIW).
    
    Parameters:
    - y_low: Torch tensor of lower bounds of prediction intervals (N, ).
    - y_high: Torch tensor of upper bounds of prediction intervals (N, ).
    
    Returns:
    - mpiw: MPIW value (scalar).
    """
    # Input validation
    if y_low.shape != y_high.shape:
        raise ValueError("y_low and y_high must have the same shape.")
    
    # Compute the width of each prediction interval
    interval_widths = y_high - y_low
    
    # Compute MPIW as the mean of interval widths
    mpiw = interval_widths.mean()
    
    return mpiw

def MSE(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between true values and predicted values.

    Parameters:
    y_true (array-like): True values
    y_pred (array-like): Predicted values

    Returns:
    float: The Mean Squared Error
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute the squared differences
    squared_errors = (y_true - y_pred) ** 2
    
    # Compute the mean of the squared errors
    mse = np.mean(squared_errors)
    
    return mse



