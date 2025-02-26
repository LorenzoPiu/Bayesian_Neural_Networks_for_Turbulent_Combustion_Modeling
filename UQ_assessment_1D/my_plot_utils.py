#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:29:54 2024

@author: lpiu
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
import os
from _utils import f, load_training_data, set_seed, set_plotting_preferences

def plot_training_data(training_folder='training_data'):
    data = load_training_data(training_folder)
    
    plt.figure(figsize=(8, 5), dpi=600)
    plt.scatter(data['x_train'].numpy().flatten()        , data['y_train'].numpy().flatten()       , c='#f0f0f0', marker='o', s=7, alpha=0.08, edgecolors='#a0a0a0')
    plt.scatter([]                         , []                        , c='#f0f0f0', edgecolors='#a0a0a0', marker='.', s=90, alpha=1  , label='Training/Validation points')

    # Line plot of the function
    steps       = 1000
    x_ptp       = 2
    x_lin       = torch.linspace(-1-x_ptp/4, 1+x_ptp/4, steps)
    x_lin       = torch.reshape(x_lin, [len(x_lin), 1])
    y_lin       = f(x_lin, noise=0)
    plt.plot(x_lin, y_lin, c='red', linewidth=2.5, label='y = f(x)')

def fig_style():
    # Labels, legend, and title
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('2D Histogram with Function Overlay')
    plt.legend(loc='upper center')
    plt.grid(visible=True, linestyle='--', alpha=0.6)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-3, 7)

def plot_model_output_kde(y_model_mean, y_model_sigma, data, model_folder, x_ptp=2, steps=1000):
    set_plotting_preferences()

    # Set up the grid for two plots (one for the KDE plot above, and the main plot below)
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.3, 1])  # 0.3 for KDE plot, 1 for main plot

    # First axis (for the KDE plot)
    ax0 = plt.subplot(gs[0])
    
    x_lin = torch.linspace(-1 - x_ptp / 4, 1 + x_ptp / 4, steps)
    x_lin = torch.reshape(x_lin, [len(x_lin), 1])
    y_lin = f(x_lin, noise=0)

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
    ax1.plot(x_lin, y_model_mean, c='blue', linestyle='--', linewidth=2, label='MCD mean prediction')

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
    plt.savefig(os.path.join(model_folder, 'model_output_kde.png'), dpi=600)

    # Show the plot
    plt.show()
    
def plot_model_output(y_model_mean, y_model_sigma, data, model_folder, x_ptp=2, steps=1000):
    set_plotting_preferences()

    # Set up the grid for the main plot only (no upper plot)
    fig = plt.figure(figsize=(8, 5))

    # Only one axis for the main plot
    ax1 = fig.add_subplot(111)

    # Scatter plot of training data
    ax1.scatter(data['x_train'].numpy().flatten(), data['y_train'].numpy().flatten(), c='#f0f0f0', marker='o', s=7, alpha=0.08, edgecolors='#FFC867')
    ax1.scatter([], [], c='#FFC867', edgecolors='#FFC867', marker='.', s=90, alpha=1, label='Training/Validation points')

    # Line plot of the function
    x_lin = torch.linspace(-1 - x_ptp / 4, 1 + x_ptp / 4, steps)
    x_lin = torch.reshape(x_lin, [len(x_lin), 1])
    y_lin = f(x_lin, noise=0)
    ax1.plot(x_lin, y_lin, c='#da0003', linewidth=2.5, label='y = f(x)')

    # Line plot of the model
    ax1.plot(x_lin, y_model_mean, c='blue', linestyle='--', linewidth=2, label='MCD mean prediction')

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

    ax1.set_xlim(x_lin.min(), x_lin.max())
    ax1.set_ylim(-2, 4.7)
    ax1.set_yticks([0, 2, 4])

    # Tight layout for a cleaner look
    plt.tight_layout()

    # Save the figure with the main plot
    plt.savefig(os.path.join(model_folder, 'model_output.png'), dpi=600)

    # Show the plot
    # plt.show()
