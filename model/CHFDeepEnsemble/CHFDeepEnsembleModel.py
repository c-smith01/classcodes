"""
Created on Mon Oct 23

@author: c-smith
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
#from chf_nn import chf_nn
from model.CHFDeepEnsemble.args_modified import args, device
import torch.nn as nn

def activation(name, *args):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    else:
        raise ValueError('Unknown activation function')

class chf_de_nn(nn.Module):
    def __init__(self, input_size, hidden_units, output_size, activation_fn, dropout_prob): # 3/19/2024 - removed dropout_prob=0.5 statement as it may be responsible ofr bottlenecking
        """
        :param input_size: The number of input features.
        :param hidden_units: A list where each item is the number of hidden units in the corresponding layer.
        :param output_size: The number of output target(s).
        :param activation_fn: The name of the activation function to use in the hidden layers.
        :param dropout_prob: The dropout probability.
        """
        super(chf_de_nn, self).__init__()

        layers = []

        # Previous layer's output size. Initialized to input size.
        prev_units = input_size

        # Iterate through each number of #from CHF.model.CHFDeepEnsemble.args_modified import args, devicehidden units to create layers
        for index, units in enumerate(hidden_units):
            layers.append(nn.Linear(prev_units, units))
            if index == len(hidden_units)-1:
                layers.append(nn.Identity())
            else:
                layers.append(activation(activation_fn))  # Activation function; To avoid flat prediction crives add an identity matrix to the final activation function layer
            layers.append(nn.Dropout(p=dropout_prob))  # Dropout layer
            prev_units = units  # Set previous layer's output size for next iteration
        
        # Add output layer
        layers.append(nn.Linear(prev_units, output_size))
#        layers.append(nn.ReLU())  # Ensure output is non-negative

        # Assuming a regression task; for classification, you might want to add Softmax or Sigmoid at the end.
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
