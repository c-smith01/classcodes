# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:29:56 2023

@author: Yang
"""

import torch.nn as nn
# Debug Notes: 11/27/23 sp() should be applied to model in order to correct for signma values in order to obtain, also model does not need input and output feature size parameters passed to it 

class chf_nn(nn.Module):
    def __init__(self, input_size, hidden_units,output_size):
        """
        :param input_size: The number of input features.
        :param hidden_units: A list where each item is the number of hidden units in the corresponding layer.
        """
        super(chf_nn, self).__init__()

        layers = []

        # Previous layer's output size. Initialized to input size.
        prev_units = input_size

        # Iterate through each number of hidden units to create layers
        for units in hidden_units:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.ReLU())  # Activation function
            prev_units = units  # Set previous layer's output size for next iteration
        
        # Add output layer
        layers.append(nn.Linear(prev_units, output_size))
#        layers.append(nn.ReLU())  # Ensure output is non-negative

        # Assuming a regression task; for classification, you might want to add Softmax or Sigmoid at the end.
        self.network = nn.Sequential(*layers)

        # Ensure that sigm from output feature is passed through softplus to convert values to actual sigma values
        # output[:,1] = sp(output[:,1])

    def forward(self, x):
        return self.network(x)

# Example usage:
#input_size = 7
#hidden_layers = [32, 16, 8]  # 3 hidden layers with 32, 64, and 128 units respectively
#output_size = 1
#model = chf_nn(input_size, hidden_layers,output_size)
print(model)
