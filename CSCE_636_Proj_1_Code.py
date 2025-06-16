# Project 1/2 Model for Submission to Project
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle

# Accessing provided data
with open('CSCE_636_Proj_Files/DS-14-samples_n_k_m_P', 'rb') as file:
    loaded_input_features = pickle.load(file)
    
print(loaded_input_features)

with open('CSCE_636_Proj_Files/DS-14-samples_mHeights.pkl', 'rb') as file:
    loaded_output_features = pickle.load(file)
    
print(loaded_output_features)


train_data = []
test_data = []
validation_data = []

# initial model
input_size=6
output_size=1




