# Project 1/2 Model for Submission to Project
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import pickle

# Accessing provided data
with open('CSCE_636_Proj_Files/DS-14-samples_n_k_m_P.pkl') as file:
    loaded_input_features = pickle.load(file)
    
print(loaded_input_features)

with open('CSCE_636_Proj_Files/DS-14-samples_n_k_m_P.pkl') as file:
    loaded_output_features = pickle.load(file)
    
print(loaded_output_features)

