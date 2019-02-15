import keras
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

###############################################################################
file = open("training_data.npy", "r")
training_data = np.load(training_data)
