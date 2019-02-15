import numpy as np
import matplotlib.pyplot as plt

###############################################################################
##Defining the necessities
#Defining (n, k)
num_channels = 2
num_messages = 4
num_bits = 2
###############################################################################
##Generating the training data, one hot vectors of width num_messages
num_samples = 10000
training_data = np.zeros([num_samples, num_messages])

for iter in range(10000):
    training_data[iter, np.randint(4)] = 1

print(training_data)
