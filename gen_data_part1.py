import numpy as np

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

file = open("training_data.npy", 'w')

for iter in range(10000):
    training_data[iter, np.random.randint(4)] = 1
<<<<<<< HEAD
=======

np.save(file, training_data)
print(training_data)
>>>>>>> 33854346a7992c3a6094a94c30cd289e4bfc4cc9
