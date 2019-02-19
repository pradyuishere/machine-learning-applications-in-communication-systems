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

file = open("training_data-a.npy", 'w')
file1 = open("training_data-b.npy", 'w')

for iter in range(num_samples):

    training_data[iter, np.random.randint(num_messages)] = 1
np.save(file, training_data)
print(training_data)

training_data = np.zeros([num_samples, num_messages])

for iter in range(num_samples):
    training_data[iter, np.random.randint(num_messages)] = 1

np.save(file1, training_data)
print(training_data)
file.close()
file1.close()
###############################################################################
##Generating the test data, one hot vectors of width num_messages
num_samples = 5000
test_data = np.zeros([num_samples, num_messages])

file = open("test_data-a.npy", 'w')
file1 = open("test_data-b.npy", 'w')

for iter in range(num_samples):
    test_data[iter, np.random.randint(num_messages)] = 1

np.save(file, test_data)
print(test_data)

test_data = np.zeros([num_samples, num_messages])

for iter in range(num_samples):
    test_data[iter, np.random.randint(num_messages)] = 1

np.save(file1, test_data)

file.close()
file1.close()
