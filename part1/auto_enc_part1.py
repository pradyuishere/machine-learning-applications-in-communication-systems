import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam
from keras.models import Model
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

###############################################################################
file_train = open("training_data.npy", "r")
file_test = open("test_data.npy", "r")

training_data = np.load(file_train)
test_data = np.load(file_test)

file_train.close()
file_test.close()
###############################################################################
learning_rate = 0.001
input_dim = 16
encoding_dim = 2
midlayer_dim = int((input_dim+encoding_dim)/2) +1
print ("midlayer_dim :", midlayer_dim )
energy_per_bit = 5

input_msg = Input(shape = (input_dim, ))

encoded = Dense(input_dim, activation='relu')(input_msg)
encoded2 = Dense(encoding_dim, activation = 'linear')(encoded)
#encoded3 = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=0))(encoded2)
#encoded3 = keras.layers.Lambda(lambda x:np.sqrt(encoding_dim)*keras.backend.l2_normalize(x, axis=1))(encoded2)
encoded3 = keras.layers.BatchNormalization(axis=1)(encoded2)
encoded4 = keras.layers.GaussianNoise(np.sqrt(encoding_dim/(2*input_dim*energy_per_bit)))(encoded3)

decoded3 = Dense(midlayer_dim, activation = 'relu')(encoded4)
decoded  = Dense(input_dim, activation = "softmax") (decoded3)

adam = Adam(learning_rate)
sgd = SGD(learning_rate)

autoencoder = Model(input_msg, decoded)
autoencoder.compile(optimizer = adam, loss='categorical_crossentropy')

###############################################################################
##Preparing the encoder and the decoder
encoder = Model(input_msg, encoded3)

encoded_input = Input(shape=(encoding_dim, ))
decoder_layer1 = autoencoder.layers[-2](encoded_input)
decoder_layer2 = autoencoder.layers[-1](decoder_layer1)
#decoder_layer3 = autoencoder.layers[-1](decoder_layer2)

decoder = Model(encoded_input, decoder_layer2)
###############################################################################

autoencoder.fit(training_data, training_data, epochs=100, batch_size=50, shuffle=True, validation_data=(test_data, test_data))
###############################################################################
test_predictions = encoder.predict(test_data)

plt.scatter(test_predictions[:, 0], test_predictions[:, 1])
plt.show()
