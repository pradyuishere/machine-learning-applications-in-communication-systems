import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam
from keras.models import Model
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf

###############################################################################
file_train = open("training_data-a.npy", "rb")
file_train1 = open("training_data-b.npy", "rb")
file_test = open("test_data-a.npy", "rb")
file_test1 = open("test_data-b.npy", "rb")

training_data_a = np.load(file_train)
test_data_a = np.load(file_test)
training_data_b = np.load(file_train1)
test_data_b = np.load(file_test1)

training_data = np.concatenate((training_data_a, training_data_b), axis = 1)
test_data  = np.concatenate((test_data_a, test_data_b), axis = 1)

file_train.close()
file_test.close()
file_train1.close()
file_test1.close()

print(training_data.shape)
print(test_data.shape)
###############################################################################
learning_rate = 0.001
msg_len = 4
input_dim = 2*msg_len
channel_num = 2
encoding_dim = 2*channel_num
midlayer_dim = int((msg_len+channel_num)/2)+1
print ("midlayer_dim :", midlayer_dim )
energy_per_bit1 = 5
energy_per_bit2 = 5
###############################################################################
input_msg = Input(shape = (input_dim, ))
print("Input shape : ", input_dim)
# input_msg2 = Input(shape = (msg_len, ))


lambda_l1_1 = keras.layers.Lambda(lambda x : x[0:msg_len, :])(input_msg)
encoded_l1_1 = Dense(msg_len, activation = 'relu')(lambda_l1_1)
encoded_l1_2 = Dense(channel_num, activation = 'linear')(encoded_l1_1)
encoded_l1_3 = keras.layers.BatchNormalization(axis=1)(encoded_l1_2)

lambda_l2_1 = keras.layers.Lambda(lambda x : x[msg_len:2*msg_len, :])(input_msg)
encoded_l2_1 = Dense(msg_len, activation = 'relu')(lambda_l2_1)
encoded_l2_2 = Dense(channel_num, activation = 'linear')(encoded_l2_1)
encoded_l2_3 = keras.layers.BatchNormalization(axis=1)(encoded_l2_2)

added_layer = keras.layers.Add()([encoded_l1_3, encoded_l2_3])
encoded_l1_4 = keras.layers.GaussianNoise(np.sqrt(channel_num/(2*msg_len*energy_per_bit1)))(added_layer)
encoded_l2_4 = keras.layers.GaussianNoise(np.sqrt(channel_num/(2*msg_len*energy_per_bit2)))(added_layer)

decoded_l1_1 = Dense(midlayer_dim, activation = 'relu')(encoded_l1_4)
decoded_l2_1 = Dense(midlayer_dim, activation = 'relu')(encoded_l2_4)

decoded_l1_2 = Dense(midlayer_dim, activation = 'softmax')(decoded_l1_1)
decoded_l2_2 = Dense(midlayer_dim, activation = 'softmax')(decoded_l2_1)

concat_layer = keras.layers.concatenate([decoded_l1_2, decoded_l2_2])

decoded_out = Dense(input_dim)(concat_layer)
################################################################################
adam = Adam(learning_rate)
autoencoder = Model(inputs = input_msg, outputs = decoded_out)
autoencoder.compile(optimizer = adam, loss='categorical_crossentropy')
################################################################################
encoder1 = Model(input_msg, encoded_l1_3)
encoder2 = Model(input_msg, encoded_l2_3)
################################################################################
print(training_data.shape)
print(keras.backend.shape(input_msg))
autoencoder.fit(training_data,training_data, epochs=100, batch_size=4, shuffle=True,validation_data=(test_data, test_data))
