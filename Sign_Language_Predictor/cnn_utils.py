import numpy as np 
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf 
import math

def load_dataset():
	train_dataset=h5py.File('dataset/train_signs.h5',"r")
	train_set_x_orig=np.array(train_dataset['train_set_x'][:])
	train_set_y_orig=np.array(train_dataset['train_set_y'][:])
	test_dataset=h5py.File('dataset/test_signs.h5',"r")
	test_set_x_orig=np.array(test_dataset['test_set_x'][:])
	test_set_y_orig=np.array(test_dataset['test_set_y'][:])
	classes=np.array(test_dataset['list_classes'][:])
	train_set_y_orig=train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
	test_set_y_orig=test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
	return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

'''train_dataset=h5py.File('dataset/train_signs.h5',"r")
train_set_y_orig=np.array(train_dataset['train_set_y'][:])
#print(train_set_y_orig)
train_set_y_orig=train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
print(train_set_y_orig.shape)

#def convert_to_one_hot(Y,C):

Y=np.eye(6)[train_set_y_orig.reshape(-1)].T
print(Y)'''

def convert_to_one_hot(Y,C):
	Y=np.eye(C)[Y.reshape(-1)].T
	return Y
def random_mini_batches(X,Y,mini_batch_size=64):
	m = X.shape[0]                  # number of training examples
	mini_batches = []
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionnin
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches

    