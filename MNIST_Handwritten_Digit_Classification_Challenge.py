# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:13:06 2019

@author: xliu
"""
	
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import time 
from keras.utils import np_utils 

#%% load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#%% plot MINIST dataset, plot 4 images as gray scale
toplot = np.random.randint(np.shape(X_train)[0], size=4)
plt.figure(num=1, figsize = (10,10))
plt.subplot(221)
plt.imshow(X_train[toplot[0]], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[toplot[1]], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[toplot[2]], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[toplot[3]], cmap=plt.get_cmap('gray'))
plt.show()

#%% K-Nearest Neighbors 
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

start = time.clock()
clf_knn = neighbors.KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
print("the score of knn: " + str(clf_knn.score(X_test, y_test)))
print("time consumption : " + str(time.clock()-start) + "[s]")

y_test_pred_knn = clf_knn.predict(X_test)

#%% K-Nesrest Neighbors with PCA
pca = PCA(n_components = 10).fit(X_train)
X_reduce = pca.transform(X_train)

start = time.clock()
clf_pca_knn = neighbors.KNeighborsClassifier()
clf_pca_knn.fit(X_reduce, y_train)

print("the score off knn+pca: "+ str(clf_pca_knn.score(pca.transform(X_test), y_test)))
print("time consumption : " + str(time.clock()-start) + "[s]")

y_test_pred_pca_knn = clf_pca_knn.predict(pca.transform(X_test))

#%% one hidden layer neural network 
X_train_nor = X_train/255
X_test_nor = X_test/255

y_train_onehot =  np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)
num_classes = y_train_onehot.shape[1]

model = Sequential()
model.add(Dense(units = num_pixels, activation='relu', input_dim = num_pixels))
model.add(Dense(units = num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start = time.clock()
model.fit(X_train_nor, y_train_onehot, epochs = 10, batch_size = 50)
loss_and_metrics = model.evaluate(X_test_nor, y_test_onehot, batch_size=50)

print("the score of MLP: " + str(loss_and_metrics[1]))
print("time consumption : " + str(time.clock()-start) + "[s]")

y_test_pred_mlp = model.predict(X_test_nor, batch_size = 50)

#%% simple convolution neural network 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start = time.clock()
model.fit(X_train/255, y_train_onehot, epochs = 10, batch_size = 50)
loss_and_metrics_cnn = model.evaluate(X_test/255, y_test_onehot, batch_size=50)

print("the score of MLP: " + str(loss_and_metrics_cnn[1]))
print("time consumption : " + str(time.clock()-start) + "[s]")

y_test_pred_mlp = model.predict(X_test/255, batch_size = 50)






























