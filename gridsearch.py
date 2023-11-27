import time
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers, initializers
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import cifar10

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

t1 = time.time()
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('Training set', x_train.shape, y_train.shape)
print('Test set', x_test.shape, y_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def create_model(optimizer='rmsprop'):
    #create sequential keras model
    model = Sequential()
    
    #input layers
    model.add(Conv2D(input_shape=(32, 32, 3), 
                 filters=32, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
    model.add(Conv2D(filters=32, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))
    # Conv Stack 2
    model.add(Conv2D(filters=64, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
    model.add(Conv2D(filters=64, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))
# Conv Stack 3
    model.add(Conv2D(filters=128, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
    model.add(Conv2D(filters=128, 
                 kernel_size=(3, 3),
                 padding='same', 
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))
#↨ FC stack
    model.add(Flatten())
    model.add(Dense(units=512, 
                activation='relu',
                kernel_initializer=initializers.he_normal(),
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, 
                activation='relu',
                kernel_initializer = initializers.he_normal(),
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model

def gridSearchScript(optimizers=['rmsprop'], init=['he-normal'], epochs=[5], batches=[32]):#, cv_stack=[2], kl_size=[(3,3)]): #, cv_filters=[[32,64,128]], dense_size_candidates=[[512,128]]):
#Params and Grid definition
    param_grid = dict(optimizer=optimizers,
                      #init=init,                       
                      epochs=epochs, 
                      batch_size=batches)
                      #cv_stack=cv_stack,
                      #kl_size=kl_size)
                      #cv_filters=cv_filters,
                      #dense_size_candidates=dense_size_candidates)

    model = KerasClassifier(build_fn=create_model, verbose=2)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(x_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    t2 = time.time()
    print("Time: %0.2fs" % (t2 - t1))
    
gridSearchScript(batches=[32, 64, 128])