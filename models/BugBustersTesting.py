import streamlit as st
from keras.models import Model, Sequential, load_model
from keras.layers import Input, BatchNormalization, Dropout
from keras.layers.core import Dense, Activation

def bbtesting_param_selector():
    # create a Sequential model
    model = Sequential()

    # Input Layer
    model.add(Dropout(0.2, input_shape=(8,)))

    # hidden layers
    model.add(Dense(32, activation='relu'))

    model.add(Dense(20, activation='relu')) 

    model.add(Dense(10, activation='relu'))

    # output layer
    model.add(Dense(1, activation='sigmoid'))
    return model