import streamlit as st
from keras.models import Model, Sequential, load_model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Activation

def bbnn_param_selector():
    inputs = Input(name='inputs', shape=[8,])
    layer = Dense(128)(inputs)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dense(1)(layer)
    outputs = Activation('sigmoid')(layer)

    model = Model(inputs=inputs, outputs=outputs)
    return model
