from pathlib import Path
import base64
import time

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt

from models.utils import model_infos, model_urls


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def generate_data():
    data = pd.read_csv('diabetes_data.csv')
    data.SkinThickness.replace(0, data.SkinThickness.median(), inplace=True)
    data.Insulin.replace(0, data.Insulin.median(), inplace=True)
    data.Glucose.replace(0, data.Glucose.median(), inplace=True)
    data.BloodPressure.replace(0, data.BloodPressure.median(), inplace=True)
    data.BMI.replace(0, data.BMI.median(), inplace=True)
    
    X = data.drop('Outcome', axis=1).values
    y = data.Outcome.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    normalizer = Normalizer()
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, X_test, y_train, y_test):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    t0 = time.time()
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
    duration = time.time() - t0

    return model, history, duration


def get_model_info(model_type):
    model_info = model_infos[model_type]
    return model_info


def get_model_summary(model):
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    join_summary = "\n".join(summary_list[3:])
    
    summary = "Summary of model:\n" + join_summary
    return summary


def get_model_url(model_type):
    model_url = model_urls[model_type]
    text = f"**Link to model source [here]({model_url})**"
    return text


def get_model_accuracy(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracy = "Accuracy: %.2f%%" % (scores[1]*100)
    return accuracy


def plot_accuracy(model, history):
    acc_line = history.history['accuracy']

    plt.plot(acc_line)
    plt.title('Model Training and Validation Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    return plt