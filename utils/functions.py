import time

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

from models.utils import model_infos, model_urls


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def generate_data():
    data = pd.read_csv('diabetes_data.csv')
    
    X = data.drop('Outcome', axis=1).values
    y = data.Outcome.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    normalizer = Normalizer()
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)

    return X_train, X_test, y_train, y_test


def train_NN_model(model, X_train, X_test, y_train, y_test):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    t0 = time.time()
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)
    duration = time.time() - t0
    return model, history, duration


def train_ML_model(model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0
    return model, duration


def get_NN_model_summary(model):
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    join_summary = "\n".join(summary_list[3:])
    
    summary = "Summary of model:\n" + join_summary
    return summary


def get_ML_model_summary(model, X_test, y_test):
    space = "            "
    prediction = model.predict(X_test)
    class_report = classification_report(y_test, prediction, output_dict=True)
    summary = "Classification Report:\n\n"
    summary +=   "precision " + "recall " + "f1-score " + "support\n\n"
    for label_dict in class_report:
        if label_dict == "accuracy":
            continue
        summary += label_dict + ":"
        for key in class_report[label_dict]:
            val = round(class_report[label_dict][key], 2)
            val = str(val)
            summary += val
        summary += "\n\n"

    return summary


def get_NN_model_test_accuracy(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracy = "Testing Accuracy: %.2f%%" % (scores[1]*100)
    return accuracy


def get_NN_model_train_accuracy(model, X_train, y_train):
    scores = model.evaluate(X_train, y_train, verbose=0)
    accuracy = "Training Accuracy: %.2f%%" % (scores[1]*100)
    return accuracy


def get_ML_model_test_accuracy(model, X_test, y_test):
    scores = round(model.score(X_test, y_test)*100, 2)
    accuracy = f"Testing Accuracy: {scores}%"
    return accuracy


def get_ML_model_train_accuracy(model, X_train, y_train):
    scores = round(model.score(X_train, y_train)*100, 2)
    accuracy = f"Training Accuracy: {scores}%"
    return accuracy


def plot_NN_accuracy(history):
    acc_line = history.history['accuracy']

    plt.plot(acc_line)
    plt.title('Model Training Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    return plt


def plot_confusion_matrix(model, X_test, y_test):
    prediction = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, prediction)
    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16})
    return plt


def get_model_url(model_type):
    model_url = model_urls[model_type]
    text = f"**Link to model source [here]({model_url})**"
    return text


def get_model_info(model_type):
    model_info = model_infos[model_type]
    return model_info