import numpy as np
import streamlit as st

from models.BugBustersNN import bbnn_param_selector
from models.SimpleNeuralNetwork import snn_param_selector
from models.KNN import knn_param_selector
from models.SVM import svm_param_selector


def introduction():
    st.title("**Welcome to Bug Busting Diabetes**")
    st.subheader("By The Bug Busters")
    st.markdown(
        """
        ---

        This is a place where you can experiment with different Neural Network models on
        the Pima Indians Diabetes Dataset from Kaggle.

        [The Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) is
        originally from the National Institute of Diabetes and Kidney Diseases. The objective of the dataset
        is to diagnostically predict whether or not a patient has diabetes.

        The training accuracy is plotted for the Neural Network models and the [Receiver Operating Characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
        is plotted for the KNN machine learning model and the confusion matrix is plotted for the SVM machine
        learning model.

        The accuracy of the models are from their predictions on the training and testing data.
        
        """

    )


def model_selector():
    model_training_container = st.sidebar.expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Bug Buster's Neural Network",
                "Simple Neural Network",
                "KNN",
                "SVM",
            ),
        )

        if model_type == "Bug Buster's Neural Network":
            model = bbnn_param_selector()

        if model_type == "Simple Neural Network":
            model = snn_param_selector()
        
        if model_type == "KNN":
            model = knn_param_selector()

        if model_type == "SVM":
            model = svm_param_selector()

    return model_type, model