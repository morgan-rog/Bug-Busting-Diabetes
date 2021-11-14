import numpy as np
import streamlit as st

from models.Morgan import mr_param_selector
from models.SimpleNeuralNetwork import snn_param_selector


def introduction():
    st.title("**Welcome to Bug Busting Diabetes**")
    st.markdown(
        """
        This is a place where you can experiment with different Neural Network models on
        the Pima Indians Diabetes Dataset from Kaggle.

        [The Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) is
        originally from the National Institute of Diabetes and Kidney Diseases. The objective of the dataset
        is to diagnostically predict whether or not a patient has diabetes.
        
        """

    )



def model_selector():
    model_training_container = st.sidebar.expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Morgan's model",
                "Simple Neural Network",
            ),
        )

        if model_type == "Morgan's model":
            model = mr_param_selector()

        if model_type == "Simple Neural Network":
            model = snn_param_selector()

    return model_type, model