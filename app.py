import numpy as np
import streamlit as st

from utils.functions import (
    generate_data,
    plot_confusion_matrix,
    train_NN_model,
    train_ML_model,
    get_NN_model_summary,
    get_ML_model_summary,
    get_NN_model_accuracy,
    get_ML_model_accuracy,
    plot_NN_accuracy,
    plot_confusion_matrix,
    get_model_url,
    get_model_info,
)

from utils.ui import (
    introduction,
    model_selector,
)

st.set_page_config(
    page_title="Bug Busting Diabetes", layout="wide"
)

def side_bar_controllers():
    model_type, model = model_selector()
    X_train, X_test, y_train, y_test = generate_data()

    return(model_type, model, X_train, X_test, y_train, y_test)


def body(X_train, X_test, y_train, y_test, model, model_type):
    introduction()
    if model_type == "Morgan's model" or model_type == "Simple Neural Network":
        (model, history, duration) = train_NN_model(model, X_train, X_test, y_train, y_test)
        model_info = get_model_info(model_type)
        model_summary = get_NN_model_summary(model)
        model_url = get_model_url(model_type)
        model_accuracy = get_NN_model_accuracy(model, X_test, y_test)
        plt = plot_NN_accuracy(history)
    else:
        model, duration = train_ML_model(model, X_train, X_test, y_train, y_test)
        model_info = get_model_info(model_type)
        model_url = get_model_url(model_type)
        model_summary = get_ML_model_summary(model, X_test, y_test)
        model_accuracy = get_ML_model_accuracy(model, X_test, y_test)
        plt = plot_confusion_matrix(model, X_test, y_test)

    col1, col2 = st.columns((1,1))

    with col1:
        plot_placeholder = st.empty()
        accuracy_placeholder = st.empty()

    with col2:
        duration_placeholder = st.empty()
        info_placeholder = st.empty()
        model_url_placeholder = st.empty()
        summary_placeholder = st.empty()

    plot_placeholder.pyplot(fig=plt)
    accuracy_placeholder.info(model_accuracy)

    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    info_placeholder.info(model_info)
    summary_placeholder.info(model_summary)


if __name__ == "__main__":
    (model_type, model, X_train, X_test, y_train, y_test) = side_bar_controllers()
    body(X_train, X_test, y_train, y_test, model, model_type)