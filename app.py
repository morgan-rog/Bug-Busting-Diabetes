import numpy as np
import streamlit as st

from utils.functions import (
    generate_data,
    train_model,
    get_model_info,
    get_model_summary,
    get_model_url,
    get_model_accuracy,
    plot_accuracy,
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
    (model, history, duration) = train_model(model, X_train, X_test, y_train, y_test)

    col1, col2 = st.columns((1,1))

    with col1:
        plot_placeholder = st.empty()
        accuracy_placeholder = st.empty()

    with col2:
        duration_placeholder = st.empty()
        info_placeholder = st.empty()
        model_url_placeholder = st.empty()
        summary_placeholder = st.empty()

    model_info = get_model_info(model_type)
    model_summary = get_model_summary(model)
    model_url = get_model_url(model_type)
    model_accuracy = get_model_accuracy(model, X_test, y_test)

    plt = plot_accuracy(model, history)

    plot_placeholder.pyplot(fig=plt)
    accuracy_placeholder.info(model_accuracy)

    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    info_placeholder.info(model_info)
    summary_placeholder.info(model_summary)


if __name__ == "__main__":
    (model_type, model, X_train, X_test, y_train, y_test) = side_bar_controllers()
    body(X_train, X_test, y_train, y_test, model, model_type)