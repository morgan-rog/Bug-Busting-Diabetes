import streamlit as st
from sklearn.svm import SVC, LinearSVC

def svm_param_selector():
    model = SVC()
    
    return model