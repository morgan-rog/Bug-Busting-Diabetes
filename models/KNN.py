import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

def knn_param_selector():
    model = KNeighborsClassifier(n_neighbors=2)
    
    return model