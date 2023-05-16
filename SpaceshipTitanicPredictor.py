import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib

st.title('Spaceship Titanic predictor')

def load_model():
    loaded_model = joblib.load("logistic_regression.joblib")
    return loaded_model

def load_data():
    df = pd.read_csv("data/test.csv")
    return df

model = load_model()
data = load_data()

with st.expander("Data"):
    st.write("Datos de entrenamiento:")
    st.write(data.head())

with st.expander("Pestaña 2"):
    st.write("Contenido de la Pestaña 2")

with st.expander("Pestaña 3"):
    st.write("Contenido de la Pestaña 3")