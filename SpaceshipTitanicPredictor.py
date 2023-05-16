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

with st.expander("Inferencia"):
    st.text("Inputs para modelo")
    
    is_vip = st.checkbox("¿Es VIP el pasajero?")
    vrdeck_expense = st.number_input("Gasto de VRDeck del pasajero", value=0)
    foodcourt_expense = st.number_input("Gasto de FoodCourt del pasajero", value=0)
    spa_expense = st.number_input("Gasto de Spa del pasajero", value=0)
    age = st.number_input("Edad del pasajero", value=0)
    roomservice_expense = st.number_input("Gasto de Room Service del pasajero", value=0)
    shoppingmall_expense = st.number_input("Gasto de Shopping del pasajero", value=0)
    
    clicked = st.button("Predecir")
    
    if clicked:
        st.text("Prediciendo...")
        
        result = model.predict(pd.DataFrame(
        {
            "VIP":[is_vip],
            "VRDeck":[vrdeck_expense],
            "FoodCourt":[foodcourt_expense],
            "Spa":[spa_expense],
            "Age":[age],
            "RoomService":[roomservice_expense],
            "ShoppingMall":[shoppingmall_expense]
        }
        ))
        
        st.text("La predicción dle modelo es: {}".format(result))