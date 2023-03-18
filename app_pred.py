import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# llamado al modelo
filename = 'LogisticRegressionClModel.sav'
model = pickle.load(open(filename, 'rb'))

# Función para predecir la supervivencia en base a nuestro modelito
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = model.predict(features)
    return prediction[0]

# Conversión de variables categóricas usando label encoder ( visto en clase hace un momento )
def encode_sex(sex):
    le = LabelEncoder()
    le.fit(['male', 'female'])
    return le.transform([sex])[0]

def encode_embarked(embarked):
    le = LabelEncoder()
    le.fit(['C', 'Q', 'S'])
    return le.transform([embarked])[0]

# Interfaz de nuestra app
st.title('Predicción de supervivencia en el Titanic')

Pclass = st.selectbox('Clase de Pasajero (Pclass)', options=[1, 2, 3])
Sex = st.selectbox('Sexo', options=['male', 'female'])
Age = st.number_input('Edad', min_value=0, max_value=100, value=25, step=1)
SibSp = st.number_input('Número de hermanos/cónyuges a bordo (SibSp)', min_value=0, max_value=10, value=0, step=1)
Parch = st.number_input('Número de padres/hijos a bordo (Parch)', min_value=0, max_value=10, value=0, step=1)
Fare = st.number_input('Tarifa pagada (Fare)', min_value=0.0, max_value=600.0, value=10.0, step=0.1)
Embarked = st.selectbox('Puerto de Embarque', options=['C', 'Q', 'S'])

Sex_encoded = encode_sex(Sex)
Embarked_encoded = encode_embarked(Embarked)

if st.button('Predecir'):
    prediction = predict_survival(Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded)
    if prediction == 1:
        st.success('El pasajero es probable que sobreviva.')
    else:
        st.error('El pasajero es probable que no sobreviva.')