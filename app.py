#Import modules

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import joblib

pd.options.display.max_colwidth = 2000
st.set_page_config(
    page_title="Crop Recommendation",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-color:#FCFCFC;

}}
[data-testid="stSidebar"] {{
background-color:#131c46;s

}}
[data-testid="stHeader"] {{
background-color:#FCFCFC;
}}
[data-testid="stToolbar"] {{
background-color:#FCFCFC;

}}
</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)

def load_bootstrap():
        return st.markdown("""<link rel="stylesheet" 
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
        crossorigin="anonymous">""", unsafe_allow_html=True)

with st.sidebar:
    
    load_bootstrap()

    image = Image.open('crop_image.jpg')

    st.image(image, width=300)
    st.markdown("<h1 style='text-align: center;'>Crop Recommendation </h1>", unsafe_allow_html= True)
    st.markdown("""
        <h4 style='text-align: center;'>
        Based on weather and field conditions
        
        </h4>
        """, unsafe_allow_html=True)

    df_desc = pd.read_csv('Dataset/Crop_Desc.csv', sep = ';', encoding = 'utf-8')

    df = pd.read_csv('Dataset/Crop_recommendation.csv')

    rdf_clf = joblib.load('Model/RDF_model.pkl')

    X = df.drop('label', axis = 1)
    y = df['label']

st.markdown("<h3 style='text-align: center;'>Introduce Field and environment conditions</h3><br>", unsafe_allow_html=True)


col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,4,1,4,1,1], gap = 'medium')

with col3:
    n_input = st.number_input('Nitrogen Density (0-140 kg N/ha):', min_value= 0, max_value= 140)
    p_input = st.number_input('Phosphorus Density (5-145 kg P/ha):', min_value= 5, max_value= 145)
    k_input = st.number_input('Potassium Density (5-205 kg K/ha)', min_value= 5, max_value= 205)
    temp_input = st.number_input('Average Temperature (9-43 °C)', min_value= 9., max_value= 43., step = 1., format="%.2f")

with col5:
    hum_input = st.number_input('Average Humidity (15-99 %):', min_value= 15., max_value= 99., step = 1., format="%.2f")
    ph_input = st.number_input('pH (3.6-9.9):', min_value= 3.6, max_value= 9.9, step = 0.1, format="%.2f")
    rain_input = st.number_input('Average Rainfall (21-298 mm):', min_value= 21.0, max_value= 298.0, step = 0.1, format="%.2f")
    location = st.selectbox('Region:',('Central', 'Eastern', 'North Eastern', 'Northern', 'Western', 'Other'))

    if location == 'Central':
        predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input,1,0,0,0,0,0]]
    elif location == 'Eastern':
        predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input,0,1,0,0,0,0]]
    elif location == 'North Eastern':
        predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input,0,0,1,0,0,0]]
    elif location == 'Northern':
        predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input,0,0,0,1,0,0]]
    elif location == 'Other':
        predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input,0,1,0,0,1,0]]
    elif location == 'Western':
        predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input,0,1,0,0,0,1]]


with col5:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button('Recommendation')
    st.markdown("<br>", unsafe_allow_html=True)


cola,colb,colc = st.columns([2,10,2])
if predict_btn:
    rdf_predicted_value = rdf_clf.predict(predict_inputs)

    st.markdown(f"<br><h5 style='text-align: center;'>Recommendation <b>{rdf_predicted_value[0]}</b></h5>", unsafe_allow_html=True)

    st.markdown(f"""<h5 style='text-align: center;'>Ideal conditions for <b>{rdf_predicted_value[0]}</b></h5>""", unsafe_allow_html=True)
    df_pred = df[df['label'] == rdf_predicted_value[0]]
    
    
    # Calcular media y desvío estándar
    mean = df.mean()
    

    # Graficar
    fig, ax = plt.subplots()
    mean.plot(kind='barh', ax=ax, capsize=5)
    plt.xlabel('Value')
    plt.ylabel('Parameter')
    plt.xticks(rotation=0)
    st.pyplot(fig)

    
    st.markdown("---")

    
    

    
    

    