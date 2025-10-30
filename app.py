import streamlit as st
import pandas as pd
import plotly.express as px

# Título
st.title("✈️ EDA Dashboard - AIAI Data Mining Project")

# Leitura dos dados
@st.cache_data
def load_data():
    customers = pd.read_csv("DM_AIAI_CustomerDB.csv")
    flights = pd.read_csv("DM_AIAI_FlightsDB.csv")
    return customers, flights

customers, flights = load_data()

st.sidebar.header("Filters")

# Mostrar amostra dos dados
st.subheader("📋 Customer Database")
st.dataframe(customers.head())

st.subheader("🛫 Flights Database")
st.dataframe(flights.head())


st.subheader("🌍 Clients per Province/State")
province_counts = customers['Province or State'].value_counts().reset_index()
province_counts.columns = ['Province or State', 'Count']
fig = px.bar(province_counts, x='Province or State', y='Count', color='Province or State')
st.plotly_chart(fig)