import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the Dashboard
st.title("AIAI Management Analysis")

# Read the data
customers = pd.read_csv("customers_clean.csv")
flights = pd.read_csv("DM_AIAI_FlightsDB.csv")

#to use 2 different data sets in the same dashboard, we are going to divide into two tabs
tab1, tab2 = st.tabs(['customers', 'flights'])

# Define 'All' in filters to be able to select all the values when filtering
def apply_all(selected, all):
    if 'All' in selected or not selected:
        return all
    else:
        return selected

#Customer Analysis (merged)
with tab1:

    st.sidebar.header("Filters Customers")
    st.header('Customers Analysis')

    # Create filters
    states = sorted(customers['Province or State'].dropna().unique())
    cities = sorted(customers['City'].dropna().unique())

    states_options = ['All'] + states
    cities_options = ['All'] + cities

    selected_state = st.sidebar.multiselect('Choose the Province/State:', options=states_options, default=['All'])
    selected_city = st.sidebar.multiselect('Choose the City:', options=cities_options, default=['All'])


    # Call function apply_all()
    state_filter = apply_all(selected_state, states)
    city_filter = apply_all(selected_city, cities)


    filtered_customers = customers[
        (customers['Province or State'].isin(state_filter)) &
        (customers['City'].isin(city_filter))]

    # Boxs
    # Calculate the CLV average
    mean_clv = filtered_customers['Customer Lifetime Value'].mean()
    st.metric("Average CLV", f"{mean_clv:,.2f}$") # Show result

    # Average time at the programme in months
    mean_prog_time = round(filtered_customers['Months_In_Program'].mean(), 0)
    st.metric("Average Time at the programme", f"{mean_prog_time:.1f} months") # Show result

    # Active customers
    active_customers = (filtered_customers['ChurnStatus'] == 'Active').dropna().mean() 
    st.metric("Active Customers", f"{active_customers:.2%}")


    #Graphs
    #Clients per Marital Status (fará sentido ver os clientes ativos se calhar)
    st.subheader("Clients per Marital Status")
    state_counts = filtered_customers['Marital Status'].value_counts().reset_index()
    state_counts.columns = ['Marital Status', 'Count']
    fig = px.bar(
        state_counts,
        x='Marital Status',
        y='Count',
        color='Marital Status'
    )
    fig.update_layout(
        xaxis_title='Marital Status',
        yaxis_title='Number of Clients',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    st.plotly_chart(fig)

    #CLV education
    st.subheader('Customer Lifetime Value customers per Education level')
    clv_sum_by_education=(
        filtered_customers
        .groupby('Education', as_index=False)['Customer Lifetime Value']
        .sum()
        .sort_values(by='Customer Lifetime Value', ascending=False)
    )
    fig=px.bar(
        clv_sum_by_education,
        x='Education',
        y='Customer Lifetime Value',
        color='Education',
        title='CLV per Education Level'
    )
    fig.update_layout(
        xaxis_title='Education Level',
        yaxis_title='Customer Lifetime Value ($)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
    
    #number of active customers with location code (falta fazer os clientes ativos)
    st.subheader("Location Code")
    state_counts = filtered_customers['Location Code'].value_counts().reset_index()
    state_counts.columns = ['Location Code', 'Count']
    fig = px.bar(
        state_counts,
        x='Location Code',
        y='Count',
        color='Location Code'
    )
    fig.update_layout(
        xaxis_title='Location Code',
        yaxis_title='Location Code',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    st.plotly_chart(fig)



    # Show sample of the data
    st.subheader("📋 Customer Database")
    st.dataframe(filtered_customers.head())

#Flights Activity
with tab2:

    st.sidebar.header("Filters Flights")
    st.header('Flights Activity')

    years = sorted(flights['Year'].dropna().unique())
    months = sorted(flights['Month'].dropna().unique())

    years_options = ['All'] + years
    months_options = ['All'] + months

    selected_year = st.sidebar.multiselect('Choose the Year:', options=years_options, default=['All'])
    selected_month = st.sidebar.multiselect('Choose the Month:', options=months_options, default=['All'])

    # Call function apply_all()
    year_filter = apply_all(selected_year, years)
    month_filter = apply_all(selected_month, months)

    filtered_flights = flights[
        (flights['Year'].isin(year_filter)) &
        (flights['Month'].isin(month_filter))]

    # Rate of points redeemed vs points accumulated
    redeemed_points_rate = filtered_flights['PointsRedeemed'].dropna().sum() / filtered_flights['PointsAccumulated'].dropna().sum()
    st.metric(f"Rate of Redeemed Points vs Accumulated Points", f"{redeemed_points_rate:.2%}") # Show result

    #Proportion of flights
    number_of_flights = filtered_flights['NumFlights'].dropna().sum()
    st.metric(f"Number of flights", f"{number_of_flights:.1f}") #show result




    # Show sample of the data
    st.subheader("📋 Flights Database")
    st.dataframe(filtered_flights.head())


