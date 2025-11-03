import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the Dashboard
st.title("AIAI Management Analysis")

# Read the data
customers = pd.read_csv("customers_clean.csv")
flights = pd.read_csv("flights_clean.csv")

#to use 2 different data sets in the same dashboard, we are going to divide into two tabs
tab1, tab2 = st.tabs(['Customers', 'Flights'])

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
    active = customers['ChurnStatus'].dropna().astype(str).str.strip().unique().tolist()

    states_options = ['All'] + states
    cities_options = ['All'] + cities
    active_options = ['All'] + active


    selected_state = st.sidebar.multiselect('Choose the Province/State:', options=states_options, default=['All'])
    selected_city = st.sidebar.multiselect('Choose the City:', options=cities_options, default=['All'])
    selected_active = st.sidebar.multiselect('Choose Active or Cancelled costumer:', options=active_options, default=['All'])

    # Call function apply_all()
    state_filter = apply_all(selected_state, states)
    city_filter = apply_all(selected_city, cities)
    active_filter = apply_all(selected_active, active)


    filtered_customers = customers[
        (customers['Province or State'].isin(state_filter)) &
        (customers['City'].isin(city_filter)) &
        customers['ChurnStatus'].isin(active_filter)]
    
    # KPI's
    st.header("KPI's")
    col1, col2, col3 = st.columns(3)
    # Calculate the CLV average
    with col1:
        mean_clv = filtered_customers['Customer Lifetime Value'].mean()
        st.metric("Average CLV", f"{mean_clv:,.2f}$") # Show result

    with col2:
        # Average time at the programme in months
        mean_prog_time = round(filtered_customers['Months_In_Program'].mean(), 0)
        st.metric("Average Time at the programme", f"{mean_prog_time:.0f} months") # Show result

    with col3:
        # Active customers
        active_customers_mean = (filtered_customers['ChurnStatus'] == 'Active').dropna().mean()
        st.metric("Active Customers", f"{active_customers_mean:.2%}") # Show result

    #Graphs
    st.header("Graphs")
    col1, col2 = st.columns(2)
    with col1:
        # Number of Clients per Marital Status
        marital_counts = filtered_customers['Marital Status'].value_counts().reset_index()
        marital_counts.columns = ['Marital Status', 'Count']
        fig = px.bar(
            marital_counts,
            x='Marital Status',
            y='Count',
            color='Marital Status',
            title='Number of Clients per Marital Status'
        )
        fig.update_layout(
            xaxis_title='Marital Status',
            yaxis_title='Number of Clients',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig)

        #CLV by Most Frequent Season
        clv_by_season = (
            filtered_customers
            .groupby('MostFrequentSeason', as_index=False)['Customer Lifetime Value']
            .sum()
            .sort_values(by='Customer Lifetime Value', ascending=False)
        )

        fig = px.bar(
            clv_by_season,
            x='MostFrequentSeason',
            y='Customer Lifetime Value',
            color='MostFrequentSeason',
            title='Total Customer Lifetime Value per most frequent Season'
        )
        fig.update_layout(
            xaxis_title='Most Frequent Season',
            yaxis_title='Total Customer Lifetime Value ($)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig)


    with col2:
        # CLV Education
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

        st.plotly_chart(fig)


        # Flights with Companions by Marital Status
        filtered_customers['TotalFlightsWithCompanions'] = filtered_customers['TotalFlightsWithCompanions'].fillna(0)
        flights_with_companions = (
            filtered_customers
            .groupby('Marital Status', as_index=False)['TotalFlightsWithCompanions']
            .sum()
            .sort_values(by='TotalFlightsWithCompanions', ascending=False)
        )

        fig = px.bar(
            flights_with_companions,
            x='Marital Status',
            y='TotalFlightsWithCompanions',
            color='Marital Status',
            title='Flights with Companions per Marital Status'
        )

        fig.update_layout(
            xaxis_title='Marital Status',
            yaxis_title='Total Flights WithCompanions',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )

        st.plotly_chart(fig)
        # Show sample of the data
        st.subheader("Customer Raw Data")
        st.dataframe(filtered_customers.head())

#Flights Activity
with tab2:

    st.sidebar.header("Filters Flights")
    st.header('Flights Activity')

    # Filters
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

    # KPI's
    st.header("KPI's")
    # Rate of points redeemed vs points accumulated
    redeemed_points_rate = filtered_flights['PointsRedeemed'].dropna().sum() / filtered_flights['PointsAccumulated'].dropna().sum()
    st.metric(f"Rate of Redeemed Points vs Accumulated Points", f"{redeemed_points_rate:.2%}") # Show result

    #Proportion of flights
    number_of_flights = filtered_flights['NumFlights'].dropna().sum()
    st.metric(f"Number of flights", f"{number_of_flights:,.0f}") #show result


    st.header("Top 10 Flyers")
    # Garantir que a coluna NumFlights não tem valores nulos
    filtered_flights['NumFlights'] = filtered_flights['NumFlights'].fillna(0)

    # Somar o número total de voos por cliente (Loyalty#)
    top_travelers = (
        filtered_flights.groupby('Loyalty#', as_index=False)['NumFlights']
        .sum()
        .sort_values(by='NumFlights', ascending=False)
        .head(10)
    )

    # Juntar com o nome do cliente (opcional)
    top_travelers = top_travelers.merge(
        filtered_customers[['Loyalty#', 'Customer Name', 'Country']],
        on='Loyalty#',
        how='left'
    )

    # Reordenar colunas para melhor visualização
    top_travelers = top_travelers[['Loyalty#', 'Customer Name', 'Country', 'NumFlights']]

    # Mostrar tabela no Streamlit
    st.dataframe(top_travelers, use_container_width=True)


    # Show sample of the data
    st.subheader("Flights Raw Data")
    st.dataframe(filtered_flights.head())


