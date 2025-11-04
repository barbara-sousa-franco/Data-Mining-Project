import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the Dashboard
st.title("AIAI Management Analysis")

# Read the data
customers = pd.read_csv("customers_clean.csv") #new data with new features from customersDB
flights = pd.read_csv("flights_clean.csv") #new data with new features from flightsDB

#to use 2 different data sets in the same dashboard, we are going to divide into two tabs
tab1, tab2 = st.tabs(['Customers Analysis', 'Flights Analysis'])

# Define 'All' in filters to be able to select all the values when filtering
def apply_all(selected, all):
    if 'All' in selected or not selected:
        return all
    else:
        return selected

#Customer Analysis
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
    col1, col2 = st.columns(2)
    # Calculate the CLV average
    with col1:
        mean_clv = filtered_customers['Customer Lifetime Value'].mean()
        st.metric("Average CLV", f"{mean_clv:,.2f}$") # Show result

    with col2:
        # Average time at the programme in months
        mean_prog_time = round(filtered_customers['Months_In_Program'].mean(), 0)
        st.metric("Average Time at the programme", f"{mean_prog_time:.0f} months") # Show result

    #Donut Charts
    col1, col2, col3 = st.columns(3)
    with col1:
        # Gender Donut Chart
        gender_counts = filtered_customers['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']

        fig = px.pie(
            gender_counts,
            names='Gender',
            values='Count',
            color='Gender',
            title='Distribution by Gender',
            hole=0.4  # Donut chart
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)

    with col2:
        # Acitve vs Cancelled Customers Donut Chart
        gender_counts = filtered_customers['ChurnStatus'].value_counts().reset_index()
        gender_counts.columns = ['ChurnStatus', 'Count']

        fig = px.pie(
            gender_counts,
            names='ChurnStatus',
            values='Count',
            color='ChurnStatus',
            title='Active vs Cancelled Customer',
            hole=0.4  # Donut chart
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)

    with col3:
        # Loyalty Status Donut Chart
        gender_counts = filtered_customers['LoyaltyStatus'].value_counts().reset_index()
        gender_counts.columns = ['LoyaltyStatus', 'Count']

        fig = px.pie(
            gender_counts,
            names='LoyaltyStatus',
            values='Count',
            color='LoyaltyStatus',
            title='LoyaltyStatus',
            hole=0.4  # Donut chart
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)

    #Graphs
    st.header("Graphs")

    #Cancellations over time
    customers["CancellationDate_Datetime"] = pd.to_datetime(customers["CancellationDate_Datetime"], errors="coerce")
    cancellations = customers.dropna(subset=["CancellationDate_Datetime"]).copy()
    cancellations["YearMonth"] = cancellations["CancellationDate_Datetime"].dt.to_period("M").astype(str)
    cancellations_over_time = (
        cancellations.groupby("YearMonth")
        .size()
        .reset_index(name="NumCancellations")
    )
    fig = px.line(
        cancellations_over_time,
        x="YearMonth",
        y="NumCancellations",
        markers=True,
        title="Customer Cancellations Over Time",
    )
    fig.update_layout(
        xaxis_title="Year-Month",
        yaxis_title="Number of Cancellations",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    #Cancellations and Enrollments by seasons
    customers["EnrollmentDateOpening_Datetime"] = pd.to_datetime(customers["EnrollmentDateOpening_Datetime"], errors="coerce")
    customers["CancellationDate_Datetime"] = pd.to_datetime(customers["CancellationDate_Datetime"], errors="coerce")
    def get_season(date):
        if pd.isnull(date):
            return None
        month = date.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
    customers["EnrollmentSeason"] = customers["EnrollmentDateOpening_Datetime"].apply(get_season)
    customers["CancellationSeason"] = customers["CancellationDate_Datetime"].apply(get_season)
    enrollments = (
        customers["EnrollmentSeason"]
        .value_counts()
        .rename_axis("Season")
        .reset_index(name="NewCustomers")
    )
    cancellations = (
        customers["CancellationSeason"]
        .value_counts()
        .rename_axis("Season")
        .reset_index(name="CancelledCustomers")
    )
    season_summary = pd.merge(enrollments, cancellations, on="Season", how="outer").fillna(0)
    season_summary_melted = season_summary.melt(
        id_vars="Season",
        value_vars=["NewCustomers", "CancelledCustomers"],
        var_name="CustomerType",
        value_name="Count"
    )
    fig = px.bar(
        season_summary_melted,
        x="Season",
        y="Count",
        color="CustomerType",
        barmode="group",
        title="New vs Cancelled Customers by Season"
    )
    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Number of Customers",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


    col1, col2 = st.columns(2)

    with col1:
        #Number of Clients by Category
        options = [
            "Gender",
            "Education",
            "Marital Status",
            "Income_Category",
            "CLV_Category"
            "LoyaltyStatus",
            "Province or State",
            "City",
            "Location Code",
            "EnrollmentType"
            ]
        
        selected_var = st.selectbox("Choose the variable:", options, key="clients_count")

        clients_count = (filtered_customers[selected_var].value_counts().reset_index())
        clients_count.columns = [selected_var, 'Count']

        # Graph
        fig = px.bar(
            clients_count,
            x=selected_var,
            y='Count',
            color=selected_var,
            text_auto=".2s",
            title=f"Number of Clients by {selected_var}"
        )

        # Layout
        fig.update_layout(
            xaxis_title=selected_var,
            yaxis_title="Number of Clients",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        #Show Result
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # CLV per categories
        options = [
        "Gender",
        "Education",
        "Marital Status",
        "Income_Category",
        "LoyaltyStatus",
        "Province or State",
        "City",
        "Location Code",
        "EnrollmentType"
        ]

        selected_var = st.selectbox("Choose the variable:", options, key="clv_category")

        clv_sum = (
            filtered_customers
            .groupby(selected_var, as_index=False)['Customer Lifetime Value']
            .sum()
            .sort_values(by='Customer Lifetime Value', ascending=False)
        )

        # Graph
        fig = px.bar(
            clv_sum,
            x=selected_var,
            y='Customer Lifetime Value',
            color=selected_var,
            text_auto=".2s",
            title=f"Total Customer Lifetime Value by {selected_var}"
        )
        # Layout
        fig.update_layout(
            xaxis_title=selected_var,
            yaxis_title='Customer Lifetime Value ($)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


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

    col1, col2, col3 = st.columns(3)
    with col1:
        # Rate of points redeemed vs points accumulated
        redeemed_points_rate = filtered_flights['PointsRedeemed'].dropna().sum() / filtered_flights['PointsAccumulated'].dropna().sum()
        st.metric(f"Redeemed Points vs Accumulated Points", f"{redeemed_points_rate:.2%}") # Show result

    with col2:
        #Proportion of flights
        number_of_flights = filtered_flights['NumFlights'].dropna().sum()
        st.metric(f"Number of flights", f"{number_of_flights:,.0f}") #show result

    with col3:
        # Flights with companions (%)
        perc_with_companions = (
            filtered_flights['NumFlightsWithCompanions'].dropna().sum() /
            filtered_flights['NumFlights'].dropna().sum()
            if filtered_flights['NumFlights'].dropna().sum() > 0 else 0
        )
        st.metric("Flights with Companions (%)", f"{perc_with_companions:.2%}")


    # 2 LINE GRAPH - NumFlights and NumFlightswithCompanions
    monthly_trends = (
        flights.groupby("YearMonth", as_index=False)[["NumFlights", "NumFlightsWithCompanions"]]
        .sum()
        .sort_values("YearMonth")
    )
    # Converter YearMonth
    monthly_trends["YearMonth"] = pd.to_datetime(monthly_trends["YearMonth"])
    # Graph
    fig = px.line(
        monthly_trends,
        x="YearMonth",
        y=["NumFlights", "NumFlightsWithCompanions"],
        title="Monthly Trends: Total Flights vs Flights with Companions",
        markers=True
    )
    # Layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Flights",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Flight Type"
    )
    #Show Result
    st.plotly_chart(fig, use_container_width=True)

    # Group Flights by Season
    flights_by_season = (
        flights.groupby("Season", as_index=False)["NumFlights"]
        .sum()
        .sort_values("NumFlights", ascending=False)
    )
    fig = px.bar(
        flights_by_season,
        x="Season",
        y="NumFlights",
        color="Season",
        text_auto=".2s",
        title="Number of Flights by Season"
    )
    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Total Number of Flights",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Type of person with the most flights
    options = [
    "Gender",
    "Education",
    "Marital Status",
    "Income_Category",
    "CLV_Category"
    "LoyaltyStatus",
    "Province or State",
    "City",
    "Location Code",
    "EnrollmentType"
    ]

    selected_var = st.selectbox("Choose the variable:", options, key='flights_total')
    flights_by_type = (
        customers.groupby(selected_var, as_index=False)["TotalFlights"]
        .sum()
        .sort_values("TotalFlights", ascending=False)
    )

    # Graph
    fig = px.bar(
        flights_by_type,
        x=selected_var,
        y="TotalFlights",
        color=selected_var,
        text_auto=".2s",
        title=f"Total Number of Flights by {selected_var}"
    )

    # Layout
    fig.update_layout(
        xaxis_title=selected_var,
        yaxis_title="Total Number of Flights",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


    # Type of person with the most flights with companions
    options = [
    "Gender",
    "Education",
    "Marital Status",
    "Income_Category",
    "CLV_Category"
    "LoyaltyStatus",
    "Province or State",
    "City",
    "Location Code",
    "EnrollmentType"
    ]

    selected_var = st.selectbox("Choose the variable:", options, key='flights_companions')
    flights_by_type = (
        customers.groupby(selected_var, as_index=False)["TotalFlightsWithCompanions"]
        .sum()
        .sort_values("TotalFlightsWithCompanions", ascending=False)
    )

    # Graph
    fig = px.bar(
        flights_by_type,
        x=selected_var,
        y="TotalFlightsWithCompanions",
        color=selected_var,
        text_auto=".2s",
        title=f"Total Number of Flights with Companions by {selected_var}"
    )

    # Layout
    fig.update_layout(
        xaxis_title=selected_var,
        yaxis_title="Total Number of Flights with Companions",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # TOP 10 FLYERS
    st.header("Top 10 Flyers")
    filtered_flights['NumFlights'] = filtered_flights['NumFlights'].fillna(0)
    # Total flights by customer
    top_travelers = (
        filtered_flights.groupby('Loyalty#', as_index=False)['NumFlights']
        .sum()
        .sort_values(by='NumFlights', ascending=False)
        .head(10)
    )
    # Merge Flights with customer
    top_travelers = top_travelers.merge(
        filtered_customers[['Loyalty#', 'Customer Name', 'Country']],
        on='Loyalty#',
        how='left'
    )
    top_travelers = top_travelers[['Loyalty#', 'Customer Name', 'Country', 'NumFlights']]
    st.dataframe(top_travelers, use_container_width=True) # Show result
