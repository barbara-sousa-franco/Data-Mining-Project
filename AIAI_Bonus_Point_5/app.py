import streamlit as st
import pandas as pd
import plotly.express as px
import umap

st.set_page_config(page_title="AIAI Airlines - Customer Segmentation", layout="wide")

st.title("🛫 AIAI Airlines Final Customer Segmentation Dashboard")

st.markdown("""
This dashboard presents the **final unified customer segmentation** for all customers.

- 3D visualization using UMAP on selected feature set
- Real-time filtering and detailed segment profiles
""")

@st.cache_data
def load_data():
<<<<<<< HEAD
    df_unscaled = pd.read_csv("AIAI_Bonus_Point_5/data/all_customers_labels_not_scaled.csv")
    df_scaled = pd.read_csv("AIAI_Bonus_Point_5/data/all_customers_labels.csv")
=======
    df_unscaled = pd.read_csv("data/customers_labels_not_scaled.csv")
    df_scaled = pd.read_csv("data/customers_labels.csv")
>>>>>>> a4dba4b (dashboard)
    
    df = df_unscaled.copy()
    df["value_cluster"] = df_scaled["value_labels"]
    df["behav_cluster"] = df_scaled["behav_labels"]
    df["cluster"] = df_scaled["final_cluster_labels"]
    
    return df

df = load_data()

filtered_df = df.copy()

st.sidebar.header("Real-time Filtering by Attributes")
features = [col for col in df.columns if col not in ["value_cluster", "behav_cluster", "cluster"]]

for feature in features:
    if df[feature].nunique() < 15:
        options = st.sidebar.multiselect(feature, sorted(df[feature].unique()), default=sorted(df[feature].unique()), key=feature)
        filtered_df = filtered_df[filtered_df[feature].isin(options)]
    else:
        min_val, max_val = float(df[feature].min()), float(df[feature].max())
        range_val = st.sidebar.slider(feature, min_val, max_val, (min_val, max_val), key=feature)
        filtered_df = filtered_df[(filtered_df[feature] >= range_val[0]) & (filtered_df[feature] <= range_val[1])]

numeric_cols = filtered_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if len(numeric_cols) < 3:
    st.error("Not enough numeric features.")
    st.stop()

feature_set = st.radio(
    "Feature Set for Visualization",
    ["All Features", "Value-Based Features Only", "Behavioral Features Only"],
    horizontal=True
)

value_features = ['Customer Lifetime Value', 'Months_Since_Enrollment', 'TotalPoints', 'NumFlights_Max', 'NumFlightsWithCompanions_Max', 'Ratio_Flights_Companions', 'Ratio_Points_Redeemed', 'MeanDistancePerFlight', 'PropNrFlights', 'PropNrFlightsWithCompanions', 'PropCLV', 'PropPoints', 'PropPointsRedem', 'DiversitySeason', 'Recency_Months']
behavioral_features = ['Recency_Months', 'NumFlights_Max', 'NumFlightsWithCompanions_Max', 'Ratio_Flights_Companions', 'Ratio_Points_Redeemed', 'MeanDistancePerFlight', 'PropNrFlights', 'PropNrFlightsWithCompanions', 'DiversitySeason', 'TotalFlights', 'TotalFlightsWithCompanions']

if feature_set == "Value-Based Features Only":
    umap_cols = value_features
    title = "3D View – Final Segmentation (Value-Based Features via UMAP)"
    use_umap = True
elif feature_set == "Behavioral Features Only":
    umap_cols = behavioral_features
    title = "3D View – Final Segmentation (Behavioral Features via UMAP)"
    use_umap = True
else:
    umap_cols = numeric_cols
    use_umap = st.radio("View Mode", ["UMAP (All Features)", "Custom 3 Features"], horizontal=True) == "UMAP (All Features)"
    title = "3D View – Final Segmentation" + (" (All Features via UMAP)" if use_umap else " (Custom Features)")

if use_umap:
    with st.spinner("Computing UMAP..."):
        umap_3d = umap.UMAP(n_components=3, init='random', n_neighbors=50, random_state=1)
        umap_3d_data = umap_3d.fit_transform(filtered_df[umap_cols])
    
    plot_data = pd.DataFrame({
        'UMAP_1': umap_3d_data[:, 0],
        'UMAP_2': umap_3d_data[:, 1],
        'UMAP_3': umap_3d_data[:, 2],
        'Cluster': filtered_df['cluster'].astype(str)
    })
    
    x, y, z = 'UMAP_1', 'UMAP_2', 'UMAP_3'
else:
    variance = filtered_df[numeric_cols].var().sort_values(ascending=False)
    top3 = variance.index[:3].tolist()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index(top3[0]))
    with col2:
        y = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(top3[1]))
    with col3:
        z = st.selectbox("Z-axis", numeric_cols, index=numeric_cols.index(top3[2]))
    
    plot_data = filtered_df.copy()
    plot_data["cluster"] = filtered_df["cluster"].astype(str)

st.subheader(title)

custom_colors = [
    "#1f77b4",
    "#9467bd",
    "#e377c2",
    "#ff7f0e",
    "#ffd700"
]

fig = px.scatter_3d(
    plot_data,
    x=x, y=y, z=z,
    color="Cluster" if use_umap else "cluster",  # Fixed: use 'Cluster' for UMAP, 'cluster' for custom
    hover_data=features if not use_umap else None,
    color_discrete_sequence=custom_colors,
    height=800
)
fig.update_traces(marker=dict(size=4, opacity=0.85))
fig.update_layout(
    scene=dict(
        xaxis_title=x,
        yaxis_title=y,
        zaxis_title=z,
        aspectmode="data"
    )
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Final Customer Segment Profiles (Current View)")
for cluster_id in sorted(filtered_df["cluster"].unique()):
    cluster_data = filtered_df[filtered_df["cluster"] == cluster_id]
    n = len(cluster_data)
    with st.expander(f"Customer Segment {int(cluster_id)} – {n:,} customers"):
        st.write("**Key Characteristics (Averages)**")
        avg = cluster_data.drop(columns=["value_cluster", "behav_cluster", "cluster"]).mean(numeric_only=True).round(2)
        st.dataframe(avg.to_frame(name="Average"))
        st.write("**Full Profile**")
        st.dataframe(cluster_data.drop(columns=["value_cluster", "behav_cluster", "cluster"]).describe().round(2))

st.subheader("Export for Stakeholder Sharing")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=filtered_df.to_csv(index=False).encode(),
    file_name="final_segments_filtered.csv",
    mime="text/csv"
)