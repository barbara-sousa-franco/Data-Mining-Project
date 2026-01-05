import streamlit as st
import pandas as pd
import plotly.express as px
import umap

st.set_page_config(page_title="AIAI Airlines - Customer Segmentation", layout="wide")

st.title("AIAI Airlines Final Customer Segmentation Dashboard")

st.markdown("""
This dashboard presents the **final unified customer segmentation** for all customers.

- 3D visualization using UMAP on selected feature set
- Real-time filtering and detailed segment profiles
            
For stakeholder sharing, the link to this dashboard can be shared directly.
""")

@st.cache_data
def load_data():
    df_unscaled = pd.read_csv("AIAI_Bonus_Point_5/data/customers_labels_not_scaled.csv")
    df_scaled = pd.read_csv("AIAI_Bonus_Point_5/data/customers_labels.csv")
    
    return df_scaled, df_unscaled

df_scaled, df_unscaled = load_data()

value_segmentation = ['Customer Lifetime Value', 'PropCLV', 'TotalPoints', 'PropPoints', 'Months_Since_Enrollment']
behavioral_segmentation = ['Recency_Months', 'NumFlights_Max', 'NumFlightsWithCompanions_Max', 'Ratio_Flights_Companions', 'Ratio_Points_Redeemed', 'PropNrFlights', 'PropNrFlightsWithCompanions', 'MeanDistancePerFlight', 'DiversitySeason']
clustering_features = value_segmentation + behavioral_segmentation



st.sidebar.header("Real-time Filtering by Attributes")

filtered_df = df_unscaled.copy()

for feature in clustering_features:
    if df_unscaled[feature].nunique() < 15:
        options = st.sidebar.multiselect(feature, sorted(df_unscaled[feature].unique()), default=sorted(df_unscaled[feature].unique()), key=feature)
        filtered_df = filtered_df[filtered_df[feature].isin(options)]
        df_to_plot = df_scaled[df_scaled.index.isin(filtered_df.index)]
    else:
        min_val, max_val = float(df_unscaled[feature].min()), float(df_unscaled[feature].max())
        range_val = st.sidebar.slider(feature, min_val, max_val, (min_val, max_val), key=feature)
        filtered_df = filtered_df[(filtered_df[feature] >= range_val[0]) & (filtered_df[feature] <= range_val[1])]
        df_to_plot = df_scaled[df_scaled.index.isin(filtered_df.index)]


if len(filtered_df) < 3:
    st.error("Not enough numeric features.")
    st.stop()

feature_set = st.radio(
    "Feature Set for Visualization",
    ["All Features", "Value-Based Features Only", "Behavioral Features Only"],
    horizontal=True
)

if feature_set == "Value-Based Features Only":
    umap_cols = value_segmentation
    title = "3D View – Final Segmentation (Value-Based Features via UMAP)"
    use_umap = True
elif feature_set == "Behavioral Features Only":
    umap_cols = behavioral_segmentation
    title = "3D View – Final Segmentation (Behavioral Features via UMAP)"
    use_umap = True
else:
    umap_cols = clustering_features
    use_umap = st.radio("View Mode", ["UMAP (All Features)", "Custom 3 Features"], horizontal=True) == "UMAP (All Features)"
    title = "3D View – Final Segmentation" + (" (All Features via UMAP)" if use_umap else " (Custom Features)")

if use_umap:
    with st.spinner("Computing UMAP..."):
        umap_3d = umap.UMAP(n_components=3, init='random', n_neighbors=50, random_state=1)
        umap_3d_data = umap_3d.fit_transform(df_to_plot[umap_cols])
    
    plot_data = pd.DataFrame({
        'UMAP_1': umap_3d_data[:, 0],
        'UMAP_2': umap_3d_data[:, 1],
        'UMAP_3': umap_3d_data[:, 2],
        'Cluster': df_to_plot['final_cluster_labels'].astype(str)
    })
    
    x, y, z = 'UMAP_1', 'UMAP_2', 'UMAP_3'
else:
    variance = df_to_plot[clustering_features].var().sort_values(ascending=False)
    top3 = variance.index[:3].tolist()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x = st.selectbox("X-axis", clustering_features, index=clustering_features.index(top3[0]))
    with col2:
        y = st.selectbox("Y-axis", clustering_features, index=clustering_features.index(top3[1]))
    with col3:
        z = st.selectbox("Z-axis", clustering_features, index=clustering_features.index(top3[2]))
    
    plot_data = df_to_plot.copy()
    plot_data["cluster"] = plot_data["final_cluster_labels"].astype(str)

st.subheader(title)

custom_colors = [
    "#1f77b4",
    "#e377c2",
    "#ffd700",
    "#ff7f0e",
    "#9467bd"
]

fig = px.scatter_3d(
    plot_data,
    x=x, y=y, z=z,
    color="Cluster" if use_umap else "cluster",  # Fixed: use 'Cluster' for UMAP, 'cluster' for custom
    hover_data=clustering_features if not use_umap else None,
    color_discrete_sequence=custom_colors,
    height=800
)


fig.update_traces(marker=dict(size=4, opacity=0.7, line=dict(width=0)))
fig.update_layout(width=1200, height=900, dragmode='orbit', scene=dict(dragmode='orbit', aspectmode='data'))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Final Customer Segment Profiles (Current View)")
for cluster_id in sorted(df_unscaled["cluster_labels"].unique()):
    cluster_data = df_unscaled[df_unscaled["cluster_labels"] == cluster_id]
    n = len(cluster_data)
    with st.expander(f"Customer Segment {int(cluster_id)} – {n:,} customers"):
        st.write("**Key Characteristics (Averages)**")
        avg = cluster_data.drop(columns=["cluster_labels"]).mean(numeric_only=True).round(2)
        st.dataframe(avg.to_frame(name="Average"))
        st.write("**Full Profile**")
        st.dataframe(cluster_data.drop(columns=["cluster_labels"]).describe().round(2))
