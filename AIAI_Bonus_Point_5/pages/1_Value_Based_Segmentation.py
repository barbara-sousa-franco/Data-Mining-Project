# For 1_Value_Based_Segmentation.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    df_unscaled = pd.read_csv("AIAI_Bonus_Point_5/data/all_customers_labels_not_scaled.csv")
    df_scaled = pd.read_csv("AIAI_Bonus_Point_5/data/all_customers_labels.csv")
    
    df = df_unscaled.copy()
    df['cluster'] = df_scaled['value_labels']
    
    return df

st.title("Value-Based Segmentation (All Customers – K-Means, 6 clusters)")

df = load_data()

st.sidebar.header("Real-time Filtering by Attributes")
filtered_df = df.copy()

# filter 
features = [
    col for col in df.columns 
    if col not in ['cluster', 'cluster_labels', 'behav_labels', 'value_labels', 'final_cluster_labels']
]

# dynamic filtering
for feature in features:
    if df[feature].nunique() < 15:
        options = st.sidebar.multiselect(
            feature,
            options=sorted(df[feature].unique()),
            default=sorted(df[feature].unique()),
            key=f"vb_{feature}"
        )
        filtered_df = filtered_df[filtered_df[feature].isin(options)]
    else:
        min_val, max_val = float(df[feature].min()), float(df[feature].max())
        range_val = st.sidebar.slider(
            feature,
            min_val, max_val, (min_val, max_val),
            key=f"vb_{feature}"
        )
        filtered_df = filtered_df[(filtered_df[feature] >= range_val[0]) & (filtered_df[feature] <= range_val[1])]

# numerical columns for 3D plot
numeric_cols = filtered_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if len(numeric_cols) < 3:
    st.error("Not enough numeric features for 3D plot.")
    st.stop()

# Post variate features for default 3D axes
variance = filtered_df[numeric_cols].var().sort_values(ascending=False)
top3 = variance.index[:3].tolist()

col1, col2, col3 = st.columns(3)
with col1:
    x = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index(top3[0]))
with col2:
    y = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(top3[1]))
with col3:
    z = st.selectbox("Z-axis", numeric_cols, index=numeric_cols.index(top3[2]))

# scale for better 3D visualization
scaler = StandardScaler()
plot_data = filtered_df.copy()
plot_data[numeric_cols] = scaler.fit_transform(filtered_df[numeric_cols])

st.subheader("3D Cluster Visualization (rotate, zoom, hover)")
fig = px.scatter_3d(
    plot_data,
    x=x, y=y, z=z,
    color="cluster",
    hover_data=features,  # <- juste les features
    color_discrete_sequence=px.colors.qualitative.Bold,
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

st.subheader("Customer Detail Pop-ups with Segment Characteristics")
for cluster_id in sorted(filtered_df["cluster"].unique()):
    cluster_data = filtered_df[filtered_df["cluster"] == cluster_id]
    n = len(cluster_data)
    with st.expander(f"Segment {int(cluster_id)} – {n:,} customers"):
        st.write("Key Characteristics (Averages)")
        avg = cluster_data.drop(columns="cluster").mean(numeric_only=True).round(2)
        st.dataframe(avg.to_frame(name="Average"))
        st.write("Full Profile")
        st.dataframe(cluster_data.drop(columns="cluster").describe().round(2))

st.subheader("Export for Stakeholder Sharing")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=filtered_df.to_csv(index=False).encode(),
    file_name="value_based_segments_filtered.csv",
    mime="text/csv"
)
