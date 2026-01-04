import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AIAI Airlines - Customer Segmentation", layout="wide")

st.title("🛫 AIAI Airlines Customer Segmentation Dashboard")

st.markdown("""
This dashboard shows the final customer segmentation using all features.

- **Default 3D view**: All features reduced to 3 dimensions (PCA) for global cluster view
- **Custom view**: Choose any 3 features manually
- Switch between Final, Value-Based, or Behavioral segmentation
- Real-time filtering and segment profiles
""")

@st.cache_data
def load_data():
    df_unscaled = pd.read_csv("data/all_customers_labels_not_scaled.csv")
    df_scaled = pd.read_csv("data/all_customers_labels.csv")
    
    df = df_unscaled.copy()
    df["value_cluster"] = df_scaled["value_labels"]
    df["behav_cluster"] = df_scaled["behav_labels"]
    df["final_cluster"] = df_scaled["final_cluster_labels"]
    
    return df

df = load_data()

# View selector
view = st.radio(
    "Select segmentation view:",
    ["Final Unified Segmentation", "Value-Based Segmentation", "Behavioral Segmentation"],
    horizontal=True
)

cluster_col = {
    "Final Unified Segmentation": "final_cluster",
    "Value-Based Segmentation": "value_cluster",
    "Behavioral Segmentation": "behav_cluster"
}[view]

filtered_df = df.copy()

# Filtering
st.sidebar.header("Real-time Filtering by Attributes")
features = [col for col in df.columns if col not in ["value_cluster", "behav_cluster", "final_cluster"]]

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

# 3D view mode
view_mode = st.radio("3D View Mode", ["All Features (PCA)", "Custom 3 Features"], horizontal=True)

if view_mode == "All Features (PCA)":
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_df[numeric_cols])
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(scaled_data)
    
    plot_data = pd.DataFrame({
        "PC1": pca_components[:, 0],
        "PC2": pca_components[:, 1],
        "PC3": pca_components[:, 2],
        "cluster": filtered_df[cluster_col].astype(str)
    })
    
    x, y, z = "PC1", "PC2", "PC3"
    title = f"3D View – {view} (All Features via PCA)"
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
    
    scaler = StandardScaler()
    plot_data = filtered_df.copy()
    plot_data[numeric_cols] = scaler.fit_transform(filtered_df[numeric_cols])
    plot_data["cluster"] = filtered_df[cluster_col].astype(str)
    
    title = f"3D View – {view} (Custom Features)"

st.subheader(title)

# Exact colors from your notebook t-SNE plot
custom_colors = [
    "#1f77b4",  # dark blue (Cluster 0)
    "#9467bd",  # purple (Cluster 1)
    "#e377c2",  # red/pink (Cluster 2)
    "#ff7f0e",  # orange (Cluster 3)
    "#ffd700"   # yellow (Cluster 4)
]

fig = px.scatter_3d(
    plot_data,
    x=x, y=y, z=z,
    color="cluster",
    hover_data=features if view_mode == "Custom 3 Features" else None,
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

st.subheader("Customer Segment Profiles")
for cluster_id in sorted(filtered_df[cluster_col].unique()):
    cluster_data = filtered_df[filtered_df[cluster_col] == cluster_id]
    n = len(cluster_data)
    with st.expander(f"Customer Segment {int(cluster_id)} – {n:,} customers"):
        st.write("Key Characteristics (Averages)")
        avg = cluster_data.drop(columns=[cluster_col, "value_cluster", "behav_cluster", "final_cluster"]).mean(numeric_only=True).round(2)
        st.dataframe(avg.to_frame(name="Average"))
        st.write("Full Profile")
        st.dataframe(cluster_data.drop(columns=[cluster_col, "value_cluster", "behav_cluster", "final_cluster"]).describe().round(2))

st.subheader("Export for Stakeholder Sharing")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=filtered_df.to_csv(index=False).encode(),
    file_name=f"{view.lower().replace(' ', '_')}_filtered.csv",
    mime="text/csv"
)