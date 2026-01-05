import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import umap

st.set_page_config(page_title="AIAI Airlines - Final Segmentation", layout="wide")

st.title("AIAI Airlines Final Customer Segmentation Dashboard")

st.markdown("""
This dashboard presents the **final unified customer segmentation** for all customers.

- Interactive 3D visualization using all features (UMAP default)
- Option to manually select 3 features
- Optional view restricted to Value-Based or Behavioral pre-segment
- Real-time attribute filtering and detailed segment profiles
""")

@st.cache_data
def load_data():
    df_unscaled = pd.read_csv("data/all_customers_labels_not_scaled.csv")
    df_scaled = pd.read_csv("data/all_customers_labels.csv")
    
    df = df_unscaled.copy()
    df["value_cluster"] = df_scaled["value_labels"]
    df["behav_cluster"] = df_scaled["behav_labels"]
    df["cluster"] = df_scaled["final_cluster_labels"]
    
    return df

df = load_data()

filtered_df = df.copy()

st.sidebar.header("View Restriction (Optional)")
pre_segment_view = st.sidebar.radio(
    "Show customers from:",
    ["All customers", "Value-Based pre-segment ", "Behavioral pre-segment "]
)

if pre_segment_view == "Value-Based pre-segment ":
    value_selected = st.sidebar.selectbox(
        "Select Value-Based pre-segment",
        sorted(df["value_cluster"].unique())
    )
    filtered_df = filtered_df[filtered_df["value_cluster"] == value_selected]

elif pre_segment_view == "Behavioral pre-segment ":
    behav_selected = st.sidebar.selectbox(
        "Select Behavioral pre-segment",
        sorted(df["behav_cluster"].unique())
    )
    filtered_df = filtered_df[filtered_df["behav_cluster"] == behav_selected]

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

view_mode = st.radio("3D View Mode", ["All Features (UMAP)", "Custom 3 Features"], horizontal=True)

if view_mode == "All Features (UMAP)":
    with st.spinner("Computing UMAP..."):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(filtered_df[numeric_cols])
        reducer = umap.UMAP(n_components=3, random_state=42)
        umap_components = reducer.fit_transform(scaled_data)
    
    plot_data = pd.DataFrame({
        "UMAP1": umap_components[:, 0],
        "UMAP2": umap_components[:, 1],
        "UMAP3": umap_components[:, 2],
        "cluster": filtered_df["cluster"].astype(str)
    })
    
    x, y, z = "UMAP1", "UMAP2", "UMAP3"
    title = "3D View – Final Segmentation"
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
    plot_data["cluster"] = filtered_df["cluster"].astype(str)
    
    title = "3D View – Final Segmentation (Custom Features)"

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

st.subheader("Final Customer Segment Profiles")
for cluster_id in sorted(filtered_df["cluster"].unique()):
    cluster_data = filtered_df[filtered_df["cluster"] == cluster_id]
    n = len(cluster_data)
    with st.expander(f"Customer Segment {int(cluster_id)} – {n:,} customers"):
        st.write("**Key Characteristics (Averages)**")
        avg = cluster_data.drop(columns=["value_cluster", "behav_cluster", "cluster"]).mean(numeric_only=True).round(2)
        st.dataframe(avg.to_frame(name="Average"))
        st.write("**Full Profile**")
        st.dataframe(cluster_data.drop(columns=["value_cluster", "behav_cluster", "cluster"]).describe().round(2))

st.subheader("Export CSV")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=filtered_df.to_csv(index=False).encode(),
    file_name="final_segments_filtered.csv",
    mime="text/csv"
)