import streamlit as st

st.set_page_config(
    page_title="AIAI Airlines - Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛫 AIAI Airlines Interactive Cluster Visualization Dashboard")

st.markdown("""
**Project Deliverable 2 – Customer Segmentation Analysis**

**Features used for the bonus point 5 **
- Interactive 3D cluster visualization 
- Real-time filtering by attributes
- Expandable segment pop-ups with detailed characteristics
- Export functionality for stakeholder sharing

""")

st.success("Select a segmentation page from the left sidebar.")