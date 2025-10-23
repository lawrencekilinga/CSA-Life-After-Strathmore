import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Data Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- LOAD DATA ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.title("📊 Interactive Data Analytics Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"✅ Dataset loaded successfully — {df.shape[0]} rows × {df.shape[1]} columns")

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filters")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # --- Quick Overview ---
    with st.expander("📈 Dataset Preview"):
        st.dataframe(df.head())

    # --- METRICS OVERVIEW ---
    st.subheader("📌 Key Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # --- CHARTS SECTION ---
    st.subheader("📊 Data Visualizations")

    # Categorical Plot
    if categorical_cols:
        cat_col = st.selectbox("Select Categorical Column", categorical_cols)
        cat_fig = px.histogram(df, x=cat_col, color_discrete_sequence=['#0072B2'])
        st.plotly_chart(cat_fig, use_container_width=True)

    # Numerical Distribution
    if numeric_cols:
        num_col = st.selectbox("Select Numerical Column", numeric_cols)
        num_fig = px.histogram(df, x=num_col, nbins=30, color_discrete_sequence=['#009E73'])
        st.plotly_chart(num_fig, use_container_width=True)

    # --- SCATTER / RELATIONSHIP ---
    if len(numeric_cols) >= 2:
        st.subheader("📉 Variable Relationships")
        x_axis = st.selectbox("X-axis", numeric_cols, index=0)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
        color_by = st.selectbox("Color by (optional)", [None] + categorical_cols)
        scatter_fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by)
        st.plotly_chart(scatter_fig, use_container_width=True)

    # --- FILTERING AND INSIGHTS ---
    st.subheader("🔎 Filtered View")
    if categorical_cols:
        filter_col = st.selectbox("Filter by column", categorical_cols)
        filter_value = st.multiselect("Select values", df[filter_col].unique())
        if filter_value:
            filtered_df = df[df[filter_col].isin(filter_value)]
            st.dataframe(filtered_df)
        else:
            st.dataframe(df)

    # --- CORRELATION HEATMAP ---
    st.subheader("🔥 Correlation Heatmap")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

else:
    st.info("📥 Please upload a CSV file to start the analysis.")