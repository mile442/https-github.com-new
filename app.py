import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Customer Segmentation",
    layout="wide"
)

st.title("AI Customer Segmentation using AHC")

st.write("Hierarchical Clustering Dashboard")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

data_option = st.sidebar.radio(
    "Choose data source",
    ["Upload Excel", "Use local file"]
)

if data_option == "Upload Excel":

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

elif data_option == "Use local file":

    file_path = "data/customers.xlsx"

    try:
        df = pd.read_excel(file_path)
        st.sidebar.success("Loaded data/customers.xlsx")

    except:
        st.sidebar.error("File not found")

# ------------------------------------------------
# DISPLAY DATA
# ------------------------------------------------

if 'df' in locals():

    st.subheader("Dataset")

    st.dataframe(df)

    # ------------------------------------------------
    # DATA PREPROCESSING
    # ------------------------------------------------

    st.subheader("Data Preprocessing")

    df_features = df.drop(columns=["ID"])

    df_features = df_features.replace({
        "Yes":1,
        "No":0
    })

    numeric_df = df_features.select_dtypes(include=np.number)

    st.write("Features used for clustering")

    st.dataframe(numeric_df)

    # ------------------------------------------------
    # STANDARDIZE
    # ------------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(numeric_df)

    # ------------------------------------------------
    # HIERARCHICAL CLUSTERING
    # ------------------------------------------------

    st.subheader("AHC Dendrogram")

    Z = linkage(X, method="ward")

    fig, ax = plt.subplots(figsize=(12,6))

    dendrogram(
        Z,
        labels=df["ID"].values,
        leaf_rotation=90
    )

    st.pyplot(fig)

    # ------------------------------------------------
    # CLUSTER SELECTION
    # ------------------------------------------------

    st.subheader("Select number of clusters")

    k = st.slider("Clusters",2,10,3)

    clusters = fcluster(Z, k, criterion="maxclust")

    df["Cluster"] = clusters

    # ------------------------------------------------
    # CLUSTER TABLE
    # ------------------------------------------------

    st.subheader("Cluster Result")

    st.dataframe(df)

    # ------------------------------------------------
    # CLUSTER SIZE
    # ------------------------------------------------

    st.subheader("Cluster Size")

    cluster_counts = df["Cluster"].value_counts()

    fig2 = px.bar(
        cluster_counts,
        title="Customers per Cluster"
    )

    st.plotly_chart(fig2,use_container_width=True)

    # ------------------------------------------------
    # CLUSTER STATS
    # ------------------------------------------------

    st.subheader("Cluster Statistics")

    cluster_stats = df.groupby("Cluster")[numeric_df.columns].mean()

    st.dataframe(cluster_stats)

    # ------------------------------------------------
    # CORRELATION HEATMAP
    # ------------------------------------------------

    st.subheader("Feature Correlation")

    corr = numeric_df.corr()

    fig3, ax3 = plt.subplots(figsize=(8,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax3
    )

    st.pyplot(fig3)

    # ------------------------------------------------
    # PCA VISUALIZATION
    # ------------------------------------------------

    st.subheader("PCA Cluster Visualization")

    pca = PCA(n_components=2)

    components = pca.fit_transform(X)

    pca_df = pd.DataFrame()

    pca_df["PC1"] = components[:,0]
    pca_df["PC2"] = components[:,1]
    pca_df["Cluster"] = clusters
    pca_df["ID"] = df["ID"]

    fig4 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["ID"],
        title="Customer Segmentation Map"
    )

    st.plotly_chart(fig4,use_container_width=True)

    # ------------------------------------------------
    # SCATTER PLOT
    # ------------------------------------------------

    st.subheader("Cluster Scatter Plot")

    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox("X axis",numeric_df.columns)

    with col2:
        y_feature = st.selectbox("Y axis",numeric_df.columns,index=1)

    fig5 = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="Cluster",
        hover_data=["ID"],
        title="Cluster Scatter"
    )

    st.plotly_chart(fig5,use_container_width=True)

    # ------------------------------------------------
    # CLUSTER PROFILE
    # ------------------------------------------------

    st.subheader("Cluster Profiles")

    overall = numeric_df.mean()

    for c in sorted(df["Cluster"].unique()):

        st.markdown(f"### Cluster {c}")

        sub = df[df["Cluster"]==c]

        st.write("Customers:",len(sub))

        stats = sub[numeric_df.columns].mean()

        st.write(stats)

    # ------------------------------------------------
    # AI SEGMENTATION REPORT
    # ------------------------------------------------

    st.subheader("AI Customer Segmentation Report")

    report_text = ""

    for c in sorted(df["Cluster"].unique()):

        sub = df[df["Cluster"]==c]

        stats = sub[numeric_df.columns].mean()

        report_text += f"\nCluster {c}\n"

        if stats["Brand loyalty"] > overall["Brand loyalty"]:
            report_text += "High brand loyalty\n"

        if stats["Price sensitivity"] > overall["Price sensitivity"]:
            report_text += "Highly price sensitive\n"

        if stats["Online buyer"] > overall["Online buyer"]:
            report_text += "Active online buyers\n"

        if stats["Age"] < overall["Age"]:
            report_text += "Younger customer segment\n"

        if stats["Crunchy"] > overall["Crunchy"]:
            report_text += "Prefer crunchy products\n"

        if stats["Bitter"] > overall["Bitter"]:
            report_text += "Like bitter taste\n"

        if stats["Frozen"] > overall["Frozen"]:
            report_text += "Prefer frozen food\n"

        report_text += "\n"

    st.text(report_text)

    # ------------------------------------------------
    # MARKETING STRATEGY
    # ------------------------------------------------

    st.subheader("Marketing Strategy Suggestions")

    for c in sorted(df["Cluster"].unique()):

        sub = df[df["Cluster"]==c]

        stats = sub[numeric_df.columns].mean()

        st.markdown(f"### Cluster {c}")

        if stats["Brand loyalty"] > overall["Brand loyalty"]:
            st.write("Focus on loyalty programs")

        if stats["Price sensitivity"] > overall["Price sensitivity"]:
            st.write("Offer promotions and discounts")

        if stats["Online buyer"] > overall["Online buyer"]:
            st.write("Invest in online marketing")

        if stats["Age"] < overall["Age"]:
            st.write("Target social media marketing")