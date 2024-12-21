import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set Streamlit page config
st.set_page_config(page_title="Clustering Dashboard", layout="wide")

# App title
st.title("Clustering Dashboard")

# File uploader for data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Select numeric columns
    numeric_columns = data.select_dtypes(include=[np.number])
    st.write("### Numeric Features Used for Clustering")
    st.dataframe(numeric_columns.describe())

    # Data preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_columns)

    # Sidebar for K-Means configuration
    st.sidebar.title("K-Means Configuration")
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3, step=1)
    random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=42, step=1)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    data['Cluster'] = kmeans_labels

    # Cluster Distribution
    st.write("### Cluster Distribution")
    cluster_counts = data['Cluster'].value_counts()
    st.bar_chart(cluster_counts)

    # Visualizations
    st.write("### Clustering Visualizations")

    # Cluster scatterplot
    st.write("#### 2D Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=scaled_data[:, 0], 
        y=scaled_data[:, 1], 
        hue=kmeans_labels, 
        palette="viridis", 
        s=100, 
        ax=ax
    )
    ax.set_title("Clusters Visualization (First Two Features)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)

    # Pairplot
    st.write("#### Pairplot of Numeric Features by Cluster")
    pairplot_data = pd.DataFrame(scaled_data, columns=numeric_columns.columns)
    pairplot_data['Cluster'] = kmeans_labels
    pairplot_fig = sns.pairplot(pairplot_data, hue="Cluster", palette="viridis")
    st.pyplot(pairplot_fig)

    # Heatmap of cluster centers
    st.write("#### Heatmap of Cluster Centers")
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_columns.columns)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cluster_centers.T, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Cluster Centers Heatmap")
    st.pyplot(fig)

    # Download clustered data
    st.write("### Download Clustered Data")
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Clustered Data as CSV", 
        data=csv, 
        file_name="clustered_data.csv", 
        mime="text/csv"
    )
else:
    st.write("Upload a CSV file to visualize clustering results.")
