from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram  # explicit import

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import plotting

def compute_sse(data, k_range):
    sse = []

    for k in range(1, k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(12, 8))
    plt.xticks(range(1, k_range))
    plt.plot(range(1, k_range), sse, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("SSE (Inertia)")
    plt.title("Elbow Method")
    plt.show()



def pipeline(data, scaler, pca):
    # Pipeline for the preprocessing of the dataset
    if scaler==None and pca==None:
        return data
    elif scaler==None and pca!=None:
        pca.fit(data)
        variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance)
        threshold_variance = 0.85
        n_components = np.argmax(cumulative_variance >= threshold_variance) + 1
        pca_plot = PCA(n_components=n_components)
        data_pca = pca_plot.fit_transform(data)
        return data_pca
    elif scaler!=None and pca!=None:
        data_scaled = scaler.fit_transform(data)
        pca.fit(data_scaled)
        variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance)
        threshold_variance = 0.85
        n_components = np.argmax(cumulative_variance >= threshold_variance) + 1
        pca_plot = PCA(n_components=n_components)
        data_pca = pca_plot.fit_transform(data_scaled)
        return data_pca
    elif scaler!=None and pca==None:
        data_scaled = scaler.fit_transform(data)
        return data_scaled

    return data


def kmeans(data, kmeans_params, agglo_params, scaler=None, pca=None, pipeline_id=0):
    # Function that does kmeans clustering using different pipelines
    ids = list(data.keys())
    embeddings = np.array(list(data.values()))
    if pipeline_id == 0:
        # Data preprocessing
        data_preprocessed = pipeline(embeddings, scaler, pca)
        if kmeans_params["n_clusters"] is None:
            # If n_clusters not defined asks user for an input
            # Computes and plots the sse score so the user can decide how many cluster they should use
            compute_sse(data_preprocessed, 11)
            n = int(input("How many clusters: "))
            kmeans_params["n_clusters"] = n
        # Running Kmeans clustering algorithm
        clustering = KMeans(n_clusters=kmeans_params["n_clusters"], n_init=kmeans_params["n_init"],
                            random_state=kmeans_params["random_state"], init=kmeans_params["init"],
                            max_iter=kmeans_params["max_iter"], tol=kmeans_params["tol"],
                            verbose=kmeans_params["verbose"], copy_x=kmeans_params["copy_x"],
                            algorithm=kmeans_params["algorithm"])
        clustering_labels = clustering.fit_predict(data_preprocessed)
        unique_clusters = np.unique(clustering_labels)
        # Plotting data and clusters
        data_2d = PCA(n_components=2).fit_transform(data_preprocessed)
        fig = plotting.plot_cluster_with_silhouette(data_preprocessed, data_2d, clustering.cluster_centers_, len(unique_clusters), clustering_labels)
        return unique_clusters, clustering_labels, ids, embeddings, fig

    elif pipeline_id == 1:
        # First Agglomerative clustering and after Kmeans
        # Data preprocessing
        data_preprocessed = pipeline(embeddings, scaler, pca)

        if agglo_params["n_clusters"] is None and agglo_params["distance_threshold"] is None:
            # If n_clusters and distance_threshold not defined asks user for an input
            dendrogram(data_preprocessed, agglo_params["linkage"])
            n = int(input("How much distance threshold: "))
            agglo_params["distance_threshold"] = n
        # Running hierarchical clustering
        agglomerative_clustering = AgglomerativeClustering(n_clusters=agglo_params["n_clusters"],
                                                           distance_threshold=agglo_params["distance_threshold"],
                                                           linkage=agglo_params["linkage"],
                                                           metric=agglo_params["metric"],
                                                           compute_full_tree=agglo_params["compute_full_tree"],
                                                           connectivity=agglo_params["connectivity"])
        # Calculate cluster and centers of the clusters
        clusters = agglomerative_clustering.fit_predict(data_preprocessed)
        unique_clusters = np.unique(clusters)
        cluster_centers = np.array([data_preprocessed[clusters == cluster].mean(axis=0) for cluster in unique_clusters])
        # Running kmeans with init the centers calculated by hierarchical clustering
        agglo_params["n_clusters"] = len(unique_clusters)
        kmeans_params["n_clusters"] = len(unique_clusters)
        clustering = KMeans(n_clusters=len(unique_clusters), init=cluster_centers, n_init=1,
                            random_state=kmeans_params["random_state"],  max_iter=kmeans_params["max_iter"],
                            tol=kmeans_params["tol"], verbose=kmeans_params["verbose"], copy_x=kmeans_params["copy_x"],
                            algorithm=kmeans_params["algorithm"])
        clustering_labels = clustering.fit_predict(data_preprocessed)
        data_2d = PCA(n_components=2).fit_transform(data_preprocessed)
        # Plotting data and clusters
        plotting.plot_cluster_with_silhouette(data_preprocessed, data_2d, clustering.cluster_centers_,
                                              len(unique_clusters), clustering_labels)
        return unique_clusters, clustering_labels, ids, embeddings, fig

    raise ValueError(f"Invalid pipeline_id: {pipeline_id}")


def dbscan(data, dbscan_params, preprocess):
    # Function that runs dbscan clustering algorithm
    ids = list(data.keys())
    embeddings = np.array(list(data.values()))
    data_preprocessed = pipeline(embeddings, preprocess["scaler"], preprocess["pca"])

    eps = float(input("Eps: "))
    dbscan_params["eps"] = eps
    min_samples = int(input("MinPts: "))
    dbscan_params["min_samples"] = min_samples

    dbscan_ = DBSCAN(eps=  dbscan_params["eps"], min_samples=dbscan_params["min_samples"], metric="cosine")
    clusters = dbscan_.fit_predict(data_preprocessed)

    unique_clusters = np.unique(clusters)
    data_2d = PCA(n_components=2).fit_transform(data_preprocessed)
    cluster_centers = np.array([data_preprocessed[clusters == cluster].mean(axis=0) for cluster in unique_clusters])

    return unique_clusters, clusters, ids, embeddings





def dendrogram(data, method):
    # Constructing a dendrogram and saving it for future use
    Z = linkage(data, method=method)
    scipy_dendrogram(Z, truncate_mode='level', p=10)  # Περικοπή για ευκολία προβολής
    plt.title(f"Dendrogram using ward linkage")
    plt.xlabel('Sample Index or Cluster Size')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

