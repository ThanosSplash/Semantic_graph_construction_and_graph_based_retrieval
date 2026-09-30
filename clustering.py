from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram  # explicit import

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

import clustering
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



def preprocessing_pipeline(data, scaler, pca):
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
        data_preprocessed = preprocessing_pipeline(embeddings, scaler, pca)
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
        data_preprocessed = preprocessing_pipeline(embeddings, scaler, pca)
        # Running hierarchical clustering
        agglomerative_clustering = AgglomerativeClustering(n_clusters=None,
                                                           distance_threshold=agglo_params["distance_threshold_on_agglo"],
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
        fig = plotting.plot_cluster_with_silhouette(data_preprocessed, data_2d, clustering.cluster_centers_,
                                              len(unique_clusters), clustering_labels)
        return unique_clusters, clustering_labels, ids, embeddings, fig

    raise ValueError(f"Invalid pipeline_id: {pipeline_id}")





def perform_clustering(data, algorithm, kmeans_params,
                       agglo_params, preprocess):
    if algorithm == "kmeans" or algorithm == "agglo_kmeans":
        unique_clusters, clustering_labels, ids, embeddings, fig = kmeans(data, kmeans_params, agglo_params,
                                                        scaler=preprocess["scaler"],pca=preprocess["pca"],pipeline_id=kmeans_params["pipeline_id"])
        if kmeans_params["pipeline_id"] == 0:
            clustering_result = {
            "unique_clusters": unique_clusters,
            "clustering_labels": clustering_labels,
            "ids": ids,
            "embeddings": embeddings,
            "fig": fig,
            "clustering_params": kmeans_params,
            "preprocess": preprocess,
            "clustering_algorithm": "kmeans"
        }
        elif kmeans_params["pipeline_id"] == 1:
            clustering_result = {
                "unique_clusters": unique_clusters,
                "clustering_labels": clustering_labels,
                "ids": ids,
                "embeddings": embeddings,
                "fig": fig,
                "clustering_params": {**kmeans_params, **agglo_params},
                "preprocess": preprocess,
                "clustering_algorithm": "kmeans"
            }
        return clustering_result
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")



