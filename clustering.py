from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram  # explicit import

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
def kmeans():
    return


def kmeans_pipeline(data, threshold_distance):

    ids = list(data.keys())
    embeddings = np.array(list(data.values()))

    scaler = StandardScaler()
    pca = PCA()
    pca.fit(scaler.fit_transform(embeddings))

    variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance)
    threshold_variance = 0.85
    n_components = np.argmax(cumulative_variance >= threshold_variance) + 1
    print(n_components)
    pca_plot = PCA(n_components=n_components)
    data_pca = pca_plot.fit_transform(embeddings)

    agglomerative_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold_distance,
                                                       linkage='ward')
    clusters1 = agglomerative_clustering.fit_predict(data_pca)

    unique_clusters = np.unique(clusters1)
    cluster_centers = np.array([data_pca[clusters1 == cluster].mean(axis=0) for cluster in unique_clusters])

    clustering = KMeans(n_clusters=len(unique_clusters), init=cluster_centers, n_init=1, random_state=42)
    clustering_labels = clustering.fit_predict(data_pca)

    return unique_clusters, clustering_labels, ids


def dendrogram(data, name):
    ids = list(data.keys())
    embeddings = np.array(list(data.values()))

    scaler = StandardScaler()
    pca = PCA()
    pca.fit(scaler.fit_transform(embeddings))
    variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance)
    threshold_variance = 0.85
    n_components = np.argmax(cumulative_variance >= threshold_variance) + 1

    print(n_components)
    pca_plot = PCA(n_components=n_components)
    data_pca = pca_plot.fit_transform(embeddings)
    Z = linkage(data_pca, method="ward")
    scipy_dendrogram(Z, truncate_mode='level', p=10)  # Περικοπή για ευκολία προβολής
    plt.title(f"Dendrogram using ward linkage")
    plt.xlabel('Sample Index or Cluster Size')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig("Data/"+name, dpi=300, bbox_inches="tight")
