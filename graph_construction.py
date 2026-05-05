import networkx as nx
from itertools import combinations
import numpy as np
import data_loading as dt
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import clustering as cl
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def build_threshold_graph(corpus, threshold_params, preprocess, name, save):
    G = nx.Graph()

    ids = list(corpus.keys())
    embeddings = np.array(list(corpus.values()))


    data = cl.pipeline(
        embeddings,
        scaler=preprocess["scaler"],
        pca=preprocess["pca"]
    )


    for id in ids:
        G.add_node(id)
    sim_matrix = cosine_similarity(data)

    threshold = threshold_params["threshold_distance"]


    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):  # avoid duplicates & self-loops
            sim = sim_matrix[i, j]
            if sim >= threshold:
                G.add_edge(ids[i], ids[j], weight=sim)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if save == True:
       dt.save_graph(G, name)


    return G


def build_knn_graph(corpus, knn_params, preprocess, name, save):
    # Building a graph where the edges are formed using the knn algorithm
    G = nx.Graph()
    for id in corpus:
        G.add_node(id)
    ids = list(corpus.keys())
    embeddings = np.array(list(corpus.values()))
    data = cl.pipeline(embeddings, scaler=preprocess["scaler"], pca=preprocess["pca"])
    nn = NearestNeighbors(n_neighbors=knn_params["n_neighbors"], metric=knn_params["metric"],
                          algorithm=knn_params["algorithm"], radius=knn_params["radius"],
                          leaf_size=knn_params["leaf_size"], p=knn_params["p"],
                          metric_params=knn_params["metric_params"], n_jobs=knn_params["n_jobs"])
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

    # Constructing the graph
    for i, neighbors in enumerate(indices):
        node_id = ids[i]
        for j, nearest in enumerate(neighbors):
            if i != nearest:
                sim = 1.0 - distances[i][j]
                G.add_edge(node_id, ids[nearest], weight=distances[i][j])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    if save == True:
        dt.save_graph(G, name)


    return G


def build_mutual_knn_graph(corpus, knn_params, preprocess, name, save):
    # Building a graph where the edges are formed using the mutual knn algorithm
    G = nx.Graph()
    for id in corpus:
        G.add_node(id)

    ids = list(corpus.keys())
    embeddings = np.array(list(corpus.values()))
    data = cl.pipeline(embeddings, scaler=preprocess["scaler"], pca=preprocess["pca"])
    nn = NearestNeighbors(n_neighbors=knn_params["n_neighbors"], metric=knn_params["metric"],
                          algorithm=knn_params["algorithm"], radius=knn_params["radius"],
                          leaf_size=knn_params["leaf_size"], p=knn_params["p"],
                          metric_params=knn_params["metric_params"], n_jobs=knn_params["n_jobs"])
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    # Constructing the graph
    d = {}
    for i, neighbors in enumerate(indices):
        d[i] = (list(neighbors[1:]), distances[i][1:])  # list not set

    for id in d.keys():
        for i, neighbor in enumerate(d[id][0]):
            if id < neighbor and neighbor in d and id in d[neighbor][0]:
                sim = 1.0 - d[id][1][i]
                G.add_edge(ids[id], ids[neighbor], weight=sim)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if save == True:
        dt.save_graph(G, name)

    return G


def build_clustering_graph(data, kmeans_params, threshold_params, agglo_params, knn_params, dbscan_params, preprocess, algorithm="kmeans"):
    # Building multiples graphs using clustering
    if algorithm == "kmeans":
       unique_clusters, clustering_labels, ids, embeddings = cl.kmeans(data, kmeans_params, agglo_params, scaler=preprocess["scaler"],
                                                                    pca=preprocess["pca"], pipeline_id=kmeans_params["pipeline_id"])
    elif algorithm == "dbscan":
       return


    graphs_knn = []
    graphs_mutual_knn = []
    graphs_threshold = []
    # Constructing the graph
    for cluster_id in range(len(unique_clusters)):
        cluster_nodes = {
            ids[i]: embeddings[i]
            for i, label in enumerate(clustering_labels)
            if label == cluster_id
        }
        graphs_knn.append(build_knn_graph(cluster_nodes, knn_params, preprocess,"", False))
        graphs_mutual_knn.append(build_mutual_knn_graph(cluster_nodes, knn_params, preprocess,"", False))
        graphs_threshold.append(build_threshold_graph(cluster_nodes,threshold_params, preprocess,"", False))

    knn_g = nx.compose_all(graphs_knn)
    mutual_knn_g = nx.compose_all(graphs_mutual_knn)
    threshold_g = nx.compose_all(graphs_threshold)
    dt.save_graph(knn_g, "Graph_KNN_Kmeans.gpickle")
    dt.save_graph(mutual_knn_g, "Graph_Mutual_KNN_Kmeans.gpickle")
    dt.save_graph(threshold_g, "Graph_Threshold.gpickle")


    return




