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
from datetime import datetime
import uuid

import nx_parallel
def build_threshold_graph(corpus, threshold_params, preprocess, name, graph_params, save):
    if len(corpus) == 0:
        print("Warning: empty cluster, skipping.")
        return nx.Graph()

    if graph_params["Directed"] == False:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

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
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                if graph_params["Weighted"] == True:
                    G.add_edge(ids[i], ids[j], weight=sim)
                else:
                    G.add_edge(ids[i], ids[j])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if save == True:
        config = {}
        config["graph_type"] = "threshold graph"
        config["threshold"] = threshold_params
        config["preprocess"] = preprocess
        config["graph_construction"] = graph_params
        dt.save_graph_data(config, name)
        dt.save_graph(G, name)


    return G

def run_knn(data, preprocess, knn_params):
    """Running kth nearest neighbors algorithm for gpu and cpu. This is function is used for
    both mutual knn and knn graph construction
    data: All the document embeddings in the corpus
    preprocess: Preprocess parameters for scaler and pca
    knn_params: parameters for the kth nearest neighbors algorithm
    """

    try:
        # If gpu exists
        import cuml
        from cuml.neighbors import NearestNeighbors
        is_gpu = True
    except ImportError:
        # If doesnt gpu exist
        from sklearn.neighbors import NearestNeighbors
        print("Running on CPU using scikit-learn")
        is_gpu = False

    ids = list(data.keys())
    embeddings = np.array(list(data.values()))
    # Preprocessing the data embeddings
    data_pre = cl.pipeline(embeddings, scaler=preprocess["scaler"], pca=preprocess["pca"])

    if knn_params["n_neighbors"] < len(data):
       neighbors = knn_params["n_neighbors"]
    else:
        # if the embeddings are less than the n_neighbors drop the number of neighbors
       neighbors = len(data)
    if is_gpu:
        # If gpu perform knn in gpu
        if hasattr(data_pre, 'astype'):
            data_pre = data_pre.astype(np.float32)

        # Safety check because the gpu knn lib doesn't have all the metrics
        allowed_gpu_metrics = ['l2', 'euclidean', 'cosine', 'correlation', 'manhattan']
        metric = knn_params["metric"] if knn_params["metric"] in allowed_gpu_metrics else 'euclidean'

        nn = NearestNeighbors(
            n_neighbors=neighbors,
            metric=metric,
            algorithm='brute',
            p=knn_params["p"]
        )
        print("Gpu knn")
    else:
        # If only cpu exists use the sklearn libary
        nn = NearestNeighbors(
            n_neighbors=neighbors, metric=knn_params["metric"],
            algorithm=knn_params["algorithm"], radius=knn_params["radius"],
            leaf_size=knn_params["leaf_size"], p=knn_params["p"],
            metric_params=knn_params["metric_params"], n_jobs=knn_params["n_jobs"]
        )

    nn.fit(data_pre)
    distances, indices = nn.kneighbors(data_pre)
    if is_gpu:
         indices = indices.to_numpy() if hasattr(indices, 'to_numpy') else indices
         distances = distances.to_numpy() if hasattr(distances, 'to_numpy') else distances
    return indices, distances, ids

def build_knn_graph(data, knn_params, preprocess, name, graph_params, save):
    """Function that build a semantic graph where the connections are decides using the knn
       algorithm
       data: All the document embeddings in the corpus
       preprocess: Preprocess parameters for scaler and pca
       knn_params: parameters for the kth nearest neighbors algorithm
       name: Name of the graph
       graph_params: The parameters of the graph
       save: when its true save the graph when false dont save the graph
    """
    if len(data) == 0:
        print("Warning: empty cluster, skipping.")
        return nx.Graph()

    if graph_params["Directed"] == False:
       # Not directed graph
       G = nx.Graph()
    else:
        # Directed graph
       G = nx.DiGraph()
    # Running knn and return the connections and the distances
    indices, distances, ids = run_knn(data, preprocess, knn_params)
    # Constructing the graph
    for i, neighbors in enumerate(indices):
        node_id = ids[i]
        for j, nearest in enumerate(neighbors):
            if i != nearest:
                if knn_params["metric"] == "cosine":
                    sim = 1.0 - distances[i][j]
                elif knn_params["metric"] in ("euclidean", "minkowski", "manhattan"):
                    sim = np.exp(-distances[i][j])
                if graph_params["Weighted"] == True:
                    # Weighted graph
                    G.add_edge(node_id, ids[nearest], weight=sim)
                else:
                    # Not weighted graph
                    G.add_edge(node_id, ids[nearest])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    if save == True:
        # Saving the graph
        config = {}
        config["graph_type"] = "knn graph"
        config["knn"] = knn_params
        config["graph_construction"] = graph_params
        config["preprocess"] = (str(preprocess["scaler"]), str( preprocess["pca"]))

        dt.save_graph_data(config, name)
        dt.save_graph(G, name)


    return G


def build_mutual_knn_graph(data, knn_params, preprocess, name, graph_params, save):
    """Function that builds a semantic graph where the connections are decides using the mutual knn
           algorithm
           data: All the document embeddings in the corpus
           preprocess: Preprocess parameters for scaler and pca
           knn_params: parameters for the kth nearest neighbors algorithm
           name: Name of the graph
           graph_params: The parameters of the graph
           save: when its true save the graph when false dont save the graph
        """
    if len(data) == 0:
        print("Warning: empty cluster, skipping.")
        return nx.Graph()
    if graph_params["Directed"] == False:
        # Not directed graph
        G = nx.Graph()
    else:
        # Directed graph
        G = nx.DiGraph()
    # Running knn and return the connections and the distances
    indices, distances, ids = run_knn(data, preprocess, knn_params)
    # Constructing the graph
    d = {}
    for i, neighbors in enumerate(indices):
        d[i] = (list(neighbors[1:]), distances[i][1:])
    for id in d.keys():
        for i, neighbor in enumerate(d[id][0]):
            # if the two nodes are not neighbors both ways the connections doesn't happen
            if id < neighbor and neighbor in d and id in d[neighbor][0]:
                if knn_params["metric"] == "cosine":
                    sim = 1.0 - d[id][1][i]
                elif knn_params["metric"] in ("euclidean", "minkowski", "manhattan"):
                    sim = np.exp(-d[id][1][i])
                if graph_params["Weighted"] == True:
                    # Weighted graph
                    G.add_edge(ids[id], ids[neighbor], weight=sim)
                else:
                    # Not weighted graph
                    G.add_edge(ids[id], ids[neighbor])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if save == True:
        # Saving the graph
        config = {}
        config["graph_type"] = "mutual knn graph"
        config["knn"] = knn_params
        config["graph_construction"] = graph_params
        config["preprocess"] = (str(preprocess["scaler"]), str( preprocess["pca"]))
        dt.save_graph_data(config, name)
        dt.save_graph(G, name)

    return G

def build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess,name, save):
    """Building a semantic graph using knn but the data is pre clustered using kmeans or dbscan
       clustering_results: A dictionary with the pre clustered data and other information for the
       clustering
       knn_params: parameters for the kth nearest neighbors algorithm
       graph_params: The parameters of the graph
       save: when its true save the graph when false dont save the graph
       name: Name of the graph
    """
    graphs_knn = []
    # Constructing the graph
    for cluster_id in range(len(clustering_results["unique_clusters"])):
        # For each cluster makes a graph
        cluster_nodes = {
            clustering_results["ids"][i]: clustering_results["embeddings"][i]
            for i, label in enumerate(clustering_results["clustering_labels"])
            if label == cluster_id
        }
        graph = build_knn_graph(cluster_nodes, knn_params, clustering_results["preprocess"], "", graph_params, save)
        if graph.number_of_nodes() > 0:
            graphs_knn.append(graph)
    # Merging all the graphs in one
    knn_g = nx.compose_all(graphs_knn)
    # Saving the graph and info about its construction
    config = {}
    config["clustering"] = clustering_results["clustering_params"]
    config["preprocess"] = (str(clustering_results["preprocess"]["scaler"]), str(clustering_results["preprocess"]["pca"]))
    config["graph_type"] = "clustering knn graph"
    config["Graph building algorithm params"] = knn_params
    config["graph_params"] = graph_params
    config["Graph building algorithm"] = "KNN"
    graph_id = name
    dt.save_graph_data(config, graph_id, clustering_results["fig"])
    dt.save_graph(knn_g, graph_id)
    return
def build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess,name, save):
    """Building a semantic graph using mutual knn but the data is pre clustered using kmeans or dbscan
           clustering_results: A dictionary with the pre clustered data and other information for the
           clustering
           knn_params: parameters for the kth nearest neighbors algorithm
           graph_params: The parameters of the graph
           save: when its true save the graph when false dont save the graph
           name: Name of the graph
        """
    graphs_mutal_knn = []
    # Constructing the graph
    for cluster_id in range(len(clustering_results["unique_clusters"])):
        # For each cluster makes a graph
        cluster_nodes = {
            clustering_results["ids"][i]: clustering_results["embeddings"][i]
            for i, label in enumerate(clustering_results["clustering_labels"])
            if label == cluster_id
        }
        graph =build_mutual_knn_graph(cluster_nodes, knn_params, clustering_results["preprocess"], "", graph_params, save)
        if graph.number_of_nodes() > 0:
            graphs_mutal_knn.append(graph)
    # Merging all the graphs in one
    mutual_knn_g = nx.compose_all(graphs_mutal_knn)
    # Saving the graph and info about its construction
    config = {}
    config["clustering"] = clustering_results["clustering_params"]
    config["preprocess"] = (str(clustering_results["preprocess"]["scaler"]), str(clustering_results["preprocess"]["pca"]))
    config["graph_type"] = "clustering mutual knn graph"
    config["Building graph algorithm params"] = knn_params
    config["graph_params"] = graph_params
    config["Graph building algorithm"] = "MUTUAL KNN"
    graph_id = "mutual_" + name
    dt.save_graph_data(config, graph_id, clustering_results["fig"])
    dt.save_graph(mutual_knn_g, graph_id)

def build_clustering_threshold_graph(clustering_results, threshold_params, graph_params, preprocess, name, save):
        """Building a semantic graph using thershold method but the data is pre clustered using kmeans or dbscan
               clustering_results: A dictionary with the pre clustered data and other information for the
               clustering
               knn_params: parameters for the kth nearest neighbors algorithm
               graph_params: The parameters of the graph
               save: when its true save the graph when false don't save the graph
               name: Name of the graph
        """
        graphs_threshold = []
        # Constructing the graph
        for cluster_id in range(len(clustering_results["unique_clusters"])):
            # For each cluster makes a graph
            cluster_nodes = {
                clustering_results["ids"][i]: clustering_results["embeddings"][i]
                for i, label in enumerate(clustering_results["clustering_labels"])
                if label == cluster_id
            }
            graph = build_threshold_graph(cluster_nodes, threshold_params, clustering_results["preprocess"], "",
                                           graph_params, save)
            if graph.number_of_nodes() > 0:
                graphs_threshold.append(graph)
        # Merging all the graphs in one
        threshold_g = nx.compose_all(graphs_threshold)
        # Saving the graph and info about its construction
        config = {}
        config["clustering"] = clustering_results["clustering_params"]
        config["preprocess"] = (
        str(clustering_results["preprocess"]["scaler"]), str(clustering_results["preprocess"]["pca"]))
        config["graph_type"] = "clustering threshold graph"
        config["Building graph algorithm params"] = threshold_params
        config["graph_params"] = graph_params
        config["Graph building algorithm"] = "Threshold"
        graph_id =  name
        dt.save_graph_data(config, graph_id, clustering_results["fig"])
        dt.save_graph(threshold_g, graph_id)









