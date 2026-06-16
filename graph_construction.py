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


def build_knn_graph(corpus, knn_params, preprocess, name, graph_params, save):
    # Building a graph where the edges are formed using the knn algorithm
    if len(corpus) == 0:
        print("Warning: empty cluster, skipping.")
        return nx.Graph()
    if graph_params["Directed"] == False:
       G = nx.Graph()
    else:
       G = nx.DiGraph()

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
                if knn_params["metric"] == "cosine":
                    sim = 1.0 - distances[i][j]
                elif knn_params["metric"] in ("euclidean", "minkowski", "manhattan"):
                    sim = np.exp(-distances[i][j])

                if graph_params["Weighted"] == True:
                    G.add_edge(node_id, ids[nearest], weight=sim)
                else:
                    G.add_edge(node_id, ids[nearest])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    if save == True:
        config = {}
        config["graph_type"] = "knn graph"
        config["knn"] = knn_params
        config["graph_construction"] = graph_params
        config["preprocess"] = (str(preprocess["scaler"]), str( preprocess["pca"]))

        dt.save_graph_data(config, name)
        dt.save_graph(G, name)


    return G


def build_mutual_knn_graph(corpus, knn_params, preprocess, name, graph_params, save):
    # Building a graph where the edges are formed using the mutual knn algorithm
    if len(corpus) == 0:
        print("Warning: empty cluster, skipping.")
        return nx.Graph()
    if graph_params["Directed"] == False:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
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
        d[i] = (list(neighbors[1:]), distances[i][1:])

    for id in d.keys():
        for i, neighbor in enumerate(d[id][0]):
            if id < neighbor and neighbor in d and id in d[neighbor][0]:
                if knn_params["metric"] == "cosine":
                    sim = 1.0 - d[id][1][i]
                elif knn_params["metric"] in ("euclidean", "minkowski", "manhattan"):
                    sim = np.exp(-d[id][1][i])
                if graph_params["Weighted"] == True:
                    G.add_edge(ids[id], ids[neighbor], weight=sim)
                else:
                    G.add_edge(ids[id], ids[neighbor])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if save == True:
        config = {}
        config["graph_type"] = "mutual knn graph"
        config["knn"] = knn_params
        config["graph_construction"] = graph_params
        config["preprocess"] = (str(preprocess["scaler"]), str( preprocess["pca"]))
        dt.save_graph_data(config, name)
        dt.save_graph(G, name)

    return G

def build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess,name, flag):
    graphs_knn = []
    # Constructing the graph
    for cluster_id in range(len(clustering_results["unique_clusters"])):
        cluster_nodes = {
            clustering_results["ids"][i]: clustering_results["embeddings"][i]
            for i, label in enumerate(clustering_results["clustering_labels"])
            if label == cluster_id
        }
        graph = build_knn_graph(cluster_nodes, knn_params, clustering_results["preprocess"], "", graph_params, flag)
        if graph.number_of_nodes() > 0:
            graphs_knn.append(graph)

    knn_g = nx.compose_all(graphs_knn)

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
def build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess,name, flag):
    graphs_mutal_knn = []
    # Constructing the graph
    for cluster_id in range(len(clustering_results["unique_clusters"])):
        cluster_nodes = {
            clustering_results["ids"][i]: clustering_results["embeddings"][i]
            for i, label in enumerate(clustering_results["clustering_labels"])
            if label == cluster_id
        }
        graph =build_mutual_knn_graph(cluster_nodes, knn_params, clustering_results["preprocess"], "", graph_params, flag)
        if graph.number_of_nodes() > 0:
            graphs_mutal_knn.append(graph)
    mutual_knn_g = nx.compose_all(graphs_mutal_knn)

    config = {}
    config["clustering"] = clustering_results["clustering_params"]
    config["preprocess"] = (str(clustering_results["preprocess"]["scaler"]), str(clustering_results["preprocess"]["pca"]))
    config["graph_type"] = "clustering mutual knn graph"
    config["Building graph algorithm params"] = knn_params
    config["graph_params"] = graph_params
    config["Graph building algorithm"] = "MUTUAL KNN"
    graph_id = " mutual_" + name
    dt.save_graph_data(config, graph_id, clustering_results["fig"])
    dt.save_graph(mutual_knn_g, graph_id)

def build_clustering_threshold_graph(clustering_results, threshold_params, graph_params, preprocess, name, flag):
        graphs_threshold = []
        # Constructing the graph
        for cluster_id in range(len(clustering_results["unique_clusters"])):
            cluster_nodes = {
                clustering_results["ids"][i]: clustering_results["embeddings"][i]
                for i, label in enumerate(clustering_results["clustering_labels"])
                if label == cluster_id
            }
            graph = build_threshold_graph(cluster_nodes, threshold_params, clustering_results["preprocess"], "",
                                           graph_params, flag)
            if graph.number_of_nodes() > 0:
                graphs_threshold.append(graph)
        threshold_g = nx.compose_all(graphs_threshold)

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

def build_clustering_graph(data, kmeans_params, agglo_params, knn_params, dbscan_params, preprocess, name , graph_params, algorithm="kmeans", clustering_result=None, flag=True):
    # Building multiples graphs using clustering
    config = {}
    if algorithm == "kmeans" and clustering_result == None:
       unique_clusters, clustering_labels, ids, embeddings, fig = cl.kmeans(data, kmeans_params, agglo_params  ,scaler=preprocess["scaler"],
                                                                    pca=preprocess["pca"], pipeline_id=kmeans_params["pipeline_id"])
       clustering_result = {
           "unique_clusters": unique_clusters,
           "clustering_labels": clustering_labels,
           "ids": ids,
           "embeddings": embeddings,
           "fig": fig,
           "clustering_params": kmeans_params
       }
       if flag == True:
           return clustering_result


    elif algorithm == "dbscan" and clustering_result == None:
       cl.dbscan(data, dbscan_params, preprocess)

       return

    else:
        unique_clusters = clustering_result["unique_clusters"]
        clustering_labels = clustering_result["clustering_labels"]
        ids = clustering_result["ids"]
        embeddings = clustering_result["embeddings"]
        fig = clustering_result["fig"]
        config = {}
        config["kmeans"] = kmeans_params
        config["preprocess"] = (str(preprocess["scaler"]), str(preprocess["pca"]))
        if kmeans_params["pipeline_id"] == 1:
            config["agglo_params"] = agglo_params


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
        graphs_knn.append(build_knn_graph(cluster_nodes, knn_params, preprocess,"", graph_params, False))
        graphs_mutual_knn.append(build_mutual_knn_graph(cluster_nodes, knn_params, preprocess,"", graph_params,False))


    knn_g = nx.compose_all(graphs_knn)
    mutual_knn_g = nx.compose_all(graphs_mutual_knn)

    temp = config.copy()
    temp["graph_type"] = "clustering knn graph"
    temp["knn"] = knn_params
    graph_id = name
    dt.save_graph_data(temp, graph_id, fig)
    dt.save_graph(knn_g, graph_id)

    temp = config.copy()
    temp["graph_type"] = "clustering mutual knn graph"
    temp["mutual_knn"] = knn_params
    graph_id = " mutual_" + name
    dt.save_graph_data(temp, graph_id, fig)
    dt.save_graph(mutual_knn_g, graph_id)







