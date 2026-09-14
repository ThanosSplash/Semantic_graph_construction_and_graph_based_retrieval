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
import itertools
from copy import deepcopy


import nx_parallel
def build_threshold_graph(corpus, threshold_params, preprocess, name, graph_params, save):
    if len(corpus) == 0:
        return nx.Graph()
    if graph_params["Directed"] == False:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    ids = list(corpus.keys())
    embeddings = np.array(list(corpus.values()))
    data = cl.preprocessing_pipeline(
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
                    G.add_edge(ids[i], ids[j], weight=1.0)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if save == True:
        config = {}
        config["graph_type"] = "threshold graph"
        config["graph_name"] = f"threshold graph threshold = {threshold_params['threshold_distance']}"
        config["Graph building algorithm params"] = threshold_params
        config["preprocess"] = (str(preprocess["scaler"]), str( preprocess["pca"]))
        config["graph_construction"] = graph_params
        config["Graph building algorithm"] = "Threshold"
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

    ids = list(data.keys())
    embeddings = np.array(list(data.values()))
    # Preprocessing the data embeddings
    data_pre = cl.preprocessing_pipeline(embeddings, scaler=preprocess["scaler"], pca=preprocess["pca"])

    if knn_params["n_neighbors"] < len(data):
       neighbors = knn_params["n_neighbors"]
    else:
        # if the embeddings are less than the n_neighbors drop the number of neighbors
       neighbors = len(data)
    nn = NearestNeighbors(
            n_neighbors=neighbors, metric=knn_params["metric"],
            algorithm=knn_params["algorithm"], radius=knn_params["radius"],
            leaf_size=knn_params["leaf_size"], p=knn_params["p"],
            metric_params=knn_params["metric_params"], n_jobs=knn_params["n_jobs"]
        )

    nn.fit(data_pre)
    distances, indices = nn.kneighbors(data_pre)
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
    for id in data:
        G.add_node(id)
    for i, neighbors in enumerate(indices):
        node_id = ids[i]
        for j, nearest in enumerate(neighbors):
            if i != nearest:
                if knn_params["metric"] == "cosine":
                    sim = 1.0 - distances[i][j]
                elif knn_params["metric"] in ("euclidean", "manhattan", "chebyshev"):
                    sim = np.exp(-distances[i][j])
                if graph_params["Weighted"] == True:
                    # Weighted graph
                    G.add_edge(node_id, ids[nearest], weight=sim)
                else:
                    # Not weighted graph
                    G.add_edge(node_id, ids[nearest], weight=1.0)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    if save == True:
        # Saving the graph
        config = {}
        config["graph_type"] = "knn graph"
        config["graph_name"] = f"knn graph neighbors = {knn_params['n_neighbors']}"
        config["Graph building algorithm params"] = knn_params
        config["graph_construction"] = graph_params
        config["preprocess"] = (str(preprocess["scaler"]), str( preprocess["pca"]))
        config["Graph building algorithm"] = "KNN"

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
    for id in data:
        G.add_node(id)
    for i, neighbors in enumerate(indices):
        d[i] = (list(neighbors[1:]), distances[i][1:])
    for id in d.keys():
        for i, neighbor in enumerate(d[id][0]):
            # if the two nodes are not neighbors both ways the connections doesn't happen
            if id < neighbor and neighbor in d and id in d[neighbor][0]:
                if knn_params["metric"] == "cosine":
                    sim = 1.0 - d[id][1][i]
                elif knn_params["metric"] in ("euclidean", "manhattan", "chebyshev"):
                    sim = np.exp(-d[id][1][i])
                if graph_params["Weighted"] == True:
                    # Weighted graph
                    G.add_edge(ids[id], ids[neighbor], weight=sim)
                else:
                    # Not weighted graph
                    G.add_edge(ids[id], ids[neighbor], weight=1.0)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if save == True:
        # Saving the graph
        config = {}
        config["graph_type"] = "mutual knn graph"
        config["graph_name"] = f"mutual knn graph neighbors = {knn_params['n_neighbors']}"
        config["Graph building algorithm params"] = knn_params
        config["graph_construction"] = graph_params
        config["preprocess"] = (str(preprocess["scaler"]), str( preprocess["pca"]))
        config["Graph building algorithm"] = "MUTUAL KNN"
        dt.save_graph_data(config, name)
        dt.save_graph(G, name)

    return G

def build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess,
                               name, save_cluster_graphs, clustering_algo, save_total_graph):
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
        graph = build_knn_graph(cluster_nodes, knn_params, clustering_results["preprocess"], "", graph_params, save_cluster_graphs)
        if graph.number_of_nodes() > 0:
            graphs_knn.append(graph)
    # Merging all the graphs in one
    knn_g = nx.compose_all(graphs_knn)
    # Saving the graph and info about its construction
    if save_total_graph is True:
        config = {}
        config["clustering"] = clustering_results["clustering_params"]
        config["preprocess"] = (str(clustering_results["preprocess"]["scaler"]), str(clustering_results["preprocess"]["pca"]))
        if clustering_algo !="dbscan":
            config["graph_name"] = (f"{clustering_algo} knn graph neighbors = {knn_params['n_neighbors']} "
                                    f"clusters = {clustering_results['clustering_params']['n_clusters']}")
        else:
            config["graph_name"] = (f"{clustering_algo} knn graph neighbors = {knn_params['n_neighbors']} "
                                    f"eps = {clustering_results['clustering_params']['eps']}  min_samples = {clustering_results['clustering_params']['min_samples']}")
        config["graph_type"] = f"{clustering_algo} knn graph"
        config["Graph building algorithm params"] = knn_params
        config["graph_params"] = graph_params
        config["Graph building algorithm"] = "KNN"
        graph_id = name
        dt.save_graph_data(config, graph_id, clustering_results["fig"])
        dt.save_graph(knn_g, graph_id)
    return knn_g
def build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess,
                                      name, save_cluster_graphs, clustering_algo, save_total_graph):
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
        graph =build_mutual_knn_graph(cluster_nodes, knn_params, clustering_results["preprocess"], "", graph_params, save_cluster_graphs)
        if graph.number_of_nodes() > 0:
            graphs_mutal_knn.append(graph)
    # Merging all the graphs in one
    mutual_knn_g = nx.compose_all(graphs_mutal_knn)
    # Saving the graph and info about its construction
    if save_total_graph is True:
        config = {}
        config["clustering"] = clustering_results["clustering_params"]
        config["preprocess"] = (str(clustering_results["preprocess"]["scaler"]), str(clustering_results["preprocess"]["pca"]))
        config["graph_type"] = f"{clustering_algo} mutual knn graph"
        config["graph_name"] = (f"{clustering_algo} mutual knn graph neighbors = {knn_params['n_neighbors']} "
                                f"clusters = {clustering_results['clustering_params']['n_clusters']}")
        config["Graph building algorithm params"] = knn_params
        config["graph_params"] = graph_params
        config["Graph building algorithm"] = "MUTUAL KNN"
        graph_id = "mutual_" + name
        dt.save_graph_data(config, graph_id, clustering_results["fig"])
        dt.save_graph(mutual_knn_g, graph_id)
    return mutual_knn_g

def build_clustering_threshold_graph(clustering_results, threshold_params, graph_params, preprocess,
                                     name, save_cluster_graphs, clustering_algo, save_total_graph):
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
                                           graph_params, save_cluster_graphs)
            if graph.number_of_nodes() > 0:
                graphs_threshold.append(graph)
        # Merging all the graphs in one
        threshold_g = nx.compose_all(graphs_threshold)
        # Saving the graph and info about its construction
        if save_total_graph is True:
            config = {}
            config["clustering"] = clustering_results["clustering_params"]

            config["preprocess"] = (
            str(clustering_results["preprocess"]["scaler"]), str(clustering_results["preprocess"]["pca"]))
            config["graph_type"] = f"{clustering_algo} threshold graph"
            config["graph_name"] = (f"{clustering_algo} threshold graph threshold = {threshold_params['threshold_distance']} "
                                    f"clusters = {clustering_results['clustering_params']['n_clusters']}")
            config["Graph building algorithm params"] = threshold_params
            config["graph_params"] = graph_params
            config["Graph building algorithm"] = "Threshold"
            graph_id = name
            dt.save_graph_data(config, graph_id, clustering_results["fig"])
            dt.save_graph(threshold_g, graph_id)
        return threshold_g




kmeans_grid = {
    "n_clusters": [5, 10, 20],
    "init":       ["k-means++"],
    "random_state": [42]
}

knn_grid = {
    "n_neighbors": [5, 10, 20],
    "metric":      ["cosine"]
}
threshold_grid = {
    "threshold_distance": [0.8, 0.6, 0.4]

}

dbscan_grid = {
    "eps":         [0.4827586206896552],
    "min_samples": [22],
    "metric": ["cosine"]
}

agglo_grid = {
    "n_clusters": [3],
    "distance_threshold_on_agglo": [30],
    "init": ["k-means++"],
    "random_state": [42],
    "linkage": ["ward"]

}

graph_combinations = [{"Directed": False, "Weighted": True}]
preprocess_combinations = [{"scaler": None, "pca": None}]

# ── 2. Base param dicts (non-varied keys stay fixed) ─────────────────────────

BASE_KMEANS = {
    "init": "k-means++", "n_init": "auto", "max_iter": 300,
    "tol": 1e-4, "verbose": 0, "random_state": 42,
    "copy_x": True, "algorithm": "lloyd", "pipeline_id": 0
}
BASE_KNN = {
    "radius": 1.0, "algorithm": "auto", "leaf_size": 30,
    "p": 2, "metric_params": None, "n_jobs": None,
}
BASE_DBSCAN = {
    "metric_params": None,
    "algorithm": "auto", "leaf_size": 30, "p": None, "n_jobs": None,
}
BASE_AGGLO = {
    "metric": "euclidean", "compute_full_tree": "auto", "connectivity": None,
}

result_file = "Outputs/"
BASE_THRESHOLD = {"threshold_distance": 0.5}

def make_name(prefix, varied, graph, clustering = None) :
    parts = []
    if clustering is not None:
        for k, v in clustering.items():
            short_key = k.replace("n_clusters", "clusters_") \
                         .replace("eps", "eps_") \
                         .replace("min_samples", "ms")\
                         .replace("random_state", "seed")
            parts.append(f"{short_key}{v}")
    for k, v in graph.items():
        short_key = k.replace("Directed", "Directed_")  \
                     .replace("Weighted", "Weighted_")
        parts.append(f"{short_key}{v}")
    for k, v in varied.items():
        short_key = k.replace("n_clusters", "clusters_")  \
                     .replace("n_neighbors", "neighbors_")  \
                     .replace("distance_threshold", "dt_") \
                     .replace("min_samples", "ms_") \
                     .replace("init", "init_")
        parts.append(f"{short_key}{v}")

    return f"{prefix}_{'_'.join(parts)}"

# ── 4. Grid helpers ───────────────────────────────────────────────────────────

def grid_combinations(grid: dict):

    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))

def merge(base: dict, overrides: dict) -> dict:
    p = deepcopy(base)
    p.update(overrides)
    return p

def make_graphs(c):
    for graph_c in graph_combinations:
     graph_params= {"Directed": graph_c["Directed"], "Weighted": graph_c["Weighted"]}

     for pre in preprocess_combinations:
        preprocess = {"scaler": pre["scaler"], "pca": pre["pca"]}
        pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
        # ── knn_Graph ────────────────────────────────────────────────────────
        for knn_combo in grid_combinations(knn_grid):
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"knn_{pre_tag}", knn_combo, graph_params)
            print(f"[knn_Graph]   {name}")
            build_knn_graph(c, knn_params, preprocess, name, graph_params, True)

        # ── mutual_knn ───────────────────────────────────────────────────────
        for knn_combo in grid_combinations(knn_grid):
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"mutual_{pre_tag}", knn_combo, graph_params)
            print(f"[mutual_knn]  {name}")
            build_mutual_knn_graph(c, knn_params, preprocess, name, graph_params, True)
        # ── threshold_graph ───────────────────────────────────────────────────────
        for threshold_combo in grid_combinations(threshold_grid):
            threshold_params = merge(BASE_THRESHOLD, threshold_combo)
            name = make_name(f"threshold_{pre_tag}", threshold_combo, graph_params)
            build_threshold_graph(c, threshold_params, preprocess, name, graph_params, True)
        # ── kmeans clustering ───────────────────────────────────────────────────────
        make_kmeans_knn_graph_for_all_params(c, preprocess, graph_params, False, True, 0, "kmeans")
        make_kmeans_mutual_knn_graph_for_all_params(c, preprocess, graph_params, False, True, 0, "kmeans")
        make_kmeans_threshold_graph_for_all_params(c, preprocess, graph_params, False, True, 0, "kmeans")


def make_kmeans_knn_graph_for_all_params(corpus, preprocess, graph_params, saving_cluster_graphs, saving_total_graph, pipeline_id, cl_name):
    pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
    if pipeline_id == 0:
        cl_grid = kmeans_grid
    else:
        cl_grid = agglo_grid

    for kmeans_combo in grid_combinations(cl_grid):
        kmeans_params = merge(BASE_KMEANS, kmeans_combo)
        if pipeline_id == 0:
            agglo_params = {}
            algorithm = "kmeans"
        else:
            agglo_params = merge(BASE_AGGLO, kmeans_combo)
            algorithm = "agglo_kmeans"
        kmeans_params["pipeline_id"] = pipeline_id
        clustering_results = cl.perform_clustering(data=corpus, algorithm=algorithm, kmeans_params=kmeans_params,
                                                   agglo_params=agglo_params, dbscan_params={}, preprocess=preprocess)
        for knn_combo in grid_combinations(knn_grid):
                knn_params = merge(BASE_KNN, knn_combo)

                name = make_name(f"knn_{cl_name}_{pre_tag}", knn_combo, graph_params, kmeans_combo)
                build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, saving_cluster_graphs, algorithm, saving_total_graph)

def make_kmeans_mutual_knn_graph_for_all_params(corpus, preprocess, graph_params, saving_cluster_graphs, saving_total_graph, pipeline_id, cl_name):
    pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
    if pipeline_id == 0:
        cl_grid = kmeans_grid
    else:
        cl_grid = agglo_grid

    for kmeans_combo in grid_combinations(cl_grid):
        kmeans_params = merge(BASE_KMEANS, kmeans_combo)
        if pipeline_id == 0:
            agglo_params = {}
            algorithm = "kmeans"
        else:
            agglo_params = merge(BASE_AGGLO, kmeans_combo)
            algorithm = "agglo_kmeans"
        kmeans_params["pipeline_id"] = pipeline_id
        clustering_results = cl.perform_clustering(data=corpus, algorithm=algorithm, kmeans_params=kmeans_params,
                                                   agglo_params=agglo_params, dbscan_params={}, preprocess=preprocess)
        for knn_combo in grid_combinations(knn_grid):
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"knn_{cl_name}_{pre_tag}", knn_combo, graph_params, kmeans_combo)
            build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, saving_cluster_graphs,
                                              algorithm, saving_total_graph)

def make_kmeans_threshold_graph_for_all_params(corpus, preprocess, graph_params, saving_cluster_graphs, saving_total_graph, pipeline_id, cl_name):
    pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
    if pipeline_id == 0:
        cl_grid = kmeans_grid
    else:
        cl_grid = agglo_grid

    for kmeans_combo in grid_combinations(cl_grid):
        kmeans_params = merge(BASE_KMEANS, kmeans_combo)
        if pipeline_id == 0:
            agglo_params = {}
            algorithm = "kmeans"
        else:
            agglo_params = merge(BASE_AGGLO, kmeans_combo)
            algorithm = "agglo_kmeans"
        kmeans_params["pipeline_id"] = pipeline_id
        clustering_results = cl.perform_clustering(data=corpus, algorithm=algorithm, kmeans_params=kmeans_params,
                                                   agglo_params=agglo_params, dbscan_params={}, preprocess=preprocess)
        for threshold_combo in grid_combinations(threshold_grid):
            threshold_params = merge(BASE_THRESHOLD, threshold_combo)
            name = make_name(f"threshold_{cl_name}_{pre_tag}", threshold_combo, graph_params, kmeans_combo)
            build_clustering_threshold_graph(clustering_results, threshold_params, graph_params, preprocess, name,
                                             saving_cluster_graphs, algorithm, saving_total_graph)


def make_dbscan_knn_graph_for_all_params(corpus, preprocess, graph_params, saving_cluster_graphs, saving_total_graph):
    pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
    for dbscan_combo in grid_combinations(dbscan_grid):
        dbscan_params = merge(BASE_DBSCAN, dbscan_combo)
        name = make_name(f"dbscan_{pre_tag}", dbscan_combo, graph_params)
        clustering_results = cl.perform_clustering(data=corpus, algorithm="dbscan", kmeans_params={}, agglo_params={},
                                                  dbscan_params=dbscan_params, preprocess=preprocess)
        for knn_combo in grid_combinations(knn_grid):
             knn_params = merge(BASE_KNN, knn_combo)
             name = make_name(f"knn_dbscan_{pre_tag}", dbscan_combo, graph_params, dbscan_combo)
             build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name,saving_cluster_graphs,
                                               "dbscan", saving_total_graph)
def make_dbscan_mutual_knn_graph_for_all_params(corpus, preprocess, graph_params, saving_cluster_graphs, saving_total_graph):
    pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
    for dbscan_combo in grid_combinations(dbscan_grid):
        dbscan_params = merge(BASE_DBSCAN, dbscan_combo)
        name = make_name(f"dbscan_{pre_tag}", dbscan_combo, graph_params)
        clustering_results = cl.perform_clustering(data=corpus, algorithm="dbscan", kmeans_params={}, agglo_params={},
                                                  dbscan_params=dbscan_params, preprocess=preprocess)
        for knn_combo in grid_combinations(knn_grid):
             knn_params = merge(BASE_KNN, knn_combo)
             name = make_name(f"knn_dbscan_{pre_tag}", dbscan_combo, graph_params, dbscan_combo)
             build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name,saving_cluster_graphs,
                                               "dbscan", saving_total_graph)

def make_dbscan_threshold_graph_for_all_params(corpus, preprocess, graph_params, saving_cluster_graphs, saving_total_graph):
    pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
    for dbscan_combo in grid_combinations(dbscan_grid):
         dbscan_params = merge(BASE_DBSCAN, dbscan_combo)
         clustering_results = cl.perform_clustering(data=corpus, algorithm="dbscan", kmeans_params={}, agglo_params={},
                                               dbscan_params=dbscan_params, preprocess=preprocess)
         for threshold_combo in grid_combinations(threshold_grid):
            threshold_params = merge(BASE_THRESHOLD, threshold_combo)
            name = make_name(f"threshold_dbscan{pre_tag}", threshold_combo, graph_params, dbscan_combo)
            build_clustering_threshold_graph(clustering_results, threshold_params, graph_params, preprocess, name, saving_cluster_graphs,
                                             "dbscan", saving_total_graph)

def make_kmeans_graph_for_all_params(c, pipeline_id, cl_name):
    for graph_c in graph_combinations:
        graph_params = {"Directed": graph_c["Directed"], "Weighted": graph_c["Weighted"]}

    for pre in preprocess_combinations:
        preprocess = {"scaler": pre["scaler"], "pca": pre["pca"]}
        make_kmeans_knn_graph_for_all_params(c, preprocess, graph_params, False, True, pipeline_id, cl_name)
        make_kmeans_mutual_knn_graph_for_all_params(c, preprocess, graph_params, False, True, pipeline_id, cl_name)
        make_kmeans_threshold_graph_for_all_params(c, preprocess, graph_params, False, True,pipeline_id, cl_name)

def make_dbscan_graph_for_all_params(c):
    for graph_c in graph_combinations:
        graph_params = {"Directed": graph_c["Directed"], "Weighted": graph_c["Weighted"]}

    for pre in preprocess_combinations:
        preprocess = {"scaler": pre["scaler"], "pca": pre["pca"]}
        make_dbscan_knn_graph_for_all_params(c, preprocess, graph_params, False, True)
        make_dbscan_mutual_knn_graph_for_all_params(c, preprocess, graph_params, False, True)
        make_dbscan_threshold_graph_for_all_params(c, preprocess, graph_params, False, True)