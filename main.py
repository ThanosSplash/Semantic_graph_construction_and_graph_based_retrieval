import experiments as ex
import data_loading as dt
import retrieval as rt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotting as pt
import graph_construction as gc
import evaluation as ev
from datetime import datetime
import uuid
import itertools
from copy import deepcopy
import numpy as np
kmeans_grid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 10],
    "init":       ["k-means++", "random"],
}

knn_grid = {
    "n_neighbors": [2 ,3 ,4 ,5, 7, 10, 12, 14, 15],
    "metric":      ["cosine", "euclidean", "minkowski", "manhattan"]
}
threshold_grid  = {
    "threshold_distance ": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

}

dbscan_grid = {
    "eps":         [0.3, 0.5, 0.7],
    "min_samples": [3, 5, 10],
}

agglo_grid = {
    "linkage":            ["ward", "complete", "average"],
    "distance_threshold": [None],   # set to e.g. [0.5, 1.0] if using distance mode
    "n_clusters":         [3, 4],   # ignored when distance_threshold is not None
}

preprocess_combinations = [
    {"scaler": None, "pca": None},
    {"scaler": MinMaxScaler(),   "pca": None},
    {"scaler": StandardScaler(), "pca": None},
    {"scaler": None, "pca":  PCA()},
    {"scaler": StandardScaler(), "pca": PCA()},
    {"scaler": MinMaxScaler(),   "pca": PCA()},
]

# ── 2. Base param dicts (non-varied keys stay fixed) ─────────────────────────

BASE_KMEANS = {
    "init": "k-means++", "n_init": "auto", "max_iter": 300,
    "tol": 1e-4, "verbose": 0, "random_state": None,
    "copy_x": True, "algorithm": "lloyd", "pipeline_id": 0
}
BASE_KNN = {
    "radius": 1.0, "algorithm": "auto", "leaf_size": 30,
    "p": 2, "metric_params": None, "n_jobs": None,
}
BASE_DBSCAN = {
    "metric": "euclidean", "metric_params": None,
    "algorithm": "auto", "leaf_size": 30, "p": None, "n_jobs": None,
}
BASE_AGGLO = {
    "metric": "euclidean", "compute_full_tree": "auto", "connectivity": None,
}

result_file = "Outputs/"
BASE_THRESHOLD = {"threshold_distance": 0.5}

#pt.print_graph_stats(dt.load_graph("Graph_Μutual_KNN.gpickle"))



# Preparing dataset Part
#ex.prepare_all()
#ex.prepare_samples()



# Retrieve Data Part
#q, a, c = dt.load_sample("embeddings.pkl")
#q_ids = list(q.keys())[:20]
#ex.shortest_path_search(q_ids, dt.load_graph("20260519_094710_fdf704"), "KNN", 10, 0.5, result_file+"20260519_094710_fdf704", 3)
#ex.hits_search(q_ids, dt.load_graph("20260519_094710_fdf704"), "KNN", 10, 0.5, result_file+"20260519_094710_fdf704", 3)
#ex.baseline_search(q_ids, 10, result_file+"20260519_094710_fdf704")
#ex.k_steph_search(q_ids, dt.load_graph("20260519_094710_fdf704"), "KNN", 10, 2, 0.5, result_file+"20260519_094710_fdf704", 3)
#ex.personalised_pagerank_search(q_ids, dt.load_graph("20260519_094710_fdf704"), "Clustering + KNN" ,10, result_file+"20260519_094710_fdf704",3)

#ex.personalised_pagerank_search(q_ids, dt.load_graph("Graph_KNN.gpickle"),"KNN", 20, result_file + str(10)+".txt")
#ex.personalised_pagerank_search(q_ids, dt.load_graph("Graph_Μutual_KNN.gpickle"), "MKNN", 10, result_file + str(10)+".txt")
#ex.personalised_pagerank_search(q_ids, dt.load_graph("Graph_KNN_Kmeans.gpickle"), "Clustering + KNN" ,10, result_file + str(10)+".txt")
#ex.personalised_pagerank_search(q_ids, dt.load_graph("Graph_Mutual_KNN_Kmeans.gpickle"), "Clustering + MKNN", 10, result_file + str(10)+".txt")



# Graph Construction Part

#q, a, c = dt.load_sample("embeddings.pkl")
#ex.knn_Graph(c, knn_params, preprocess, "graph_knn_1", True)
#ex.mutual_knn(c, knn_params, preprocess, "Graph_Μutual_KNN.gpickle", True)
#ex.cluster_graph(c, kmeans_params,threshold_params, agglo_params, preprocess, knn_params, dbscan_params,
                 #"kmeans")

# Print Graph info Part
# pt.print_graph_stats(dt.load_graph("Graph_Μutual_KNN.gpickle"))
# pt.print_graph_stats(dt.load_graph("Graph_KNN.gpickle"))
# pt.print_graph_stats(dt.load_graph("Graph_Mutual_KNN_Kmeans.gpickle"))
# pt.print_graph_stats(dt.load_graph("Graph_Threshold.gpickle"))
def make_name(prefix, varied: dict) -> str:

    parts = []
    for k, v in varied.items():
        short_key = k.replace("n_clusters", "k")  \
                     .replace("n_neighbors", "n")  \
                     .replace("distance_threshold", "dt") \
                     .replace("min_samples", "ms") \
                     .replace("scaler", "sc")       \
                     .replace("pca", "pca")
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


def make_graphs(c, ex):
    results = []   # collect (name, function, params) for logging

    for pre in preprocess_combinations:
        preprocess = {"scaler": pre["scaler"], "pca": pre["pca"]}
        pre_tag = f"sc{pre['scaler']}_pca{pre['pca']}"

        # ── knn_Graph ────────────────────────────────────────────────────────
        for knn_combo in grid_combinations(knn_grid):
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"knn_{pre_tag}", knn_combo)
            print(f"[knn_Graph]   {name}")
            ex.knn_Graph(c, knn_params, preprocess, name, True)
            results.append(("knn_Graph", name, knn_combo, pre))

        # ── mutual_knn ───────────────────────────────────────────────────────
        for knn_combo in grid_combinations(knn_grid):
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"mutual_{pre_tag}", knn_combo)
            print(f"[mutual_knn]  {name}")
            ex.mutual_knn(c, knn_params, preprocess, name, True)
            results.append(("mutual_knn", name, knn_combo, pre))
        # ── kmeans clustering ───────────────────────────────────────────────────────
        for kmeans_combo in grid_combinations(kmeans_grid):
            kmeans_params = merge(BASE_KMEANS, kmeans_combo)
            name = make_name(f"kmeans_{pre_tag}", kmeans_combo)
            print(f"[kmeans]  {name}")
            clustering_result = clustring_result = ex.cluster_graph(sampled_items=c, kmeans_params=kmeans_params,agglo_params=agglo_grid,
                                                                knn_params={},dbscan_params=dbscan_grid,preprocess=preprocess,
                                                                name="",algorithm="kmeans")
            for knn_combo in grid_combinations(knn_grid):
                knn_params = merge(BASE_KNN, knn_combo)
                name_ = make_name(f"knn_{pre_tag}", knn_combo) + name
                print(f"[knn_clustering_Graph]   {name_}")
                ex.cluster_graph(sampled_items=c, kmeans_params=kmeans_params, agglo_params=agglo_grid,
                                 knn_params=knn_params, dbscan_params=dbscan_grid, preprocess=preprocess,
                                 name=name_, algorithm="kmeans",clustering_result=clustering_result, flag=False)


    print(f"\nDone — {len(results)} graphs generated.")
    return results
def run_retrieval():
    q, a, c = dt.load_sample("embeddings.pkl")
    q_ids = list(q.keys())[:20]
    files = dt.get_files()
    for file in files:
     print(f"for file : {file} evaluations")
     ex.baseline_search(q_ids, 10, result_file + file)
     for alpha in np.arange(0.0, 1.0, 0.1):
         for init in range(2, 10):
             ex.shortest_path_search(q_ids, dt.load_graph(file), "", 10, alpha, result_file+file, init)
             ex.hits_search(q_ids, dt.load_graph(file), "", 10, alpha, result_file+file, init)
             ex.k_steph_search(q_ids, dt.load_graph(file), "", 10, 2, alpha, result_file+file, init)
             ex.personalised_pagerank_search(q_ids, dt.load_graph(file), "" ,10, result_file+file,init)









q, a, c = dt.load_sample("embeddings.pkl")
make_graphs(c, ex)
#q_ids = list(q.keys())[:20]
#run_retrieval()

