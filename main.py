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
import clustering as cl
import graph_construction as gc
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import plotting as pt
import tables
import pandas as pd
knn_metrics = ['cosine', 'euclidean', 'manhattan', 'minkowski']
k_means_algorithms = ['k-means++', 'random']
rerankers = ['BM25', 'graph_aware', 'cross_encoder']
kmeans_grid = {
    "n_clusters": [5, 10, 20],
    "init":       ["k-means++"],
}

knn_grid = {
    "n_neighbors": [5, 10, 20],
    "metric":      ["cosine"]
}
threshold_grid = {
    "threshold_distance": [0.8, 0.6, 0.4]

}

dbscan_grid = {
    "eps":         [0.5],
    "min_samples": [10],
}

agglo_grid = {
    "linkage":            ["ward", "complete", "average"],
    "distance_threshold": [None],
    "n_clusters":         [3, 4],
}
graph_grid = {
     "Directed": False,
     "Weighted": True
}
#preprocess_combinations = [
#    {"scaler": None, "pca": None},
#    {"scaler": MinMaxScaler(),   "pca": None},
#    {"scaler": StandardScaler(), "pca": None},
#    {"scaler": None, "pca":  PCA()},
#    {"scaler": StandardScaler(), "pca": PCA()},
#    {"scaler": MinMaxScaler(),   "pca": PCA()},
#]
#graph_combinations = [
#    {"Directed": False, "Weighted": True},
#    {"Directed": True, "Weighted": True},
#    {"Directed": True, "Weighted": False},
#    {"Directed": False, "Weighted": False},

#]
graph_combinations = [{"Directed": False, "Weighted": True}]
preprocess_combinations = [{"scaler": None, "pca": None}]

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


def make_name(prefix, varied, graph, clustering = None) :
    parts = []
    if clustering is not None:
        for k, v in clustering.items():
            short_key = k.replace("n_clusters", "clusters_") \
                         .replace("eps", "eps_") \
                         .replace("min_samples", "ms")
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
    results = []   # collect (name, function, params) for logging
    count = 0
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
            gc.build_knn_graph(c, knn_params, preprocess, name, graph_params, True)
            count+=1

        # ── mutual_knn ───────────────────────────────────────────────────────
        for knn_combo in grid_combinations(knn_grid):
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"mutual_{pre_tag}", knn_combo, graph_params)
            print(f"[mutual_knn]  {name}")
            gc.build_mutual_knn_graph(c, knn_params, preprocess, name, graph_params, True)
            count+=1
        # ── threshold_graph ───────────────────────────────────────────────────────
        for threshold_combo in grid_combinations(threshold_grid):
            threshold_params = merge(BASE_THRESHOLD, threshold_combo)
            name = make_name(f"threshold_{pre_tag}", threshold_combo, graph_params)
            gc.build_threshold_graph(c, threshold_params, preprocess, name, graph_params, True)
            results.append(("threshold", name, threshold_combo, preprocess))
        # ── kmeans clustering ───────────────────────────────────────────────────────
        for kmeans_combo in grid_combinations(kmeans_grid):
            kmeans_params = merge(BASE_KMEANS, kmeans_combo)
            clustering_results = cl.perform_clustering(data=c, algorithm="kmeans", kmeans_params=kmeans_params,
                                                       agglo_params={}, dbscan_params={}, preprocess=preprocess)
            for knn_combo in grid_combinations(knn_grid):
                knn_params = merge(BASE_KNN, knn_combo)

                name = make_name(f"knn_kmeans_{pre_tag}", knn_combo, graph_params, kmeans_combo)
                gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False, "kmeans")
                gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name,False, "kmeans")
                count += 2
        for kmeans_combo in grid_combinations(kmeans_grid):
            kmeans_params = merge(BASE_KMEANS, kmeans_combo)
            clustering_results = cl.perform_clustering(data=c, algorithm="kmeans", kmeans_params=kmeans_params,
                                                       agglo_params={}, dbscan_params={}, preprocess=preprocess)
            for threshold_combo in grid_combinations(threshold_grid):
                threshold_params = merge(BASE_THRESHOLD, threshold_combo)
                name = make_name(f"threshold_kmeans_{pre_tag}", threshold_combo, graph_params, kmeans_combo)
                gc.build_clustering_threshold_graph(clustering_results, threshold_params, graph_params, preprocess, name, False, "kmeans")

        #for dbscan_combo in grid_combinations(dbscan_grid):
            #dbscan_params = merge(BASE_DBSCAN, dbscan_combo)
            #name = make_name(f"dbscan_{pre_tag}", dbscan_combo, graph_params)
            #clustering_results = cl.perform_clustering(data=c, algorithm="dbscan", kmeans_params={}, agglo_params={},
             #                                          dbscan_params=dbscan_params, preprocess=preprocess)
            #for knn_combo in grid_combinations(knn_grid):
                #knn_params = merge(BASE_KNN, knn_combo)
                #name = make_name(f"knn_dbscan_{pre_tag}", dbscan_combo, graph_params, dbscan_combo)
                #gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False)
                #gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name,
                 #                                    False)
                #count += 2
        #for dbscan_combo in grid_combinations(dbscan_grid):
            #dbscan_params = merge(BASE_DBSCAN, dbscan_combo)
            #clustering_results = cl.perform_clustering(data=c, algorithm="dbscan", kmeans_params={}, agglo_params={},
            #                                          dbscan_params=dbscan_params, preprocess=preprocess)
            #for threshold_combo in grid_combinations(threshold_grid):
                #threshold_params = merge(BASE_THRESHOLD, threshold_combo)
                #name = make_name(f"threshold_dbscan{pre_tag}", threshold_combo, graph_params, dbscan_combo)
                #gc.build_clustering_threshold_graph(clustering_results, threshold_params, graph_params, preprocess, name, False)

    print(f"\nDone — {count} graphs generated.")
    return results



def get_preprocess_graph_input():
    # Scaler
    SCALER = str(input("Scaler (None/MinMax/Standard): "))
    if SCALER == "None":
        scaler = None
    elif SCALER == "MinMax":
        scaler = MinMaxScaler()
    elif SCALER == "Standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Wrong Input Error: {SCALER}")

    # PCA
    PCA_INPUT = input("PCA n_components (None/PCA): ").strip()

    if PCA_INPUT == "None":
        pca = None
    elif PCA_INPUT == "PCA":
        pca = PCA()
    else:
        raise ValueError(f"Wrong Input Error: {PCA_INPUT}")

    preprocess = {"scaler": scaler, "pca": pca}
    DIRECTED = input('Directed graph? (y/n): ').lower().startswith('y')
    WEIGHTED = input('Weighted graph?  (y/n): ').lower().startswith('y')
    graph_params = {"Directed": DIRECTED, "Weighted": WEIGHTED}
    return preprocess, graph_params



def get_knn_input():
    N_NEIGHBORS = int(input("n_neighbors (int): ").strip())
    METRIC = input("Metric (cosine/euclidean/manhattan/minkowski): ").strip()

    if N_NEIGHBORS <= 0:
        raise ValueError(f"Wrong Input Error: {N_NEIGHBORS}")
    if METRIC not in knn_metrics:
        raise ValueError(f"Wrong Input Error: {METRIC}")

    knn_combo = {"n_neighbors": N_NEIGHBORS, "metric": METRIC}
    return knn_combo
def get_dbscan_input():
    EPS = float(input("eps (float): ").strip())
    MIN_SAMPLES = int(input("min_samples (int): ").strip())
    if EPS <= 0:
        raise ValueError(f"Negative Value Error on EPS: {EPS}")
    if MIN_SAMPLES <= 0:
        raise ValueError(f"Negative Value Error on MIN_SAMPLES: {MIN_SAMPLES}")
    dbscan_combo = {"eps": EPS, "min_samples": MIN_SAMPLES}
    return dbscan_combo
def get_kmeans_input():
    N_CLUSTERS = int(input("n_clusters (int): ").strip())
    INIT = input("Algorithm (k-means++/random: ").strip()
    if N_CLUSTERS <= 0:
        raise ValueError(f"Negative Value Error On Clusters: {N_CLUSTERS}")
    if INIT not in k_means_algorithms:
        raise ValueError(f"Algorithm Not Found Error: {INIT}")

    kmeans_combo = {"n_clusters": N_CLUSTERS, "init": INIT}
    return kmeans_combo


def make_dev_query_split():
    q, a, c = dt.load_texts()
    q_emb, _, _ = dt.load_data()
    vectorizer = TfidfVectorizer()
    analyzer = vectorizer.build_analyzer()
    query_labels = {}
    lengths = [len(analyzer(text)) for text in q.values()]
    p33, p66 = np.percentile(lengths, [33, 66])
    for qid, text in q.items():
        n = len(analyzer(text))
        query = q_emb[qid]
        query_correct_results = query[1]

        if n <= p33:
            label = "small"
        elif n <= p66:
            label = "medium"
        else:
            label = "long"

        query_labels[qid] = label

    print(len(query_labels.keys()))
    total_sample = int(len(query_labels) * 0.04)
    each_label_size = total_sample // 3
    print(each_label_size)
    small_queries = []
    medium_queries = []
    long_queries = []
    rng = np.random.default_rng(42)

    small_pool = [qid for qid, l in query_labels.items() if l == "small"]
    medium_pool = [qid for qid, l in query_labels.items() if l == "medium"]
    long_pool = [qid for qid, l in query_labels.items() if l == "long"]
    small_queries = rng.choice(
        small_pool,
        size=min(each_label_size, len(small_pool)),
        replace=False
    ).tolist()

    medium_queries = rng.choice(
        medium_pool,
        size=min(each_label_size, len(medium_pool)),
        replace=False
    ).tolist()

    long_queries = rng.choice(
        long_pool,
        size=min(each_label_size, len(long_pool)),
        replace=False
    ).tolist()

    print(small_queries)
    #print(len(medium_queries))
    #print(len(long_queries))
    #dt.sanity_check(small_queries + medium_queries + long_queries)
    dt.save_split(small_queries, medium_queries, long_queries, f"Data/splits", "dev")

def make_test_split():
    q, a = dt.load_texts_tests()
    q_emb, _ = dt.load_data_tests()
    vectorizer = TfidfVectorizer()
    analyzer = vectorizer.build_analyzer()
    query_labels = {}
    lengths = [len(analyzer(text)) for text in q.values()]
    p33, p66 = np.percentile(lengths, [33, 66])
    for qid, text in q.items():
        n = len(analyzer(text))
        query = q_emb[qid]
        if n <= p33:
            label = "small"
        elif n <= p66:
            label = "medium"
        else:
            label = "long"

        query_labels[qid] = label

    print(len(query_labels.keys()))
    small_queries = []
    medium_queries = []
    long_queries = []
    rng = np.random.default_rng(42)

    small_pool = [qid for qid, l in query_labels.items() if l == "small"]
    medium_pool = [qid for qid, l in query_labels.items() if l == "medium"]
    long_pool = [qid for qid, l in query_labels.items() if l == "long"]
    #print(small_pool)
    #print(medium_pool)
    #print(long_pool)
    #dt.sanity_check(small_pool + medium_pool + long_pool)
    dt.save_split(small_pool, medium_pool, long_pool, f"Data/splits", "test")


if __name__ == "__main__":

    MODE = str(input("Choose Mode, Make graphs, Make a test graph, Run experiments, Make embeddings Graph performance "
                     "make query samples: "))


    #MODE = ""
    if MODE.lower() == "make graphs":
        q, a, c = dt.load_data()
        make_graphs(c)
    elif MODE.lower() == "make embeddings":
        ex.prepare_dataset()
    elif MODE.lower() == "run experiments":
        METHOD = input("Examples: ").strip()
        if METHOD.lower() == "examples":
            ex.run_an_baseline()
            ex.run_an_example_retrieval_ppr()
            ex.run_an_example_retrieval_k_steph()
            ex.ppr_score_calibration_study()
        ex.run_retrieval_ppr_deep_sensitivity_experiment()
    elif MODE.lower() == "make query samples":
        make_dev_query_split()
    elif MODE.lower() == "graph performance":
        GRAPH_NAME = input("Graph name: ").strip()
        dt.graph_perf(GRAPH_NAME)
    elif MODE.lower() == "make a test graph":
        q, a, c = dt.load_data()
        preprocess, graph_params = get_preprocess_graph_input()
        pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
        METHOD = input("Choose method, Mutual knn, Knn, Clustering, threshold: ").strip()
        if METHOD.lower() == "mutual knn":
            knn_combo = get_knn_input()
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"mutual_knn_{pre_tag}", knn_combo, graph_params)
            gc.build_mutual_knn_graph(c, knn_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "knn":
            knn_combo = get_knn_input()
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"knn_{pre_tag}", knn_combo, graph_params)
            print(name)
            gc.build_knn_graph(c, knn_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "threshold":
            threshold = float(input("threshold distance (float): ").strip())
            threshold_params = {"threshold_distance": threshold}
            name = make_name(f"threshold_{pre_tag}", threshold_params, graph_params)
            gc.build_threshold_graph(c, threshold_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "clustering":
            GRAPH_ALGO = input("Choose algorithm, Kmeans/Dbscan: ").strip()
            if GRAPH_ALGO.lower() == "kmeans":
                kmeans_combo = get_kmeans_input()
                kmeans_params = merge(BASE_KMEANS, kmeans_combo)

                knn_combo = get_knn_input()
                knn_params = merge(BASE_KNN, knn_combo)
                name = make_name(f"knn_kmeans_{pre_tag}", knn_combo, graph_params, kmeans_combo)

                clustering_results = cl.perform_clustering(data=c, algorithm="kmeans", kmeans_params=kmeans_params, agglo_params={}, dbscan_params={},  preprocess= preprocess)
                gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False, "kmeans")
                gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False, "kmeans")
            elif GRAPH_ALGO.lower() == "dbscan":
                dbscan_combo = get_dbscan_input()
                dbscan_params = merge(BASE_DBSCAN, dbscan_combo)
                knn_combo = get_knn_input()
                knn_params = merge(BASE_KNN, knn_combo)

                pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
                name = make_name(f"knn_dbscan_{pre_tag}", knn_combo, graph_params, dbscan_combo)
                clustering_results = cl.perform_clustering(data=c, algorithm="dbscan", kmeans_params={}, agglo_params={}, dbscan_params=dbscan_params,  preprocess= preprocess)
                gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False, "dbscan")
                gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name,
                                                  False, "dbscan")
            else:
                raise ValueError(f"Wrong graph construction method: {GRAPH_ALGO}")
    else:
        raise ValueError(f"Wrong Input Error: {MODE}")
