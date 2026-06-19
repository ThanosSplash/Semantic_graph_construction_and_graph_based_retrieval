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
import pandas as pd

kmeans_grid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 10],
    "init":       ["k-means++", "random"],
}

knn_grid = {
    "n_neighbors": [2 ,3 ,4 ,5, 7, 10, 12, 14, 15],
    "metric":      ["cosine", "euclidean", "minkowski", "manhattan"]
}
threshold_grid = {
    "threshold_distance ": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

}

dbscan_grid = {
    "eps":         [0.3, 0.5, 0.7],
    "min_samples": [3, 5, 10],
}

agglo_grid = {
    "linkage":            ["ward", "complete", "average"],
    "distance_threshold": [None],
    "n_clusters":         [3, 4],
}
graph_grid = {
     "Directed": True,
     "Weighted": True
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


def make_name(prefix, varied: dict, graph: dict) -> str:

    parts = []
    for k, v in graph.items():
        short_key = k.replace("Directed", "k")  \
                     .replace("Weighted", "n")
        parts.append(f"{short_key}{v}")
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



def get_preprocess_graph_input():
    # Scaler
    SCALER = str(input("Scaler (None/MinMax/Standard): "))
    if SCALER == "None":
        scaler = None
    elif SCALER == "MinMax":
        scaler = MinMaxScaler()
    elif SCALER == "Standard":
        scaler = StandardScaler()

    # PCA
    PCA_INPUT = input("PCA n_components (None/PCA): ").strip()

    if PCA_INPUT == "None":
        pca = None
    else:
        pca = PCA()

    preprocess = {"scaler": scaler, "pca": pca}
    DIRECTED = str(input("Directed graph? (True/False): "))
    WEIGHTED = str(input("Weighted graph? (True/False): "))
    graph_params = {"Directed": DIRECTED, "Weighted": WEIGHTED}
    return preprocess, graph_params



def get_knn_input():
    N_NEIGHBORS = int(input("n_neighbors (int): ").strip())
    METRIC = input("Metric (cosine/euclidean/manhattan/minkowski): ").strip()
    knn_combo = {"n_neighbors": N_NEIGHBORS, "metric": METRIC}
    return knn_combo
def get_kmeans_input():
    N_CLUSTERS = int(input("n_clusters (int): ").strip())
    INIT = input("Init (k-means++/random: ").strip()
    kmeans_combo = {"n_clusters": N_CLUSTERS, "init": INIT}
    return kmeans_combo

def get_dbscan_input():
    EPS = float(input("eps (float): ").strip())
    MIN_SAMPLES = int(input("min_samples (int): ").strip())
    dbscan_combo = {"eps": EPS, "min_samples": MIN_SAMPLES}
    return dbscan_combo

def make_query_samples():
    q, a, c = dt.load_texts()
    vectorizer = TfidfVectorizer()
    analyzer = vectorizer.build_analyzer()
    query_labels = {}
    lengths = [len(analyzer(text)) for text in q.values()]
    p33, p66 = np.percentile(lengths, [33, 66])
    for qid, text in q.items():
        n = len(analyzer(text))

        if n <= p33:
            label = "small"
        elif n <= p66:
            label = "medium"
        else:
            label = "long"

        query_labels[qid] = label

    print(len(query_labels.keys()))
    total_sample = int(len(query_labels) * 0.1)
    each_label_size = total_sample // 3
    print(each_label_size)
    small_queries = []
    medium_queries = []
    long_queries = []
    for qid, label in query_labels.items():
        if label == "small" and len(small_queries) < each_label_size:
            small_queries.append(qid)
        elif label == "medium" and len(medium_queries) < each_label_size:
            medium_queries.append(qid)
        elif label == "long" and len(long_queries) < each_label_size:
            long_queries.append(qid)
    print(len(small_queries))
    print(len(medium_queries))
    print(len(long_queries))
    dt.save_samples(small_queries, medium_queries, long_queries)

def run_eval_tests():

    predictions = ["A", "B", "C", "D"]
    ground_truth = ["C", "X"]

    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=2)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=4)}")
    print("-----------------------------------------------------")

    predictions = ["C", "B", "X", "D"]
    ground_truth = ["C", "B", "X"]
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=4)}")
    print("-----------------------------------------------------")

    predictions = ["A", "E", "F"]
    ground_truth = ["A", "B", "X"]
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=3)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=3)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=3)}")
    print("-----------------------------------------------------")

    predictions = ["E", "F", "A"]
    ground_truth = ["A", "B", "X"]
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=3)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=3)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=3)}")
    print("-----------------------------------------------------")


    predictions = ["A", "I", "E", "D"]
    ground_truth = ["C", "B", "X"]
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=4)}")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    MODE = str(input("Choose Mode, Make a test graph/Run retrieval/Make embeddings/Make leaderboard: "))




    #MODE = ""
    if MODE == "Make graphs":
        q, a, c = dt.load_data()
        make_graphs(c, ex)
    elif MODE == "Make embeddings":
        ex.prepare_dataset()
    elif MODE == "Run retrieval":
        ex.run_retrieval()
    elif MODE == "Run eval tests":
        run_eval_tests()
    elif MODE == "Make Global leaderboard":
        dt.save_leaderboard(
            "Outputs\knn_example_kFalse_nTrue_n5_metriccosine",
            "leaderboards"
        )
    elif MODE == "Make query samples":
        make_query_samples()
    elif MODE == "Make a test graph":
        q, a, c = dt.load_data()
        preprocess, graph_params = get_preprocess_graph_input()
        METHOD = str(input("Choose method, Mutual knn, Knn, Clustering, threshold: "))
        if METHOD == "Mutual knn":
            knn_combo = get_knn_input()
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"mutual_knn_", knn_combo, graph_params)
            ex.mutual_knn(c, knn_params, preprocess, name, graph_params, True)
        elif METHOD == "Knn":
            knn_combo = get_knn_input()
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"knn_", knn_combo, graph_params)
            ex.knn_Graph(c, knn_params, preprocess, name, graph_params, True)
        elif METHOD == "Threshold":
            threshold = float(input("threshold distance (float): ").strip())
            threshold_params = {"threshold_distance": threshold}
            name = make_name(f"threshold_", threshold_params, graph_params)
            gc.build_threshold_graph(c, threshold_params, preprocess, name, graph_params, False)
        elif METHOD == "Clustering":
            GRAPH_ALGO = str(input("Choose algorithm, Kmeans/Dbscan: "))
            if GRAPH_ALGO == "Kmeans":
                kmeans_combo = get_kmeans_input()
                kmeans_params = merge(BASE_KMEANS, kmeans_combo)

                knn_combo = get_knn_input()
                knn_params = merge(BASE_KNN, knn_combo)


                name = make_name(f"kmeans_", kmeans_combo, graph_params)
                name_ = make_name(f"knn_", knn_combo, graph_params) + name

                clustering_results = cl.perform_clustering(data=c, algorithm="kmeans", kmeans_params=kmeans_params, agglo_params={}, dbscan_params={},  preprocess= preprocess)
                gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name_, False)
                gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name_, False)
            elif GRAPH_ALGO == "Dbscan":
                dbscan_combo = get_dbscan_input()
                dbscan_params = merge(BASE_DBSCAN, dbscan_combo)
                knn_combo = get_knn_input()
                knn_params = merge(BASE_KNN, knn_combo)

                name = make_name(f"dbscan_", dbscan_combo, graph_params)
                name_ = make_name(f"knn_", dbscan_combo, graph_params) + name
                clustering_results = cl.perform_clustering(data=c, algorithm="dbscan", kmeans_params={}, agglo_params={}, dbscan_params=dbscan_params,  preprocess= preprocess)
                gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name_, False)
                gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name_,
                                                  False)

