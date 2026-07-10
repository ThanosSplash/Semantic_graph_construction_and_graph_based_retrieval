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
    "threshold_distance": [0.3, 0,4, 0.5, 0.6]

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
        pre_tag = f"sc{pre['scaler']}_pca{pre['pca']}"


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
        #for threshold_combo in grid_combinations(threshold_grid):
            #threshold_params = merge(BASE_THRESHOLD, threshold_combo)
            #name = make_name(f"threshold_{pre_tag}", threshold_combo, graph_params)
            #gc.build_threshold_graph(c, threshold_params, preprocess, name, graph_params, True)
            #results.append(("threshold", name, threshold_combo, preprocess))
        # ── kmeans clustering ───────────────────────────────────────────────────────
        #for kmeans_combo in grid_combinations(kmeans_grid):
            #kmeans_params = merge(BASE_KMEANS, kmeans_combo)
            #clustering_results = cl.perform_clustering(data=c, algorithm="kmeans", kmeans_params=kmeans_params,
                 #                                      agglo_params={}, dbscan_params={}, preprocess=preprocess)
            #for knn_combo in grid_combinations(knn_grid):
                #knn_params = merge(BASE_KNN, knn_combo)

                #name = make_name(f"knn_kmeans_{pre_tag}", knn_combo, graph_params, kmeans_combo)
                #gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False)
                #gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name,
                  #                                   False)
                #count += 2
        #for kmeans_combo in grid_combinations(kmeans_grid):
            #kmeans_params = merge(BASE_KMEANS, kmeans_combo)
            #clustering_results = cl.perform_clustering(data=c, algorithm="kmeans", kmeans_params=kmeans_params,
             #                                          agglo_params={}, dbscan_params={}, preprocess=preprocess)
            #for threshold_combo in grid_combinations(threshold_grid):
                #threshold_params = merge(BASE_THRESHOLD, threshold_combo)
                #name = make_name(f"threshold_kmeans_{pre_tag}", threshold_combo, graph_params, kmeans_combo)
                #gc.build_clustering_threshold_graph(clustering_results, threshold_params, graph_params, preprocess, name, False)

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
def get_kmeans_input():
    N_CLUSTERS = int(input("n_clusters (int): ").strip())
    INIT = input("Algorithm (k-means++/random: ").strip()
    if N_CLUSTERS <= 0:
        raise ValueError(f"Negative Value Error On Clusters: {N_CLUSTERS}")
    if INIT not in k_means_algorithms:
        raise ValueError(f"Algorithm Not Found Error: {INIT}")

    kmeans_combo = {"n_clusters": N_CLUSTERS, "init": INIT}
    return kmeans_combo

def get_dbscan_input():
    EPS = float(input("eps (float): ").strip())
    MIN_SAMPLES = int(input("min_samples (int): ").strip())
    if EPS <= 0:
        raise ValueError(f"Negative Value Error on EPS: {EPS}")
    if MIN_SAMPLES <= 0:
        raise ValueError(f"Negative Value Error on MIN_SAMPLES: {MIN_SAMPLES}")
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
    print(f"Mine ndcg@k {ev.ndcg_score_(predictions = predictions, ground_truth=ground_truth, k=4)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=4)}")
    print("-----------------------------------------------------")
    predictions = ["C", "B", "X", "D"]
    ground_truth = ["C", "B", "X"]
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"Mine ndcg@k {ev.ndcg_score_(predictions=predictions, ground_truth=ground_truth, k=4)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=4)}")
    print("-----------------------------------------------------")

    predictions = ["A", "E", "F"]
    ground_truth = ["A", "B", "X"]
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=3)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=3)}")
    print(f"Mine ndcg@k {ev.ndcg_score_(predictions=predictions, ground_truth=ground_truth, k=3)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=3)}")
    print("-----------------------------------------------------")

    predictions = ["E", "F", "A"]
    ground_truth = ["A", "B", "X"]
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=3)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=3)}")
    print(f"Mine ndcg@k {ev.ndcg_score_(predictions=predictions, ground_truth=ground_truth, k=3)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=3)}")
    print("-----------------------------------------------------")


    predictions = ["A", "I", "E", "D"]
    ground_truth = ["C", "B", "X"]
    print(f"recall@k {ev.recallk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"Mrr {ev.RR_score(predictions=predictions, correct_results=ground_truth)}")
    print(f"ndcg@k {ev.nDCGk_score(predictions=predictions, correct_results=ground_truth, k=4)}")
    print(f"Mine ndcg@k {ev.ndcg_score_(predictions=predictions, ground_truth=ground_truth, k=4)}")
    print(f"map@k {ev.avg_precision(predictions=predictions, correct_results=ground_truth, k=4)}")
    print("-----------------------------------------------------")


if __name__ == "__main__":



    MODE = str(input("Choose Mode, Make graphs, Run eval tests, Make a test graph, Run a test retrieval, "
                     "Run retrieval, Make embeddings, Make Global leaderboard, Graph performance, plot graph stats: "))


    #MODE = ""
    if MODE.lower() == "make graphs":
        q, a, c = dt.load_data()
        make_graphs(c)
    elif MODE.lower() == "make embeddings":
        ex.prepare_dataset()
    elif MODE.lower() == "run retrieval":
        METHOD = input("Choose method, Baseline, PPR, K steph, Hits, Shortest Path: ").strip()
        if METHOD.lower() == "ppr":
            ex.run_retrieval_ppr()
        elif METHOD.lower() == "k steph":
            ex.run_retrieval_k_steph()
    elif MODE.lower() == "run eval tests":
        run_eval_tests()
    elif MODE.lower() == "make global leaderboard":
        dt.make_global_leaderboard()
    elif MODE.lower() == "make query samples":
        make_query_samples()
    elif MODE.lower() == "graph performance":
        GRAPH_NAME = input("Graph name: ").strip()
        dt.graph_perf(GRAPH_NAME)
    elif MODE.lower() == "plot graph stats":
        pt.getting_plot_graph_stats()
    elif MODE.lower() == "run a test retrieval":
        METHOD = input("Choose method, Baseline, PPR, K steph, Hits, Shortest Path: ").strip()
        small, medium, long = dt.load_samples()
        files = dt.get_files()
        file = files[2]
        all_samples = small + medium + long
        metrics = ["recall", "rr", "ndcg", "avg_precisions"]
        k = int(input("k (int): ").strip())
        print(file)
        eval_scores = None
        params = None
        start_time = time.perf_counter()
        if METHOD.lower() == "baseline":
            for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                scores, p = ex.baseline_search(dataset, k, "Outputs/Global leaderboard", name)
                if eval_scores is None:
                    eval_scores, params = scores, p
                else:
                    for m in metrics:
                        eval_scores[m] += scores[m]
            params["sample_type"] = "all_samples"
            ex.evaluate_method("Baseline", eval_scores, "Outputs/Global leaderboard", params)

        elif METHOD.lower() == "ppr":
            graph = dt.load_graph(file)
            init = int(input("init (int): ").strip())
            if init <= 0 :
                raise ValueError(f"Negative Value Error: {init}")
            alpha = float(input("alpha (float): ").strip())
            if alpha < 0.0 or alpha > 1.0 :
                raise ValueError(f"Οut Οf Βounds Error: {alpha}")
            for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                scores, p = ex.personalised_pagerank_search(dataset, graph, k, f"Outputs/graphs/{file}", init, name, alpha, file)
                if eval_scores is None:
                    eval_scores, params = scores, p
                else:
                    for m in metrics:
                        eval_scores[m] += scores[m]
            params["sample_type"] = "all_samples"
            ex.evaluate_method("PPR", eval_scores, f"Outputs/graphs/{file}", params)
        elif METHOD.lower() == "k steph":
            graph = dt.load_graph(file)
            init = int(input("init (int): ").strip())
            if init <= 0:
                raise ValueError(f"Negative Value Error: {init}")
            alpha = float(input("alpha (float): ").strip())
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"Οut Οf Βounds Error: {alpha}")
            k_step = int(input("k_steph (int): ").strip())
            if k_step <= 0:
                raise ValueError(f"Negative Value Error: {k_step}")
            RERANKER = str(input("Choose reranker, BM25, graph_aware, cross_encoder: "))
            if RERANKER not in rerankers:
                raise ValueError(f"Wrong reranker input {RERANKER}")
            for name, dataset in [("small", small), ("medium", medium), ("long", long)]:

                scores, p = ex.k_steph_search(dataset, graph, RERANKER, k, k_step, alpha, f"Outputs/graphs/{file}", init, name)
                if eval_scores is None:
                    eval_scores, params = scores, p
                else:
                    for m in metrics:
                        eval_scores[m] += scores[m]
            params["sample_type"] = "all_samples"
            ex.evaluate_method("k-steph", eval_scores, f"Outputs/graphs/{file}", params)

            #ex.k_steph_search(small, graph, RERANKER, k, k_step, alpha,f"Outputs/graphs/{file}", init, "small")
            #ex.k_steph_search(medium, graph, RERANKER, k, k_step, alpha, f"Outputs/graphs/{file}", init, "medium")
            #ex.k_steph_search(long, graph, RERANKER, k, k_step, alpha, f"Outputs/graphs/{file}", init, "long")
            #ex.k_steph_search(all_samples, graph, RERANKER, k, k_step, alpha, f"Outputs/graphs/{file}", init, "all_samples")
        elif METHOD.lower() == "hits":
            ex.hits_search()
        elif METHOD.lower() == "shortest path":
            graph = dt.load_graph(file)
            init = int(input("init (int): ").strip())
            if init <= 0:
                raise ValueError(f"Negative Value Error: {init}")
            alpha = float(input("alpha (float): ").strip())
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"Οut Οf Βounds Error: {alpha}")
            RERANKER = str(input("Choose reranker, BM25, graph_aware, cross_encoder: "))
            if RERANKER not in rerankers:
                raise ValueError(f"Wrong reranker input {RERANKER}")
            ex.shortest_path_search(small, graph, RERANKER, k, alpha, f"Outputs/graphs/{file}", init, "small")
            ex.shortest_path_search(medium, graph, RERANKER, k, alpha, f"Outputs/graphs/{file}", init, "medium")
            ex.shortest_path_search(long, graph, RERANKER, k, alpha, f"Outputs/graphs/{file}", init, "long")
            #ex.shortest_path_search(all_samples, graph, RERANKER, k, alpha, f"Outputs/graphs/{file}", init, "all_samples")
        else:
            raise ValueError(f"Wrong Retrieval Method: {METHOD}")
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Time: {execution_time:.4f} seconds")
        dt.save_leaderboard(f"Outputs/graphs/{file}", "leaderboards")
    elif MODE.lower() == "make a test graph":
        q, a, c = dt.load_data()
        preprocess, graph_params = get_preprocess_graph_input()
        METHOD = input("Choose method, Mutual knn, Knn, Clustering, threshold: ").strip()
        if METHOD.lower() == "mutual knn":
            knn_combo = get_knn_input()
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"mutual_knn_", knn_combo, graph_params)
            ex.mutual_knn(c, knn_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "knn":
            knn_combo = get_knn_input()
            knn_params = merge(BASE_KNN, knn_combo)
            name = make_name(f"knn_", knn_combo, graph_params)
            ex.knn_Graph(c, knn_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "threshold":
            threshold = float(input("threshold distance (float): ").strip())
            threshold_params = {"threshold_distance": threshold}
            name = make_name(f"threshold_", threshold_params, graph_params)
            gc.build_threshold_graph(c, threshold_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "clustering":
            GRAPH_ALGO = input("Choose algorithm, Kmeans/Dbscan: ").strip()
            if GRAPH_ALGO.lower() == "kmeans":
                kmeans_combo = get_kmeans_input()
                kmeans_params = merge(BASE_KMEANS, kmeans_combo)

                knn_combo = get_knn_input()
                knn_params = merge(BASE_KNN, knn_combo)
                pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
                name = make_name(f"knn_kmeans_{pre_tag}", knn_combo, graph_params, kmeans_combo)

                clustering_results = cl.perform_clustering(data=c, algorithm="kmeans", kmeans_params=kmeans_params, agglo_params={}, dbscan_params={},  preprocess= preprocess)
                gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False)
                gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False)
            elif GRAPH_ALGO.lower() == "dbscan":
                dbscan_combo = get_dbscan_input()
                dbscan_params = merge(BASE_DBSCAN, dbscan_combo)
                knn_combo = get_knn_input()
                knn_params = merge(BASE_KNN, knn_combo)

                pre_tag = f"sc{preprocess['scaler']}_pca{preprocess['pca']}"
                name = make_name(f"knn_dbscan_{pre_tag}", knn_combo, graph_params, dbscan_combo)
                clustering_results = cl.perform_clustering(data=c, algorithm="dbscan", kmeans_params={}, agglo_params={}, dbscan_params=dbscan_params,  preprocess= preprocess)
                gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False)
                gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name,
                                                  False)
            else:
                raise ValueError(f"Wrong graph construction method: {GRAPH_ALGO}")
    else:
        raise ValueError(f"Wrong Input Error: {MODE}")
