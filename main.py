import plotting

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
import sknetwork
knn_metrics = ['cosine', 'euclidean', 'manhattan', 'chebyshev']
k_means_algorithms = ['k-means++', 'random']
rerankers = ['BM25', 'graph_aware', 'cross_encoder']

result_file = "Outputs/"
FILE_PPR = "2026-09-02_deep sensitivity experiment_38ffe2c1"
FILE_K_STEPH = "2026-09-15_deep sensitivity experiment k steph_640cf4a9"
FILE_K_STEPH_BM25 = "2026-09-16_reranker experiment on k steph for BM25_bd31ed6e"
FILE_SEED_SELECTION = "2026-08-27_ppr_seed_selection_study_12764e8e"
FILE_NORM_SELECTION = "2026-08-26_ppr_score_calibration_study_e13e6e09"
FILE_RANDOM_STATE_SELECTION = "2026-08-29_kmeans_random_state_selection_study_41fd8da1"
FILE_PPR_GRAPH_CONSTRUCTION = "2026-09-13_deep sensitivity experiment for different params in the contruction of the graph graph_construction_9c229bdb"
FILE_PPR_PREPROCESS = "2026-09-13_deep sensitivity experiment for different params in the contruction of the graph preprocess_1959e7da"
FILE_PPR_GRAPH_METRIC = "2026-09-12_deep sensitivity experiment for different params in the contruction of the graph metric_f452199c"
FILE_PPR_AGGLO = "2026-09-11_deep sensitivity experiment for agglo kmeans_ac4b41f4"
FILE_PPR_TEST = "2026-09-18_Test set for ppr_024c2f6e"
FILE_K_STEPH_TEST = "2026-09-18_test set for k-steph_652346d3"


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
    METRIC = input("Metric (cosine/euclidean/manhattan/chebyshev): ").strip()

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




if __name__ == "__main__":

    MODE = str(input("Choose Mode, Make graphs, Make a test graph, Run experiments, Make embeddings Graph performance "
                     "make query samples: "))


    #MODE = ""
    if MODE.lower() == "make graphs":
        METHOD = input("All types, kmeans knn, mutual kmeans knn, threshold kmeans knn: ").strip()
        _, _, c = dt.load_data()
        preprocess = {"scaler": None, "pca": None}
        graph_params = {"Directed": False, "Weighted": True}
        if METHOD.lower() == "all types":
            gc.make_graphs(c)
        elif METHOD.lower() == "kmeans knn":
            gc.make_kmeans_knn_graph_for_all_params(c, preprocess, graph_params, False, True)
        elif METHOD.lower() == "mutual kmeans knn":
            gc.make_kmeans_mutual_knn_graph_for_all_params(c, preprocess, graph_params, False, True)
        elif METHOD.lower() == "threshold kmeans knn":
            gc.make_kmeans_threshold_graph_for_all_params(c, preprocess, graph_params, False, True)
    elif MODE.lower() == "make embeddings":
        ex.prepare_dataset()
    elif MODE.lower() == "run experiments":
        METHOD = input("baseline_dev, baseline_test,"
                       "\n prr calibration study, ppr seed selection study, k steph exhaustive,\n"
                       " ppr exhaustive, ablation study preprocess, ablation study metric,\n "
                       "ablation study graph construction, random state selection study,\n"
                       "reranker study bm25,\n"
                       " test set ppr, test set k-steph: ").strip()
        if METHOD.lower() == "baseline_dev":
            ex.run_baseline("dev")
        elif METHOD.lower() == "baseline_test":
            ex.run_baseline("test")
        elif METHOD.lower() == "prr calibration study":
            ex.ppr_score_calibration_study()
        elif METHOD.lower() == "ppr seed selection study":
            ex.ppr_seed_selection_study()
        elif METHOD.lower() == "ppr exhaustive":
            ex.run_retrieval_ppr_deep_sensitivity_experiment()
        elif METHOD.lower() == "k steph exhaustive":
            if FILE_PPR == "":
                raise ValueError(f"Wrong ppr experiment {FILE_PPR}")
            top_2 = dt.get_top_2_graph_families("PPR", "all_samples", FILE_PPR, "dev")
            ex.run_retrieval_k_steph("graph_aware", top_2)
        elif METHOD.lower() == "random state selection study":
            ex.k_means_random_state_study()
        elif METHOD.lower() == "ablation study preprocess":
            ex.ablation_study("preprocess")
        elif METHOD.lower() == "ablation study metric":
            ex.ablation_study("metric")
        elif METHOD.lower() == "ablation study graph construction":
            ex.ablation_study("graph_construction")
        elif METHOD.lower() == "reranker study bm25":
            top_3_graphs = dt.get_top_3_best_performing_graphs("k-steph", "all_samples", FILE_K_STEPH, "dev")
            top_3_graph_names = [graph["graph"] for graph in top_3_graphs]
            ex.reranker_study("BM25", top_3_graph_names)
        elif METHOD.lower() == "test set ppr":
            ex.run_ppr_on_test_set()
        elif METHOD.lower() == "test set k-steph":
            ex.run_k_steph_on_test_set()
    elif MODE.lower() == "make query samples":
        dt.make_dev_test_splits()
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
            knn_params = gc.merge(gc.BASE_KNN, knn_combo)
            name = make_name(f"mutual_knn_{pre_tag}", knn_combo, graph_params)
            gc.build_mutual_knn_graph(c, knn_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "knn":
            knn_combo = get_knn_input()
            knn_params = gc.merge(gc.BASE_KNN, knn_combo)
            name = make_name(f"knn_{pre_tag}", knn_combo, graph_params)
            print(name)
            gc.build_knn_graph(c, knn_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "threshold":
            threshold = float(input("threshold distance (float): ").strip())
            threshold_params = {"threshold_distance": threshold}
            name = make_name(f"threshold_{pre_tag}", threshold_params, graph_params)
            gc.build_threshold_graph(c, threshold_params, preprocess, name, graph_params, True)
        elif METHOD.lower() == "clustering":
            GRAPH_ALGO = input("Choose algorithm, Kmeans: ").strip()
            if GRAPH_ALGO.lower() == "kmeans":
                kmeans_combo = get_kmeans_input()
                kmeans_params = gc.merge(gc.BASE_KMEANS, kmeans_combo)

                knn_combo = get_knn_input()
                knn_params = gc.merge(gc.BASE_KNN, knn_combo)
                name = make_name(f"knn_kmeans_{pre_tag}", knn_combo, graph_params, kmeans_combo)

                clustering_results = cl.perform_clustering(data=c, algorithm="kmeans", kmeans_params=kmeans_params, agglo_params={},  preprocess=preprocess)
                gc.build_clustering_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False, "kmeans", True)
                #gc.build_clustering_mutual_knn_graph(clustering_results, knn_params, graph_params, preprocess, name, False, "kmeans",True)

            else:
                raise ValueError(f"Wrong graph construction method: {GRAPH_ALGO}")
    else:
        raise ValueError(f"Wrong Input Error: {MODE}")
