import numpy as np

import embeddings as emb
import data_loading as dt
import graph_construction as gc
import retrieval as rt
from sklearn.cluster import AgglomerativeClustering, KMeans
import random
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue
from sklearn.metrics import precision_score, recall_score
import evaluation as ev
import clustering
import pandas as pd
import itertools
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import time
from tqdm import tqdm
from scipy.sparse import csr_matrix
import tables
import plotting as pt
import clustering as cl
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
seed_selection_method = "cosine"
normalization = "Min_Max"
random_state = 42

best_ppr = {
    "graph": "knn_scNone_pcaNone_Directed_False_Weighted_True_neighbors_10_metriccosine",
    "k": 20,
    "alpha": 0.5,
    "init": 2,
    "normalization": "Min_Max",
    "seed_selection": "cosine"
}

best_k_steph = {
    "graph": "knn_scNone_pcaNone_Directed_False_Weighted_True_neighbors_5_metriccosine",
    "k": 20,
    "alpha": 0.8,
    "init": 2,
    "hops": 2,
    "reranker": "BM25",
    "normalization": "Min_Max",
    "seed_selection": "cosine"
}
def prepare_all():
    prepare_dataset()
    return


def prepare_dataset():
    # Function that loads the dataset and calculating the embeddings and then save them

    # Loading data from the dataset bioasq
    bioasq, bioasq_corpus, bioasq_test = dt.read_dataset_bioasq("Datasets/rag-mini-bioasq/")

    # Converting bioasq, bioasq_corpus dataframes to three dictionaries
    questions_emb, answers_emb, answers_text, questions_text = emb.make_embeddings(bioasq)
    questions_emb_test, answers_emb_test, answers_text_test, questions_text_test = emb.make_embeddings(bioasq_test)
    corpus_emb, corpus_text = emb.make_embeddings_corpus(bioasq_corpus)

    # Saving the dictionaries questions, answers, corpus in a binary file
    dataset_embs = [questions_emb, answers_emb, corpus_emb]
    dataset_text = [questions_text, answers_text, corpus_text]
    dataset_test_emb = [questions_emb_test, answers_emb_test]
    dataset_test_text = [questions_text_test, answers_text_test]
    dt.save_data(dataset_embs, dataset_text,dataset_test_emb,dataset_test_text)


def rearrange_results_and_save(method, scores, results_file, params, predictions_per_query, relevant_passage_ids_per_query, latency_per_query, query_type):
    recall_total = [eval[1] for eval in scores["recall"]]
    rr_total = [eval[1] for eval in scores["rr"]]
    recall_map = {r[0]: r[1] for r in scores["recall"]}
    ndcg_map = {r[0]: r[1] for r in scores["ndcg"]}
    map_map = {r[0]: r[1] for r in scores["avg_precisions"]}
    eval_results_for_each_query = {eval[0]: {"mrr": eval[1],
                                    "rank": eval[2] ,
                                    "recall": recall_map[eval[0]],
                                    "predictions": predictions_per_query[eval[0]],
                                    "latency": latency_per_query[eval[0]],
                                    "ndcg": ndcg_map[eval[0]],
                                    "map": map_map[eval[0]]
                                    }
                          for eval in scores["rr"]}
    ndcg_total = [eval[1] for eval in scores["ndcg"]]
    avg_prec_sum = [eval[1] for eval in scores["avg_precisions"]]

    final_scores = {}
    final_scores["recallk"] = sum(recall_total)/len(recall_total)
    final_scores["mrr"] = ev.MRR_score(rr_total)
    final_scores["ndcg"] = sum(ndcg_total)/len(ndcg_total)
    final_scores["mapk"] = sum(avg_prec_sum)/len(avg_prec_sum)
    final_scores["latency"] = scores["latency"]
    dt.save_aggregate_record(method, final_scores, params, results_file, query_type)
    dt.save_eval_results_for_each_query(params, eval_results_for_each_query, method, relevant_passage_ids_per_query, results_file, query_type)

def rearrange_results_and_save_for_all_samples(method, scores, results_file, params, query_type, total_latency):
    recall_total = [eval[1] for eval in scores["recall"]]
    rr_total = [eval[1] for eval in scores["rr"]]
    recall_map = {r[0]: r[1] for r in scores["recall"]}
    ndcg_total = [eval[1] for eval in scores["ndcg"]]
    avg_prec_sum = [eval[1] for eval in scores["avg_precisions"]]

    total_scores = {}
    total_scores["recallk"] = sum(recall_total) / len(recall_total)
    total_scores["mrr"] = ev.MRR_score(rr_total)
    total_scores["ndcg"] = sum(ndcg_total) / len(ndcg_total)
    total_scores["mapk"] = sum(avg_prec_sum) / len(avg_prec_sum)
    total_scores["latency"] = total_latency
    dt.save_aggregate_record(method, total_scores, params, results_file, query_type)
"""-------------------------------------------------------------Search-------------------------------------------------------------"""
def baseline_search(query_ids, k, results_file, sample_type, query_type, seed_selection = "cosine"):
    # Function that retrieves for each query the top-k most similar data and evaluates the results
    rr_scores = []
    recallk_scores =[]
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    relevant_passage_ids_per_query = {}
    predictions_per_query = {}
    latency_per_query = {}
    params = {}
    total_latency = 0.0
    for query_id in query_ids:
       # For each query calculates the top-k
       start_time = time.perf_counter()
       predictions, correct, _ = rt.top_k(query_id, query_type, seed_selection, k)
       end_time = time.perf_counter()
       execution_time = round(end_time - start_time, 3)

       # Evaluating the results
       total_latency += execution_time
       predictions_per_query[query_id] = predictions
       relevant_passage_ids_per_query[query_id] = correct
       latency_per_query[query_id] = execution_time
       recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
       score, rank = ev.RR_score(predictions, correct, k)
       rr_scores.append((query_id,  score, rank))
       ndcg_scores.append((query_id, ev.nDCGk_score(predictions, correct, k)))
       avg_precisions.append((query_id, ev.avg_precision(predictions, correct, k)))

    # Printing and saving the results
    eval_scores["recall"] = recallk_scores
    eval_scores["rr"] = rr_scores
    eval_scores["ndcg"] = ndcg_scores
    eval_scores["avg_precisions"] = avg_precisions
    eval_scores["latency"] = total_latency
    params["k"] = k
    params["sample_type"] = sample_type
    params["seed_selection"] = seed_selection
    rearrange_results_and_save("Baseline", eval_scores, results_file, params, predictions_per_query,
                               relevant_passage_ids_per_query, latency_per_query, query_type)
    return eval_scores, params, predictions_per_query, relevant_passage_ids_per_query, total_latency


def personalised_pagerank_search(query_ids, graph, k, results_file, init, sample_type, alpha, file_name, run_id, query_type, norm_pipeline, seed_selection):
    # Function that retrieves data from a graph using personalised pagerank and evaluates the results
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    relevant_passage_ids_per_query = {}
    predictions_per_query = {}
    latency_per_query = {}
    graph_cache = dt.load_graph_cache(file_name)
    adjacency = csr_matrix(graph_cache['adjacency'])
    total_latency = 0.0
    for query_id in query_ids:
        # For each query calculates the top-k
        imporant_nodes, correct, sims = rt.top_k(query_id, query_type, seed_selection, init)
        # Running personalised pagerank
        start_time = time.perf_counter()
        predictions = rt.personalised_pagerank(graph, adjacency, graph_cache['node_to_idx'], graph_cache['global_node_list'],
                                               imporant_nodes, sims, k, query_id, alpha, query_type, norm_pipeline)
        end_time = time.perf_counter()
        execution_time = round(end_time - start_time, 3)
        total_latency += execution_time
        predictions_per_query[query_id] = predictions
        relevant_passage_ids_per_query[query_id] = [int(x) for x in correct]
        latency_per_query[query_id] = execution_time
        recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
        score, rank = ev.RR_score(predictions, correct, k)
        rr_scores.append((query_id, score, rank))
        ndcg_scores.append((query_id, ev.nDCGk_score(predictions, correct, k)))
        avg_precisions.append((query_id, ev.avg_precision(predictions, correct, k)))
    # Printing and saving the results
    eval_scores["recall"] = recallk_scores
    eval_scores["rr"] = rr_scores
    eval_scores["ndcg"] = ndcg_scores
    eval_scores["avg_precisions"] = avg_precisions
    eval_scores["latency"] = total_latency
    params["k"] = k
    params["alpha"] = alpha
    params["sample_type"] = sample_type
    params["init"] = init
    params["graph"] = file_name
    params["run_id"] = run_id
    params["norm"] = norm_pipeline
    params["seed_selection"] = seed_selection
    rearrange_results_and_save("PPR", eval_scores, results_file, params, predictions_per_query, relevant_passage_ids_per_query, latency_per_query, query_type)

    return eval_scores, params, total_latency
def k_steph_search(query_ids, graph, reranker_type, k, hops, alpha, results_file, init, sample_type, file_name, run_id, query_type, norm_pipeline,
                   seed_selection):

    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    relevant_passage_ids_per_query = {}
    predictions_per_query = {}
    latency_per_query = {}
    total_latency = 0.0
    if reranker_type == "graph_aware":
        graph_cache = dt.load_graph_cache(file_name)
        adjacency = csr_matrix(graph_cache['adjacency'])

    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, query_type, seed_selection, init)
        start_time = time.perf_counter()
        if reranker_type == "graph_aware":
            predictions, scores = rt.k_step_neighborhood_expansion(graph, imporant_nodes, query_id, k, hops, alpha,
                                                                   sims, reranker_type, query_type, norm_pipeline, adjacency, graph_cache['node_to_idx'], graph_cache['global_node_list'])
        else:
            predictions, scores = rt.k_step_neighborhood_expansion(graph, imporant_nodes, query_id, k, hops, alpha,
                                                                   sims, reranker_type, query_type, norm_pipeline)
        end_time = time.perf_counter()
        execution_time = round(end_time - start_time, 3)
        total_latency += execution_time
        predictions_per_query[query_id] = predictions
        relevant_passage_ids_per_query[query_id] = [int(x) for x in correct]
        latency_per_query[query_id] = execution_time
        recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
        score, rank = ev.RR_score(predictions, correct, k)
        rr_scores.append((query_id, score, rank))
        ndcg_scores.append((query_id, ev.nDCGk_score(predictions, correct, k)))
        avg_precisions.append((query_id, ev.avg_precision(predictions, correct, k)))

    eval_scores["recall"] = recallk_scores
    eval_scores["rr"] = rr_scores
    eval_scores["ndcg"] = ndcg_scores
    eval_scores["avg_precisions"] = avg_precisions
    eval_scores["latency"] = total_latency
    params["k"] = k
    params["reranker"] = reranker_type
    params["sample_type"] = sample_type
    params["init"] = init
    params["alpha"] = alpha
    params["hops"] = hops
    params["graph"] = file_name
    params["graph"] = file_name
    params["run_id"] = run_id
    params["norm"] = norm_pipeline
    params["seed_selection"] = seed_selection
    rearrange_results_and_save("k-steph", eval_scores, results_file, params, predictions_per_query, relevant_passage_ids_per_query, latency_per_query, query_type)
    return eval_scores, params, total_latency

"""-------------------------------------------------------------Search-------------------------------------------------------------"""
"""-------------------------------------------------------------Experiments-------------------------------------------------------------"""

def run_retrieval_ppr_deep_sensitivity_experiment():
    small, medium, long = dt.load_splits_dev()

    files = dt.get_files_with_random_state("Outputs/graphs", random_state)

    # Parameter grids
    k_retrive = [5, 10, 20]
    alphas = [0 ,0.2, 0.5, 0.8, 1.0]
    inits = [2, 5, 10, 50]
    norm_pipeline = normalization
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    seed_selection = seed_selection_method
    EXPERIMENT_NAME = "deep sensitivity experiment"
    NOTES = "testing ppr for a lot of different parameters for a small chunk of data for debug"
    SPLITS_USED = "dev split"
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits, "norm_pipeline": norm_pipeline
        , "query_type": query_type, "seed_selection": seed_selection}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small+ medium+ long, save_dir, query_type)
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        #print(f"\n=== File: {file} ===")

        # --- ppr_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
           for alpha in alphas:
             for init in inits:
                 eval_scores = None
                 params = None
                 total_latency = 0.0
                 for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                     scores, p, latency = personalised_pagerank_search(dataset, graph, k, save_dir, init,
                                                                              name, alpha, file, run_id, query_type, norm_pipeline,
                                                                              seed_selection)
                     total_latency += latency
                     if eval_scores is None:
                         eval_scores, params = scores, p
                     else:
                         for m in metrics:
                             eval_scores[m] += scores[m]
                 params["sample_type"] = "all_samples"
                 rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params,
                                                            query_type, total_latency)

    tables.make_ppr_tables(run_id, "dev")

    dt.freeze_ppr_configs(run_id, "small", "dev")
    dt.freeze_ppr_configs(run_id, "long", "dev")
    dt.freeze_ppr_configs(run_id, "medium", "dev")
    dt.freeze_ppr_configs(run_id, "all_samples", "dev")

    metrics = ["recall", "mrr", "ndcg", "map"]
    sample_types = ["small", "medium", "long", "all_samples"]
    pt.make_spearmanr_plots(run_id)
    for metric in metrics:
        for sample_type in sample_types:
            pt.make_plot_performance_of_different_graph_types(metric, "PPR", sample_type, "dev", True,
                                                              run_id)
def ablation_study(param_to_study):
    small, medium, long = dt.load_splits_dev()
    files = dt.get_files_for_param("Outputs/graphs", param_to_study)

    # Parameter grids
    k_retrive = [5, 10, 20]
    alphas = [0, 0.2, 0.5, 0.8, 1.0]
    inits = [2, 5, 10, 50]
    norm_pipeline = normalization
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    seed_selection = seed_selection_method
    EXPERIMENT_NAME = f"deep sensitivity experiment for different params in the contruction of the graph {param_to_study}"
    NOTES = "testing ppr for a lot of different parameters for a small chunk of data for debug"
    SPLITS_USED = "dev split"
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits, "norm_pipeline": norm_pipeline
        , "query_type": query_type, "seed_selection": seed_selection, "param": param_to_study}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        # print(f"\n=== File: {file} ===")

        # --- ppr_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
            for alpha in alphas:
                for init in inits:
                    eval_scores = None
                    params = None
                    total_latency = 0.0
                    for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                        scores, p, latency = personalised_pagerank_search(dataset, graph, k, save_dir, init,
                                                                          name, alpha, file, run_id, query_type,
                                                                          norm_pipeline,
                                                                          seed_selection)
                        total_latency += latency
                        if eval_scores is None:
                            eval_scores, params = scores, p
                        else:
                            for m in metrics:
                                eval_scores[m] += scores[m]
                    params["sample_type"] = "all_samples"
                    rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params,
                                                               query_type, total_latency)

    return

def run_retrieval_ppr_deep_sensitivity_experiment_for_agglo():
    small, medium, long = dt.load_splits_dev()

    files = dt.get_agglo_files("Outputs/graphs")

    # Parameter grids
    k_retrive = [5, 10, 20]
    alphas = [0, 0.2, 0.5, 0.8, 1.0]
    inits = [2, 5, 10, 50]
    norm_pipeline = normalization
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    seed_selection = seed_selection_method
    EXPERIMENT_NAME = "deep sensitivity experiment for agglo kmeans"
    NOTES = "testing ppr for a lot of different parameters for a small chunk of data for debug"
    SPLITS_USED = "dev split"
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits, "norm_pipeline": norm_pipeline
        , "query_type": query_type, "seed_selection": seed_selection}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        # print(f"\n=== File: {file} ===")

        # --- ppr_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
            for alpha in alphas:
                for init in inits:
                    eval_scores = None
                    params = None
                    total_latency = 0.0
                    for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                        scores, p, latency = personalised_pagerank_search(dataset, graph, k, save_dir, init,
                                                                          name, alpha, file, run_id, query_type,
                                                                          norm_pipeline,
                                                                          seed_selection)
                        total_latency += latency
                        if eval_scores is None:
                            eval_scores, params = scores, p
                        else:
                            for m in metrics:
                                eval_scores[m] += scores[m]
                    params["sample_type"] = "all_samples"
                    rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params,
                                                               query_type, total_latency)
    dt.seperate_results(run_id, query_type)
    tables.make_query_table(run_id, query_type, "all")
    tables.make_query_table(run_id, query_type, "one hop")
    tables.make_query_table(run_id, query_type, "multi hop")
    tables.make_leaderboard_table(run_id)
    tables.make_alpha_sensitivity_table(run_id)
    tables.make_init_sensitivity_table(run_id)
    tables.make_graph_construction_sensitivity_table(run_id)
    tables.make_paired_bootstrap_table(run_id, "PPR", "all_samples", query_type)

def run_retrieval_k_steph(reranker, files=None):
    small, medium, long = dt.load_splits_dev()
    if files is None:
        files = dt.get_files_with_random_state("Outputs/graphs", 42)


    # Parameter grids
    k_retrive = [5, 10, 20]
    alphas = [0 ,0.2, 0.5, 0.8, 1.0]
    inits = [2, 5, 10]
    hops = [1, 2, 3]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    norm_pipeline = normalization
    seed_selection = seed_selection_method

    EXPERIMENT_NAME = "deep sensitivity experiment k steph"
    NOTES = "k steph for a lot of different parameters "
    SPLITS_USED = "dev split"
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits, "hops": hops,"norm_pipeline": norm_pipeline
        , "query_type": query_type, "seed_selection": seed_selection}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)


    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        # --- k_steph_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
            for alpha in alphas:
                for init in inits:
                    for hop in hops:
                        eval_scores = None
                        params = None
                        total_latency = 0.0
                        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                            scores, p, latency = k_steph_search(dataset, graph, reranker, k, hop, alpha,
                                                       save_dir, init, name, file, run_id, query_type, norm_pipeline, seed_selection)

                            total_latency += latency
                            if eval_scores is None:
                                eval_scores, params = scores, p
                            else:
                                for m in metrics:
                                    eval_scores[m] += scores[m]
                        params["sample_type"] = "all_samples"
                        rearrange_results_and_save_for_all_samples("k-steph", eval_scores, save_dir, params, query_type,
                                                                   total_latency)


    tables.make_k_steph_tables(run_id, query_type)

    dt.freeze_k_steph_configs(run_id, "small", "dev")
    dt.freeze_k_steph_configs(run_id, "long", "dev")
    dt.freeze_k_steph_configs(run_id, "medium", "dev")
    dt.freeze_k_steph_configs(run_id, "all_samples", "dev")

    metrics = ["recall", "mrr", "ndcg", "map"]
    sample_types = ["small", "medium", "long", "all_samples"]

    for metric in metrics:
        for sample_type in sample_types:
            pt.make_plot_performance_of_different_graph_types(metric, "k-steph", sample_type, "dev",
                                                              run_id)

def reranker_study(reranker, files=None):
    small, medium, long = dt.load_splits_dev()
    if files is None:
        files = dt.get_files_with_random_state("Outputs/graphs", 42)


    # Parameter grids
    k_retrive = [5, 10, 20]
    alphas = [0 ,0.2, 0.5, 0.8, 1.0]
    inits = [2, 5, 10]
    hops = [1, 2, 3]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    norm_pipeline = normalization
    seed_selection = seed_selection_method

    EXPERIMENT_NAME = f"reranker experiment on k steph for {reranker}"
    NOTES = "k steph experiment for reranker BM25 "
    SPLITS_USED = "dev split"
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits, "hops": hops,"norm_pipeline": norm_pipeline
        , "query_type": query_type, "seed_selection": seed_selection}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)


    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        # --- k_steph_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
            for alpha in alphas:
                for init in inits:
                    for hop in hops:
                        eval_scores = None
                        params = None
                        total_latency = 0.0
                        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                            scores, p, latency = k_steph_search(dataset, graph, reranker, k, hop, alpha,
                                                       save_dir, init, name, file, run_id, query_type, norm_pipeline, seed_selection)

                            total_latency += latency
                            if eval_scores is None:
                                eval_scores, params = scores, p
                            else:
                                for m in metrics:
                                    eval_scores[m] += scores[m]
                        params["sample_type"] = "all_samples"
                        rearrange_results_and_save_for_all_samples("k-steph", eval_scores, save_dir, params, query_type,
                                                                   total_latency)


    tables.make_k_steph_tables(run_id, query_type)

def run_ppr_on_test_set():
    small, medium, long = dt.load_splits_test()
    # Parameter grids
    k = best_ppr["k"]
    alpha = best_ppr["alpha"]
    init = best_ppr["init"]
    norm_pipeline = best_ppr["normalization"]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "test"
    seed_selection = best_ppr["seed_selection"]
    EXPERIMENT_NAME = "Test set for ppr"
    NOTES = "testing prr  for test set"
    SPLITS_USED = "test split"
    parameters = {'alphas': alpha, 'k': k, 'inits': init, "norm_pipeline": norm_pipeline
        , "query_type": query_type, "seed_selection": seed_selection}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)
    pbar = tqdm(range(len([best_ppr["graph"]])))
    for i in pbar:
        file = best_ppr["graph"]
        pbar.set_description(f"Processing {file}")
        graph = dt.load_graph(file)

        eval_scores = None
        params = None
        total_latency = 0.0
        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
            scores, p, latency = personalised_pagerank_search(dataset, graph, k, save_dir, init,
                                                                          name, alpha, file, run_id, query_type,
                                                                          norm_pipeline,
                                                                          seed_selection)
            total_latency += latency
            if eval_scores is None:
               eval_scores, params = scores, p
            else:
               for m in metrics:
                   eval_scores[m] += scores[m]
        params["sample_type"] = "all_samples"
        rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params,
                                                               query_type, total_latency)

    tables.make_ppr_tables(run_id, "test")

def run_k_steph_on_test_set():
    small, medium, long = dt.load_splits_test()
    # Parameter grids
    k = best_k_steph["k"]
    alpha = best_k_steph["alpha"]
    init = best_k_steph["init"]
    hop = best_k_steph["hops"]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "test"
    norm_pipeline = best_k_steph["normalization"]
    reranker = best_k_steph["reranker"]
    seed_selection = best_k_steph["seed_selection"]

    EXPERIMENT_NAME = "test set for k-steph"
    NOTES = "k steph for test set "
    SPLITS_USED = "test split"
    parameters = {'alphas': alpha, 'k': k, 'inits': init, "hops": hop, "norm_pipeline": norm_pipeline
        , "query_type": query_type, "seed_selection": seed_selection}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)

    pbar = tqdm(range(len([best_k_steph["graph"]])))
    for i in pbar:
        file = best_k_steph["graph"]
        pbar.set_description(f"Processing {file}")
        graph = dt.load_graph(file)
        eval_scores = None
        params = None
        total_latency = 0.0
        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                 scores, p, latency = k_steph_search(dataset, graph, reranker, k, hop, alpha,
                                                                save_dir, init, name, file, run_id, query_type,
                                                                norm_pipeline, seed_selection)

                 total_latency += latency
                 if eval_scores is None:
                    eval_scores, params = scores, p
                 else:
                     for m in metrics:
                         eval_scores[m] += scores[m]
        params["sample_type"] = "all_samples"
        rearrange_results_and_save_for_all_samples("k-steph", eval_scores, save_dir, params, query_type,
                                                                   total_latency)

    tables.make_k_steph_tables(run_id, query_type)

def run_an_example_retrieval_ppr():
    small, medium, long = dt.load_splits_dev()
    small = small[:5]
    files = dt.get_files("Outputs/graphs")
    files = files[:3]
    # Parameter grids
    k_retrive = [5, 10]
    alphas = [0 ,0.2, 0.5]
    inits = [2, 5, 10]
    norm = ""
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    EXPERIMENT_NAME = "ppr_working_example"
    NOTES = "testing for a very small amount of samples how tables, record, saving works"
    SPLITS_USED = "small"
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id,NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small, save_dir, query_type)

    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        #print(f"\n=== File: {file} ===")
        for k in k_retrive:
           for alpha in alphas:
             for init in inits:
                 for name, dataset in [("small", small)]:
                     scores, p, latency = personalised_pagerank_search(dataset, graph, k, save_dir, init,
                                                                 name, alpha, file, run_id,  query_type, norm)

    dt.seperate_results(run_id, query_type)
    tables.make_query_table(run_id, query_type)
    tables.make_leaderboard_table(run_id)
    tables.make_alpha_sensitivity_table(run_id)
    tables.make_init_sensitivity_table(run_id)
    tables.make_graph_construction_sensitivity_table(run_id)

def run_an_example_retrieval_k_steph():
    small, medium, long = dt.load_splits_dev()
    files = dt.get_files("Outputs/graphs")
    files = files[:3]
    small = small[:5]

    # Parameter grids
    k_retrive = [10]
    alphas = [0 ,0.2, 0.5]
    inits = [2, 5, 10]
    hops = [1, 2, 3, 4]
    norm_pipeline = ""
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits, "hops": hops, "norm": norm_pipeline}
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    seed_selection = "cosine"
    EXPERIMENT_NAME = "k-steph_working_example"
    NOTES = "testing for a very small amount of samples how tables, record, saving works"
    SPLITS_USED = "small"
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small, save_dir, query_type)
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")

        graph = dt.load_graph(file)
        # --- k_steph_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
            for alpha in alphas:
                for init in inits:
                    for hop in hops:
                        for name, dataset in [("small", small)]:
                            scores, p, latency = k_steph_search(dataset, graph, "graph_aware", k, hop, alpha,
                                                       save_dir, init, name, file, run_id, query_type, norm_pipeline, seed_selection)

    dt.seperate_results(run_id, query_type)
    tables.make_query_table(run_id, query_type)
    tables.make_leaderboard_table(run_id)
    tables.make_alpha_sensitivity_table(run_id)
    tables.make_init_sensitivity_table(run_id)
    tables.make_graph_construction_sensitivity_table(run_id)
    tables.make_hop_sensitivity_table(run_id)

def run_baseline(query_type):
    """Function that calculate baseline scores"""
    # loading the correct split
    if query_type == "dev":
        small, medium, long = dt.load_splits_dev()
    elif query_type == "test":
        small, medium, long = dt.load_splits_test()
    else:
        raise ValueError(f"Wrong query type {query_type}")

    k_retrive = [5, 10, 20]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    dt.sanity_check(small + medium + long, "Outputs/Baseline-RAG", query_type)
    dt.clean_baseline(query_type)

    for k in k_retrive:
        eval_scores = None
        total_latency = 0
        params = None
        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
            scores, p, predictions_per_query, relevant_passage_ids_per_query, latency = baseline_search(dataset, k,
                                                                                                  "Outputs/Baseline-RAG",
                                                                                                  name, query_type)
            total_latency += latency
            if eval_scores is None:
                eval_scores, params = scores, p
            else:
                for m in metrics:
                    eval_scores[m] += scores[m]
        params["sample_type"] = "all_samples"
        rearrange_results_and_save_for_all_samples("Baseline", eval_scores, "Outputs/Baseline-RAG", params, query_type, total_latency)

def ppr_score_calibration_study():
    small, medium, long = dt.load_splits_dev()
    files = ["mutual_scNone_pcaNone_Directed_False_Weighted_True_neighbors_20_metriccosine",
             "knn_scNone_pcaNone_Directed_False_Weighted_True_neighbors_10_metriccosine",
             "threshold_scNone_pcaNone_Directed_False_Weighted_True_threshold_distance0.4"]

    alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
    inits = [2]
    norm_pipeline = ["", "log + Min_Max", "Min_Max", "z_score"]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    seed_selection = "cosine"
    EXPERIMENT_NAME = "ppr_score_calibration_study"
    NOTES = "Testing ppr fusion with or without normalisation"
    SPLITS_USED = "dev split"
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    parameters = {'alpha': alphas, 'inits': inits, "norm_pipelines": norm_pipeline, "query_type": query_type, "seed_selection": seed_selection}

    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        for norm in norm_pipeline:
            for init in inits:
                for alpha in alphas:
                        eval_scores = None
                        params = None
                        total_latency = 0.0
                        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                            scores, p, latency = personalised_pagerank_search(dataset, graph, 10, save_dir, init,
                                                                     name, alpha, file, run_id, query_type, norm, seed_selection)
                            total_latency += latency
                            if eval_scores is None:
                                eval_scores, params = scores, p
                            else:
                                for m in metrics:
                                    eval_scores[m] += scores[m]
                        params["sample_type"] = "all_samples"
                        rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params, query_type, total_latency)

    dt.seperate_results(run_id, query_type)
    tables.make_leaderboard_table(run_id)
    tables.make_norm_not_norm_table(run_id)

def ppr_seed_selection_study():
    small, medium, long = dt.load_splits_dev()
    files = ["mutual_scNone_pcaNone_Directed_False_Weighted_True_neighbors_20_metriccosine",
             "knn_scNone_pcaNone_Directed_False_Weighted_True_neighbors_10_metriccosine",
             "threshold_scNone_pcaNone_Directed_False_Weighted_True_threshold_distance0.4"]

    alphas = [0.5]
    inits = [2, 5, 10, 20, 50]
    norm_pipeline = ["Min_Max"]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    seed_selection = ["cosine", "bm25"]
    EXPERIMENT_NAME = "ppr_seed_selection_study"
    NOTES = "Testing ppr seed selection using cosine and bm25"
    SPLITS_USED = "dev split"
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    parameters = {'alpha': alphas, 'inits': inits, "norm_pipelines": norm_pipeline, "query_type": query_type,
                  "seed_selection": seed_selection}

    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        for seed in seed_selection:
            for norm in norm_pipeline:
                for init in inits:
                    for alpha in alphas:
                        eval_scores = None
                        params = None
                        total_latency = 0.0
                        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                            scores, p, latency = personalised_pagerank_search(dataset, graph, 10, save_dir, init,
                                                                              name, alpha, file, run_id, query_type, norm,
                                                                              seed)
                            total_latency += latency
                            if eval_scores is None:
                                eval_scores, params = scores, p
                            else:
                                for m in metrics:
                                    eval_scores[m] += scores[m]
                        params["sample_type"] = "all_samples"
                        rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params, query_type,
                                                                   total_latency)

    dt.seperate_results(run_id, query_type)
    tables.make_leaderboard_table(run_id)
    tables.make_seed_selection_table(run_id)

def k_means_random_state_study():
    small, medium, long = dt.load_splits_dev()
    all_samples = small + medium + long
    alpha = 0.5
    init = 2
    norm_pipeline = normalization
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    query_type = "dev"
    seed_selection = seed_selection_method
    k = 10
    random_states = [1, 7, 21, 42, 84]
    EXPERIMENT_NAME = "kmeans_random_state_selection_study"
    NOTES = "Choosing the best possible random_state for kmeans for the retrieval will perform better"
    SPLITS_USED = "dev split"
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    parameters = {'alpha': alpha, 'inits': init, "norm_pipelines": norm_pipeline, "query_type": query_type,
                  "seed_selection": seed_selection}

    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir, query_type)
    files = dt.get_kmeans_files("Outputs/graphs")
    pbar = tqdm(range(len(files)))
    total_latency = 0.0
    for i in pbar:
        file = files[i]
        graph = dt.load_graph(file)

        for name, dataset in [("all_samples", all_samples)]:
            scores, p, latency = personalised_pagerank_search(dataset, graph, k, save_dir, init,
                                                                              name, alpha, file, run_id, query_type, norm_pipeline,
                                                                              seed_selection)
    dt.seperate_results(run_id, query_type)
    tables.make_random_state_table(run_id)


"""-------------------------------------------------------------Experiments-------------------------------------------------------------"""



