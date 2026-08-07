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
"""-------------------------------------------------------------Help functions-------------------------------------------------------------"""

"""-------------------------------------------------------------Help functions-------------------------------------------------------------"""
def prepare_all():
    prepare_dataset()
    return


def prepare_dataset():
   # Loading the dataset and calculating the embeddings and then save them

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


def rearrange_results_and_save(method, scores, results_file, params, predictions_per_query, relevant_passage_ids_per_query, latency_per_query):
    recall_total = [eval[1] for eval in scores["recall"]]
    rr_total = [eval[1] for eval in scores["rr"]]
    recall_map = {r[0]: r[1] for r in scores["recall"]}
    eval_results_for_each_query = {eval[0]: {"mrr": eval[1],
                                    "rank": eval[2] ,
                                    "recall": recall_map[eval[0]],
                                    "predictions": predictions_per_query[eval[0]],
                                    "latency": latency_per_query[eval[0]]
                                    }
                          for eval in scores["rr"]}
    ndcg_total = [eval[1] for eval in scores["ndcg"]]
    avg_prec_sum = [eval[1] for eval in scores["avg_precisions"]]

    final_scores = {}
    final_scores["recallk"] = sum(recall_total)/len(recall_total)
    final_scores["mrr"] = ev.MRR_score(rr_total)
    final_scores["ndcg"] = sum(ndcg_total)/len(ndcg_total)
    final_scores["mapk"] = sum(avg_prec_sum)/len(avg_prec_sum)
    dt.save_aggregate_record(method, final_scores, params, results_file)
    dt.save_eval_results_for_each_query(params, eval_results_for_each_query, method, relevant_passage_ids_per_query, results_file)

def rearrange_results_and_save_for_all_samples(method, scores, results_file, params):
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
    dt.save_aggregate_record(method, total_scores, params, results_file)
"""-------------------------------------------------------------Search-------------------------------------------------------------"""
def baseline_search(query_ids, k, results_file, sample_type):
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
    for query_id in query_ids:
       # For each query calculates the top-k
       start_time = time.perf_counter()
       predictions, correct, _ = rt.top_k(query_id, k)
       end_time = time.perf_counter()
       execution_time = round(end_time - start_time, 3)

       # Evaluating the results
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
    params["k"] = k
    params["sample_type"] = sample_type
    rearrange_results_and_save("Baseline", eval_scores, results_file, params, predictions_per_query, relevant_passage_ids_per_query, latency_per_query)
    return eval_scores, params, predictions_per_query, relevant_passage_ids_per_query


def personalised_pagerank_search(query_ids, graph, k, results_file, init, sample_type, alpha, file_name, run_id, norm):
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
    for query_id in query_ids:
        # For each query calculates the top-k
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        # Running personalised pagerank
        start_time = time.perf_counter()
        predictions = rt.personalised_pagerank(graph, adjacency, graph_cache['node_to_idx'], graph_cache['global_node_list'],
                                               imporant_nodes, sims, k, query_id, alpha, norm)
        end_time = time.perf_counter()
        execution_time = round(end_time - start_time, 3)
        # print(predictions)
        # print(correct)
        # print("---------------")
        # Evaluating the results
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
    params["k"] = k
    params["alpha"] = alpha
    params["sample_type"] = sample_type
    params["init"] = init
    params["graph"] = file_name
    params["run_id"] = run_id
    rearrange_results_and_save("PPR", eval_scores, results_file, params, predictions_per_query, relevant_passage_ids_per_query, latency_per_query)

    return eval_scores, params
def k_steph_search(query_ids, graph, reranker_type, k, hops, alpha, results_file, init, sample_type, file_name, run_id):
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    relevant_passage_ids_per_query = {}
    predictions_per_query = {}
    latency_per_query = {}
    if reranker_type == "graph_aware":
        graph_cache = dt.load_graph_cache(file_name)
        adjacency = csr_matrix(graph_cache['adjacency'])

    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        start_time = time.perf_counter()
        if reranker_type == "graph_aware":
            predictions, scores = rt.k_step_neighborhood_expansion(graph, imporant_nodes, query_id, k, hops, alpha,
                                                                   sims, reranker_type, adjacency, graph_cache['node_to_idx'], graph_cache['global_node_list'])
        else:
            predictions, scores = rt.k_step_neighborhood_expansion(graph, imporant_nodes, query_id, k, hops, alpha,
                                                                   sims, reranker_type)
        # print(predictions)
        # print(correct)
        # print("---------------")
        end_time = time.perf_counter()
        execution_time = round(end_time - start_time, 3)
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
    params["k"] = k
    params["reranker"] = reranker_type
    params["sample_type"] = sample_type
    params["init"] = init
    params["alpha"] = alpha
    params["hops"] = hops
    params["graph"] = file_name
    params["graph"] = file_name
    params["run_id"] = run_id
    rearrange_results_and_save("k-steph", eval_scores, results_file, params, predictions_per_query, relevant_passage_ids_per_query,latency_per_query)
    return eval_scores, params

def hits_search(query_ids, graph, graph_type, k, alpha, results_file, init, sample_type, file_name):
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        predictions = rt.hits(graph, imporant_nodes, k)
        recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
        score, rank = ev.RR_score(predictions, correct)
        rr_scores.append((query_id, score, rank))
        ndcg_scores.append((query_id, ev.nDCGk_score(predictions, correct, k)))
        avg_precisions.append((query_id, ev.avg_precision(predictions, correct, k)))

    eval_scores["recall"] = recallk_scores
    eval_scores["rr"] = rr_scores
    eval_scores["ndcg"] = ndcg_scores
    eval_scores["avg_precisions"] = avg_precisions
    params["k"] = k
    params["reranker"] = ""
    params["sample_type"] = sample_type
    params["init"] = init
    params["graph"] = file_name
    rearrange_results_and_save("Hits", eval_scores, results_file, params)
    return

def shortest_path_search(query_ids, graph, reranker_type, k, alpha, results_file, init, sample_type, file_name):
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        predictions, _ = rt.shortest_path(graph, query_id,imporant_nodes, k, alpha, sims, reranker_type)
        # print(predictions)
        # print(correct)
        # print("---------------")
        recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
        score, rank = ev.RR_score(predictions, correct)
        rr_scores.append((query_id, score, rank))
        ndcg_scores.append((query_id, ev.nDCGk_score(predictions, correct, k)))
        avg_precisions.append((query_id, ev.avg_precision(predictions, correct, k)))

    eval_scores["recall"] = recallk_scores
    eval_scores["rr"] = rr_scores
    eval_scores["ndcg"] = ndcg_scores
    eval_scores["avg_precisions"] = avg_precisions
    params["k"] = k
    params["reranker"] = reranker_type
    params["sample_type"] = sample_type
    params["init"] = init
    rearrange_results_and_save("Shortest Path", eval_scores, results_file, params)
    return
"""-------------------------------------------------------------Search-------------------------------------------------------------"""
"""-------------------------------------------------------------Experiments-------------------------------------------------------------"""
def run_retrieval_ppr_deep_sensitivity_experiment():
    small, medium, long = dt.load_splits_dev()
    files = dt.get_files("Outputs/graphs")

    # Parameter grids
    k_retrive = [5, 10, 20]
    alphas = [0 ,0.2, 0.5, 0.8, 1.0]
    inits = [2, 5, 10, 20, 50]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    EXPERIMENT_NAME = "deep sensitivity experiment"
    NOTES = "testing ppr for a lot of different parameters"
    SPLITS_USED = "dev split"
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small+ medium+ long, save_dir)
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
                 for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                     scores, p = personalised_pagerank_search(dataset, graph, k, save_dir, init,
                                                                 name, alpha, file, run_id, True)
                     if eval_scores is None:
                         eval_scores, params = scores, p
                     else:
                         for m in metrics:
                             eval_scores[m] += scores[m]
                 params["sample_type"] = "all_samples"
                 rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params)

    dt.seperate_results(run_id)
    tables.make_query_table(run_id)
    tables.make_leaderboard_table(run_id)
    tables.make_alpha_sensitivity_table(run_id)
    tables.make_init_sensitivity_table(run_id)
    tables.make_graph_construction_sensitivity_table(run_id)

def run_retrieval_k_steph(files = None):
    small, medium, long = dt.load_splits_dev()
    if files is None:
        files = dt.get_files()


    # Parameter grids
    k_retrive = [10]
    alphas = [0 ,0.2, 0.5, 0.8, 1.0]
    inits = [2, 5, 10, 20, 50]
    hops = [1, 2, 3, 4]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
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
                        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                            scores, p = k_steph_search(dataset, graph, "graph_aware", k, hop, alpha,
                                                       f"Outputs/graphs/{file}", init, name, file)
                            if eval_scores is None:
                                eval_scores, params = scores, p
                            else:
                                for m in metrics:
                                    eval_scores[m] += scores[m]
                        params["sample_type"] = "all_samples"
                        rearrange_results_and_save_for_all_samples("k-steph", eval_scores, f"Outputs/graphs/{file}", params)

        tables.save_leaderboard(f"Outputs/graphs/{file}", "leaderboards")
        #dt.seperate_results(file)


def run_an_example_retrieval_ppr():
    small, medium, long = dt.load_splits_dev()
    small = small[:5]
    files = dt.get_files("Outputs/graphs")
    files = files[:3]
    # Parameter grids
    k_retrive = [5, 10]
    alphas = [0 ,0.2, 0.5]
    inits = [2, 5, 10]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    EXPERIMENT_NAME = "ppr_working_example"
    NOTES = "testing for a very small amount of samples how tables, record, saving works"
    SPLITS_USED = "small"
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits}
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id,NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small, save_dir)

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
                     scores, p = personalised_pagerank_search(dataset, graph, k, save_dir, init,
                                                                 name, alpha, file, run_id, True)

    dt.seperate_results(run_id)
    tables.make_query_table(run_id)
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
    parameters = {'alphas': alphas, 'k': k_retrive, 'inits': inits, "hops": hops}
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    EXPERIMENT_NAME = "k-steph_working_example"
    NOTES = "testing for a very small amount of samples how tables, record, saving works"
    SPLITS_USED = "small"
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small, save_dir)
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
                            scores, p = k_steph_search(dataset, graph, "graph_aware", k, hop, alpha,
                                                       save_dir, init, name, file, run_id)

    dt.seperate_results(run_id)
    tables.make_query_table(run_id)
    tables.make_leaderboard_table(run_id)
    tables.make_alpha_sensitivity_table(run_id)
    tables.make_init_sensitivity_table(run_id)
    tables.make_graph_construction_sensitivity_table(run_id)
    tables.make_hop_sensitivity_table(run_id)


def run_an_baseline():
    small, medium, long = dt.load_splits_dev()
    k_retrive = [5, 10]
    parameters = {'k': k_retrive}
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    dt.sanity_check(small + medium + long, "Outputs/Baseline-RAG")

    with open("Outputs/Baseline-RAG/queries.json", "w") as f:
            f.write("")
    with open("Outputs/Baseline-RAG/aggregate_results.jsonl", "w") as f:
            f.write("")
    for k in k_retrive:
        eval_scores = None
        params = None
        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
            scores, p, predictions_per_query, relevant_passage_ids_per_query = baseline_search(dataset, k,
                                                                                                  "Outputs/Baseline-RAG",
                                                                                                  name)
            if eval_scores is None:
                eval_scores, params = scores, p
            else:
                for m in metrics:
                    eval_scores[m] += scores[m]
        params["sample_type"] = "all_samples"
        rearrange_results_and_save_for_all_samples("Baseline", eval_scores, "Outputs/Baseline-RAG", params)


def ppr_score_calibration_study():
    small, medium, long = dt.load_splits_dev()
    files = ["mutual_scNone_pcaNone_Directed_False_Weighted_True_neighbors_10_metriccosine",
             "knn_scNone_pcaNone_Directed_False_Weighted_True_neighbors_10_metriccosine",
             "threshold_kmeans_scNone_pcaNone_clusters_5_initk-means++_Directed_False_Weighted_True_threshold_distance0.6"]
    alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    EXPERIMENT_NAME = "ppr_score_calibration_study_norm"
    NOTES = "Testing ppr fusion with or without normalisation"
    SPLITS_USED = "dev split"
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    parameters = {'alpha': alphas}

    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir)
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        for alpha in alphas:
                eval_scores = None
                params = None
                for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                    scores, p = personalised_pagerank_search(dataset, graph, 10, save_dir, 5,
                                                             name, alpha, file, run_id, True)
                    if eval_scores is None:
                        eval_scores, params = scores, p
                    else:
                        for m in metrics:
                            eval_scores[m] += scores[m]
                params["sample_type"] = "all_samples"
                rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params)

    dt.seperate_results(run_id)
    tables.make_query_table(run_id)
    tables.make_leaderboard_table(run_id)
    tables.make_alpha_sensitivity_table(run_id)
    tables.make_init_sensitivity_table(run_id)
    tables.make_graph_construction_sensitivity_table(run_id)

    EXPERIMENT_NAME = "ppr_score_calibration_study_raw"
    NOTES = "Testing ppr fusion with or without normalisation"
    SPLITS_USED = "dev split"
    run_id = dt.make_run_id(EXPERIMENT_NAME)
    save_dir = dt.setup_run_dir(run_id, NOTES, SPLITS_USED, parameters)
    dt.sanity_check(small + medium + long, save_dir)
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        graph = dt.load_graph(file)
        for alpha in alphas:
            eval_scores = None
            params = None
            for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                scores, p = personalised_pagerank_search(dataset, graph, 10, save_dir, 5,
                                                         name, alpha, file, run_id, False)
                if eval_scores is None:
                    eval_scores, params = scores, p
                else:
                    for m in metrics:
                        eval_scores[m] += scores[m]
            params["sample_type"] = "all_samples"
            rearrange_results_and_save_for_all_samples("PPR", eval_scores, save_dir, params)

    dt.seperate_results(run_id)
    tables.make_query_table(run_id)
    tables.make_leaderboard_table(run_id)
    tables.make_alpha_sensitivity_table(run_id)
    tables.make_init_sensitivity_table(run_id)
    tables.make_graph_construction_sensitivity_table(run_id)

"""-------------------------------------------------------------Experiments-------------------------------------------------------------"""



