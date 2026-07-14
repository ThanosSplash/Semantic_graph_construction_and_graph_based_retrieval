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
"""-------------------------------------------------------------Help functions-------------------------------------------------------------"""

"""-------------------------------------------------------------Help functions-------------------------------------------------------------"""
def prepare_all():
    prepare_dataset()
    return


def prepare_dataset():
   # Loading the dataset and calculating the embeddings and then save them

    # Loading data from the dataset bioasq
    bioasq, bioasq_corpus = dt.read_dataset_bioasq("Datasets/rag-mini-bioasq/")

    # Converting bioasq, bioasq_corpus dataframes to three dictionaries
    questions_emb, answers_emb, questions_text, answers_text = emb.make_embeddings(bioasq)
    corpus_emb, corpus_text = emb.make_embeddings_corpus(bioasq_corpus)

    # Saving the dictionaries questions, answers, corpus in a binary file
    dt.save_data(questions_emb, answers_emb, corpus_emb, questions_text, answers_text, corpus_text)


def evaluate_method(method, scores, results_file, params):
    recall_total = [eval[1] for eval in scores["recall"]]
    rr_total = [eval[1] for eval in scores["rr"]]
    recall_map = {r[0]: r[1] for r in scores["recall"]}
    first_rel_indx = {eval[0]: {"mrr": eval[1],
                                "rank": eval[2] + 1 if eval[2] != -1 else eval[2],
                                "recall": recall_map[eval[0]]
                                }
                      for eval in scores["rr"]}
    ndcg_total = [eval[1] for eval in scores["ndcg"]]
    avg_prec_sum = [eval[1] for eval in scores["avg_precisions"]]

    total_scores = {}
    total_scores["recallk"] = sum(recall_total)/len(recall_total)
    total_scores["mrr"] = ev.MRR_score(rr_total)
    total_scores["ndcg"] = sum(ndcg_total)/len(ndcg_total)
    total_scores["mapk"] = sum(avg_prec_sum)/len(avg_prec_sum)

    #print(total_scores)




    dt.save_eval_results(first_rel_indx, params, method, total_scores, results_file)
    #dt.save_result("Top-" + str(k), recallk, mrr, ndcg, mapk, k, results_file)
    return


"""-------------------------------------------------------------Search-------------------------------------------------------------"""
def baseline_search(query_ids, k, results_file, sample_type):
    # Function that retrieves for each query the top-k most similar data and evaluates the results
    rr_scores = []
    recallk_scores =[]
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    for query_id in query_ids:
       # For each query calculates the top-k
       predictions, correct, _ = rt.top_k(query_id, k)
       #print(predictions)
       #print(correct)
       #print("---------------")
       # Evaluating the results
       recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
       score, rank = ev.RR_score(predictions, correct)
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
    evaluate_method("Baseline", eval_scores, results_file, params)
    return eval_scores, params


def personalised_pagerank_search(query_ids, graph, k, results_file, init, sample_type, alpha, file_name):
    # Function that retrieves data from a graph using personalised pagerank and evaluates the results
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    graph_cache = dt.load_graph_cache(file_name)
    adjacency = csr_matrix(graph_cache['adjacency'])
    for query_id in query_ids:
        # For each query calculates the top-k
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        # Running personalised pagerank
        predictions = rt.personalised_pagerank(graph, adjacency, graph_cache['node_to_idx'], graph_cache['global_node_list'],imporant_nodes, sims, k, query_id, alpha)
        # print(predictions)
        # print(correct)
        # print("---------------")
        # Evaluating the results
        recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
        score, rank = ev.RR_score(predictions, correct)
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
    evaluate_method("PPR", eval_scores, results_file, params)

    return eval_scores, params
def k_steph_search(query_ids, graph, reranker_type, k, hops, alpha, results_file, init, sample_type, file_name):
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    if reranker_type == "graph_aware":
        graph_cache = dt.load_graph_cache(file_name)
        adjacency = csr_matrix(graph_cache['adjacency'])

    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        if reranker_type == "graph_aware":
            predictions, scores = rt.k_step_neighborhood_expansion(graph, imporant_nodes, query_id, k, hops, alpha,
                                                                   sims, reranker_type, adjacency, graph_cache['node_to_idx'], graph_cache['global_node_list'])
        else:
            predictions, scores = rt.k_step_neighborhood_expansion(graph, imporant_nodes, query_id, k, hops, alpha,
                                                                   sims, reranker_type)
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
    params["alpha"] = alpha
    params["hops"] = hops
    evaluate_method("k-steph", eval_scores, results_file, params)
    return eval_scores, params

def hits_search(query_ids, graph, graph_type, k, alpha, results_file, init, sample_type):
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
    evaluate_method("Hits", eval_scores, results_file, params)
    return

def shortest_path_search(query_ids, graph, reranker_type, k, alpha, results_file, init, sample_type):
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
    evaluate_method("Shortest Path", eval_scores, results_file, params)
    return
"""-------------------------------------------------------------Search-------------------------------------------------------------"""
"""-------------------------------------------------------------Retrieval-------------------------------------------------------------"""
def run_retrieval_ppr():
    small, medium, long = dt.load_samples()
    files = dt.get_files()

    # Parameter grids
    k_retrive = [10]
    alphas = [0 ,0.2, 0.5, 0.8]
    inits = [2, 5, 10, 20, 50]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    indx = 0
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        start_time = time.perf_counter()
        #dt.clear_eval(f"Outputs/graphs/{file}/eval_results.json")
        graph = dt.load_graph(file)
        #print(f"\n=== File: {file} ===")

        # --- ppr_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
           for alpha in alphas:
             for init in inits:
                 eval_scores = None
                 params = None
                 for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                     #print(f"  [ppr] , init={init}, tag={name}")
                     scores, p = personalised_pagerank_search(dataset, graph, k, f"Outputs/graphs/{file}", init,
                                                                 name, alpha, file)
                     if eval_scores is None:
                         eval_scores, params = scores, p
                     else:
                         for m in metrics:
                             eval_scores[m] += scores[m]
                 params["sample_type"] = "all_samples"
                 evaluate_method("PPR", eval_scores, f"Outputs/graphs/{file}", params)
        dt.save_leaderboard(f"Outputs/graphs/{file}", "leaderboards")
        dt.seperate_results(file)
        indx+=1

def run_retrieval_k_steph():
    small, medium, long = dt.load_samples()
    files = dt.get_files()

    # Parameter grids
    k_retrive = [10]
    alphas = [0 ,0.2, 0.5, 0.8]
    inits = [2, 5, 10, 20, 50]
    k_steps = [2, 3]
    metrics = ["recall", "rr", "ndcg", "avg_precisions"]
    pbar = tqdm(range(len(files)))
    for i in pbar:
        file = files[i]
        pbar.set_description(f"Processing {files[i]}")
        #dt.clear_eval(f"Outputs/graphs/{file}/eval_results.json")
        graph = dt.load_graph(file)
        # --- k_steph_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
            for alpha in alphas:
                for init in inits:
                    for k_step in k_steps:
                        eval_scores = None
                        params = None
                        for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                            scores, p = k_steph_search(dataset, graph, "BM25", k, k_step, alpha,
                                                       f"Outputs/graphs/{file}", init, name, file)
                            if eval_scores is None:
                                eval_scores, params = scores, p
                            else:
                                for m in metrics:
                                    eval_scores[m] += scores[m]
                        params["sample_type"] = "all_samples"
                        evaluate_method("k-steph", eval_scores, f"Outputs/graphs/{file}", params)
                        #eval_scores = None
                        #params = None
                        #for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                            #scores, p = k_steph_search(dataset, graph, "graph_aware", k, k_step, alpha,
                            #                           f"Outputs/graphs/{file}", init, name, file)
                            #if eval_scores is None:
                                #eval_scores, params = scores, p
                            #else:
                                #for m in metrics:
                                    #eval_scores[m] += scores[m]
                        #params["sample_type"] = "all_samples"
                        #evaluate_method("k-steph", eval_scores, f"Outputs/graphs/{file}", params)
        #for k in k_retrive:
            #for init in inits:
                #for k_step in k_steps:
                    #eval_scores = None
                    #params = None
                    #for name, dataset in [("small", small), ("medium", medium), ("long", long)]:
                        #print(
                           # f"  [k_steph] reranker=cross_encoder, init={init}, tag={name}, k step={k_step}")
                        #scores, p = k_steph_search(dataset, graph, "cross_encoder", k, k_step, 0.0,
                        #                           f"Outputs/graphs/{file}", init, name)
                        #if eval_scores is None:
                            #eval_scores, params = scores, p
                        #else:
                            #for m in metrics:
                                #eval_scores[m] += scores[m]
                    #params["sample_type"] = "all_samples"
                    #evaluate_method("k-steph", eval_scores, f"Outputs/graphs/{file}", params)
        dt.save_leaderboard(f"Outputs/graphs/{file}", "leaderboards")
        dt.seperate_results(file)

"""-------------------------------------------------------------Retrieval-------------------------------------------------------------"""



