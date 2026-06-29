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
    first_rel_indx = {eval[0]: {"score": eval[1], "rank": eval[2] + 1 if eval[2] != -1 else eval[2]} for eval in scores["rr"]}
    ndcg_total = [eval[1] for eval in scores["ndcg"]]
    avg_prec_sum = [eval[1] for eval in scores["avg_precisions"]]

    total_scores = {}
    total_scores["recallk"] = sum(recall_total)/len(recall_total)
    total_scores["mrr"] = ev.MRR_score(rr_total)
    total_scores["ndcg"] = sum(ndcg_total)/len(ndcg_total)
    total_scores["mapk"] = sum(avg_prec_sum)/len(avg_prec_sum)

    print(total_scores)




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
       ndcg_scores.append((query_id, ev.ndcg_score_(predictions, correct, k)))
       avg_precisions.append((query_id, ev.avg_precision(predictions, correct, k)))
    # Printing and saving the results
    eval_scores["recall"] = recallk_scores
    eval_scores["rr"] = rr_scores
    eval_scores["ndcg"] = ndcg_scores
    eval_scores["avg_precisions"] = avg_precisions
    params["k"] = k
    params["sample_type"] = sample_type
    evaluate_method("Baseline", eval_scores, results_file, params)


def personalised_pagerank_search(query_ids, graph, k, results_file, init, sample_type, alpha):
    # Function that retrieves data from a graph using personalised pagerank and evaluates the results
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    for query_id in query_ids:
        # For each query calculates the top-k
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        # Running personalised pagerank
        predictions = rt.personalised_pagerank(graph, imporant_nodes, sims, k, query_id, alpha)
        # print(predictions)
        # print(correct)
        # print("---------------")
        # Evaluating the results
        recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
        score, rank = ev.RR_score(predictions, correct)
        rr_scores.append((query_id, score, rank))
        ndcg_scores.append((query_id, ev.ndcg_score_(predictions, correct, k)))
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

    return
def k_steph_search(query_ids, graph, reranker_type, k, hops, alpha, results_file, init, sample_type):
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    eval_scores = {}
    params = {}
    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        predictions, scores = rt.k_step_neighborhood_expansion(graph, imporant_nodes, query_id, k, hops, alpha, sims, reranker_type)
        # print(predictions)
        # print(correct)
        # print("---------------")
        recallk_scores.append((query_id, ev.recallk_score(predictions, correct, k)))
        score, rank = ev.RR_score(predictions, correct)
        rr_scores.append((query_id, score, rank))
        ndcg_scores.append((query_id, ev.ndcg_score_(predictions, correct, k)))
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
    evaluate_method("k-steph", eval_scores, results_file, params)
    return

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
        ndcg_scores.append((query_id, ev.ndcg_score_(predictions, correct, k)))
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
        ndcg_scores.append((query_id, ev.ndcg_score_(predictions, correct, k)))
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
def threshold_graph(sampled_items, threshold_params, preprocess, name, save):
    # Building a threshold graph
    gc.build_threshold_graph(sampled_items, threshold_params, preprocess, name, save)

    #G = dt.load_graph()
    #nodes_to_plot = list(G.nodes())[:100]
    #gc.plot_subgraph(G, nodes_to_plot)
    return
"""-------------------------------------------------------------Retrieval-------------------------------------------------------------"""
def run_retrieval():
    q, a, c = dt.load_data()
    small, medium, long = dt.load_samples()
    files = dt.get_files()
    all_samples = small + medium + long

    # Parameter grids
    k_retrive = [10]
    alphas = [0.3, 0.5]
    rerankers = ["BM25", "graph_aware", "cross_encoder"]
    inits = [5]
    k_step = 2

    sample_sets = {
        "small": small,
        "medium": medium,
        "long": long,
        "all_samples": all_samples,
    }
    flag = True
    for file in files:
        graph = dt.load_graph(file)
        print(f"\n=== File: {file} ===")
        if flag == True:
            # --- Baseline ---
            for k in k_retrive:
                 for tag, samples in sample_sets.items():
                   print(f" [baseline] tag={tag}")
                   baseline_search(samples, k, f"graphs/{file}", tag)
            flag = False

        # --- ppr_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
           for alpha in alphas:
             for init in inits:
                for tag, samples in sample_sets.items():
                    print(f"  [ppr] , init={init}, tag={tag}")
                    personalised_pagerank_search(samples, graph, k, f"graphs/{file}", init, tag, alpha)
        # --- shortest_path_search: sweep rerankers × alphas × sample sets ---
        #for alpha in alphas:
            #for init in inits:
                #for tag, samples in sample_sets.items():
                    #print(f"  [shortest_path] reranker= BM25, alpha={alpha}, init = {init} ,tag={tag}")
                    #shortest_path_search(
                        #samples, graph, "BM25", 10, alpha,
                       # f"graphs/{file}", init, tag
                    #)
                    #print(f"  [shortest_path] reranker= graph_aware, alpha={alpha}, tag={tag}")
                   #shortest_path_search(
                        #samples, graph, "graph_aware", 10, alpha,
                        #f"graphs/{file}", init, tag
                    #)
        #for init in inits:
            #for tag, samples in sample_sets.items():
                #print(f"  [shortest_path] reranker= cross_encoder, init={init}, tag={tag}")
                #shortest_path_search(
                    #samples, graph, "cross_encoder", 10, 0.0,
                    #f"graphs/{file}", init, tag
                #)

        # --- k_steph_search: sweep rerankers × alphas × inits × sample sets ---
        for k in k_retrive:
           for alpha in alphas:
                for init in inits:
                    for tag, samples in sample_sets.items():
                       print(f"  [k_steph] reranker=BM25, alpha={alpha}, init={init}, tag={tag}")
                       k_steph_search(
                            samples, graph, "BM25", k, k_step, alpha,
                            f"graphs/{file}", init, tag)
                       print(f"  [k_steph] reranker=graph_aware, alpha={alpha}, init={init}, tag={tag}")
                       k_steph_search(
                            samples, graph, "graph_aware", k, k_step, alpha,
                            f"graphs/{file}", init, tag
                        )
        for k in k_retrive:
          for init in inits:
            for tag, samples in sample_sets.items():
                print(f"  [k_steph] reranker= cross_encoder, init={init}, tag={tag}")
                k_steph_search(
                    samples, graph, "cross_encoder", k, k_step, 0.0, f"graphs/{file}", init, tag,
                )
        dt.save_leaderboard(f"Outputs/graphs/{file}", "leaderboards")
"""-------------------------------------------------------------Retrieval-------------------------------------------------------------"""



def knn_Graph(sampled_items, knn_params, preproccess, name, graph_params,save):
    # Building a graph using knn
    gc.build_knn_graph(sampled_items, knn_params, preproccess, name, graph_params, save)

    return


def mutual_knn(sampled_items, knn_params, preproccess, name, graph_params, save):
    # Building a graph using mutual knn
    gc.build_mutual_knn_graph(sampled_items, knn_params, preproccess, name, graph_params, save)
    return


def cluster_graph(sampled_items, kmeans_params, agglo_params, preprocess, knn_params, dbscan_params, name, clustering_result=None, flag=True,
                  algorithm="kmeans"):
    # Building a graph using clustering
    return gc.build_clustering_graph(sampled_items, kmeans_params, agglo_params, knn_params, dbscan_params, preprocess, name, algorithm, clustering_result, flag)
