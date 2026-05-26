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
def prepare_all():
    prepare_dataset()
    return


def prepare_dataset():
   # Loading the dataset and calculating the embeddings and then save them

    # Loading data from the dataset bioasq
    bioasq, bioasq_corpus = dt.read_dataset_bioasq("Datasets/rag-mini-bioasq/")

    # Converting bioasq, bioasq_corpus dataframes to three dictionaries
    questions, answers = emb.make_embeddings(bioasq)
    corpus = emb.make_embeddings_corpus(bioasq_corpus)

    # Saving the dictionaries questions, answers, corpus in a binary file
    dt.save_data(questions, answers, corpus)



def baseline_search(query_ids, k, results_file):
    # Function that retrieves for each query the top-k most similar data and evaluates the results
    rr_scores = []
    recallk_scores =[]
    ndcg_scores = []
    avg_precisions = []
    for query_id in query_ids:
       # For each query calculates the top-k
       predictions, correct, _ = rt.top_k(query_id, k)
       #print(predictions)
       # Evaluating the results
       recallk_scores.append(ev.recallk_score(predictions, correct, k))
       rr_scores.append(ev.RR_score(predictions, correct))
       ndcg_scores.append(ev.nDCGk_score(predictions, correct, k))
       avg_precisions.append(ev.avg_precision(predictions, correct, k))
    # Printing and saving the results
    recallk = sum(recallk_scores) / len(recallk_scores)
    mrr = ev.MRR_score(rr_scores)
    ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mapk = sum(avg_precisions) / len(avg_precisions)
    print(
        f"Baseline| Avg Recall@k score: {recallk:.4f} | MRR score: {mrr:.4f} "
        f"| Avg DCG@K: {ndcg:.4f} |  | MAP@K score: {mapk :.4f}")
    dt.save_result("Top-"+str(k), recallk, mrr, ndcg, mapk, k, results_file)
def personalised_pagerank_search(query_ids, graph, graph_type, k, results_file, init):
    # Function that retrieves data from a graph using personalised pagerank and evaluates the results
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    for query_id in query_ids:
        # For each query calculates the top-k
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        # Running personalised pagerank
        predictions = rt.personalised_pagerank(graph, imporant_nodes, sims, k)
        #print(predictions)
        # Evaluating the results
        recallk_scores.append(ev.recallk_score(predictions, correct, k))
        rr_scores.append(ev.RR_score(predictions, correct))
        ndcg_scores.append(ev.nDCGk_score(predictions, correct, k))
        avg_precisions.append(ev.avg_precision(predictions, correct, k))
    # Printing and saving the results
    recallk = sum(recallk_scores) / len(recallk_scores)
    mrr = ev.MRR_score(rr_scores)
    ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mapk = sum(avg_precisions) / len(avg_precisions)
    print(
        f"PPR| Avg Recall@k score: {recallk:.4f} | MRR score: {mrr:.4f} "
        f"| Avg DCG@K: {ndcg:.4f} |  | MAP@K score: {mapk :.4f}")
    dt.save_result(f"PPR init = {str(init)}", recallk, mrr, ndcg, mapk, k, results_file)

    return
def k_steph_search(query_ids, graph, graph_type, k, hops, alpha, results_file, init):
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        predictions, scores = rt.k_step_neighborhood_expansion(graph, imporant_nodes, query_id, k, hops, alpha, sims)
        #print(predictions)
        #print(correct)
        recallk_scores.append(ev.recallk_score(predictions, correct, k))
        rr_scores.append(ev.RR_score(predictions, correct))
        ndcg_scores.append(ev.nDCGk_score(predictions, correct,k))
        avg_precisions.append(ev.avg_precision(predictions, correct, k))

    recallk = sum(recallk_scores) / len(recallk_scores)
    mrr = ev.MRR_score(rr_scores)
    ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mapk = sum(avg_precisions) / len(avg_precisions)
    print(
        f"K-steph| Avg Recall@k score: {recallk:.4f} | MRR score: {mrr:.4f} "
        f"| Avg DCG@K: {ndcg:.4f} |  | MAP@K score: {mapk :.4f}")

    dt.save_result(f"k_step_search init = {str(init)}  a = {alpha}", recallk, mrr, ndcg, mapk, k, results_file)
    return

def hits_search(query_ids, graph, graph_type, k, alpha, results_file, init):
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        predictions = rt.hits(graph, imporant_nodes, k)
        #print(predictions)
        recallk_scores.append(ev.recallk_score(predictions, correct, k))
        rr_scores.append(ev.RR_score(predictions, correct))
        ndcg_scores.append(ev.nDCGk_score(predictions, correct,k))
        avg_precisions.append(ev.avg_precision(predictions, correct, k))

    recallk = sum(recallk_scores) / len(recallk_scores)
    mrr = ev.MRR_score(rr_scores)
    ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mapk = sum(avg_precisions) / len(avg_precisions)
    print(
        f" Hits| Avg Recall@k score: {recallk:.4f} | MRR score: {mrr:.4f} "
        f"| Avg DCG@K: {ndcg:.4f} |  | MAP@K score: {mapk :.4f}")
    dt.save_result(f"hits init = {str(init)}  a = {alpha}", recallk, mrr, ndcg, mapk, k, results_file)
    return

def shortest_path_search(query_ids, graph, graph_type, k, alpha, results_file, init):
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    for query_id in query_ids:
        imporant_nodes, correct, sims = rt.top_k(query_id, init)
        predictions, _ = rt.shortest_path(graph, query_id,imporant_nodes, k, alpha)
        #print(predictions)
        recallk_scores.append(ev.recallk_score(predictions, correct, k))
        rr_scores.append(ev.RR_score(predictions, correct))
        ndcg_scores.append(ev.nDCGk_score(predictions, correct,k))
        avg_precisions.append(ev.avg_precision(predictions, correct, k))

    recallk = sum(recallk_scores) / len(recallk_scores)
    mrr = ev.MRR_score(rr_scores)
    ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mapk = sum(avg_precisions) / len(avg_precisions)
    print(
        f" Shortest Path| Avg Recall@k score: {recallk:.4f} | MRR score: {mrr:.4f} "
        f"| Avg DCG@K: {ndcg:.4f} |  | MAP@K score: {mapk :.4f}")
    dt.save_result(f"shortest_path init = {str(init)}  a = {alpha}", recallk, mrr, ndcg, mapk, k, results_file)
    return

def threshold_graph(sampled_items, threshold_params, preprocess, name, save):
    # Building a threshold graph
    gc.build_threshold_graph(sampled_items, threshold_params, preprocess, name, save)

    #G = dt.load_graph()
    #nodes_to_plot = list(G.nodes())[:100]
    #gc.plot_subgraph(G, nodes_to_plot)
    return

def knn_Graph(sampled_items, knn_params, preproccess, name, save):
    # Building a graph using knn
    gc.build_knn_graph(sampled_items, knn_params, preproccess, name, save)

    return


def mutual_knn(sampled_items, knn_params, preproccess, name, save):
    # Building a graph using mutual knn
    gc.build_mutual_knn_graph(sampled_items, knn_params, preproccess, name, save)
    return


def cluster_graph(sampled_items, kmeans_params, agglo_params, preprocess, knn_params, dbscan_params, name, clustering_result=None, flag=True,
                  algorithm="kmeans"):
    # Building a graph using clustering
    return gc.build_clustering_graph(sampled_items, kmeans_params, agglo_params, knn_params, dbscan_params, preprocess, name, algorithm, clustering_result, flag)
