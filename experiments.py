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
        f"Avg Recall@k score: {recallk:.4f} | MRR score: {mrr:.4f} "
        f"| Avg DCG@K: {ndcg:.4f} |  | MAP@K score: {mapk :.4f}")
    dt.save_result("Top-"+str(k), recallk, mrr, ndcg, mapk, k, results_file)
def personalised_pagerank_search(query_ids, graph, graph_type, k, results_file):
    # Function that retrieves data from a graph using personalised pagerank and evaluates the results
    rr_scores = []
    recallk_scores = []
    ndcg_scores = []
    avg_precisions = []
    for query_id in query_ids:
        # For each query calculates the top-k
        imporant_nodes, correct, sims = rt.top_k(query_id, k)
        # Running personalised pagerank
        predictions = rt.personalised_pagerank(graph, imporant_nodes, sims, k)
        # Evaluating the results
        recallk_scores.append(ev.recallk_score(predictions, correct, k))
        rr_scores.append(ev.RR_score(predictions, correct))
        ndcg_scores.append(ev.nDCGk_score(predictions, correct, k))
        avg_precisions.append(ev.avg_precision(imporant_nodes, correct, k))
    # Printing and saving the results
    recallk = sum(recallk_scores) / len(recallk_scores)
    mrr = ev.MRR_score(rr_scores)
    ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mapk = sum(avg_precisions) / len(avg_precisions)
    print(
        f"Avg Recall@k score: {recallk:.4f} | MRR score: {mrr:.4f} "
        f"| Avg DCG@K: {ndcg:.4f} |  | MAP@K score: {mapk :.4f}")
    dt.save_result(graph_type + " + PPR ", recallk, mrr, ndcg, mapk, k, results_file)

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


def cluster_graph(sampled_items, kmeans_params, threshold_params, agglo_params, preprocess, knn_params, dbscan_params,
                  algorithm="kmeans"):
    # Building a graph using clustering
    gc.build_clustering_graph(sampled_items, kmeans_params, threshold_params, agglo_params, knn_params, dbscan_params, preprocess, algorithm)
