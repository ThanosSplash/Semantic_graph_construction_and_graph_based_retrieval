import embeddings as emb
import data_loading as dt
import graph_construction as gc
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
    prepare_samples()
    prepare_for_clustering()
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


def prepare_for_clustering(data, name):
    clustering.dendrogram(data, name)
    return


def prepare_samples():
    # Preparing samples for other experiments
    questions, answers, corpus = dt.load_data()

    sample_size = 15
    sampled_items = random.sample(list(corpus.items()), sample_size)
    sampled_dict = dict(sampled_items)
    dt.save_samples(sampled_dict, "15")
    prepare_for_clustering(sampled_dict, "dendrogram_15.png")

    sample_size = 100
    sampled_items = random.sample(list(corpus.items()), sample_size)
    sampled_dict = dict(sampled_items)
    dt.save_samples(sampled_dict, "100")
    prepare_for_clustering(sampled_dict, "dendrogram_100.png")

    sample_size = 1000
    sampled_items = random.sample(list(corpus.items()), sample_size)
    sampled_dict = dict(sampled_items)
    dt.save_samples(sampled_dict, "1000")
    prepare_for_clustering(sampled_dict, "dendrogram_1000.png")




def baseline_search(query_ids, k):
    rr_scores = []
    recallk_scores =[]
    dcg_scores = []
    for query_id in query_ids:
       recallk, rr , dcg= baseline(query_id, k)
       recallk_scores.append(recallk)
       rr_scores.append(rr)
       dcg_scores.append(dcg)
    print(f"Avg Recall@k score: {sum(recallk_scores)/len(recallk_scores)} | MRR score: {ev.MRR_score(rr_scores)} | Avg DCG@K: {sum(dcg_scores)/len(dcg_scores)}")
def baseline(query_id, k):
   # Given an list of ids using cosine similarity score find the top k most similar passages and then calcualting the metrics

   # Loading data
   q, a, c = dt.load_data()
   query = q[query_id]
   query_emb = query[0].reshape(1, -1)
   query_correct_results = query[1]
   pq = PriorityQueue()

   # For every passage calculating the cosine similarity score
   for document in c.keys():
       document_emb = c[document].reshape(1, -1)
       sim = cosine_similarity(query_emb, document_emb)
       pq.put((-float(sim), query_id, document))
   i = 0
   pred_results = []
   # Taking from the priority queue the passages with the highest score
   while i < k:
       neg_sim, query_id, document = pq.get()
       #print(f"query: {query_id} | doc: {document} | sim: {-neg_sim:.2f}")
       pred_results.append(document)
       i+=1

   # Evaluating the results
   recallk = ev.recallk_score(pred_results, query_correct_results, k)
   rr = ev.RR_score(pred_results, query_correct_results)
   dcg = ev.DCGk_score(pred_results, query_correct_results, k)
   return recallk, rr, dcg
def threshold_graph(sampled_items):
    # Building a threshold graph
    gc.build_thershold_graph(sampled_items)

    #G = dt.load_graph()
    #nodes_to_plot = list(G.nodes())[:100]
    #gc.plot_subgraph(G, nodes_to_plot)
    return

def knn_Graph(sampled_items):
    # Building a graph using knn
    gc.build_knn_graph(sampled_items)

    return


def mutual_knn(sampled_items):
    # Building a graph using mutual knn
    gc.build_mutual_knn_graph(sampled_items)
    return

def cluster_graph(sampled_items, threshold_distance):
    # Building a graph using clustering
    gc.build_clustering_graph(sampled_items, threshold_distance)