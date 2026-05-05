from sknetwork.ranking import PageRank
import data_loading as dt
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.sparse import csr_matrix
from queue import PriorityQueue
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def top_k(query_id, k):
   # Given an list of ids using cosine similarity score find the top k most similar passages and then calcualting the metrics
   # Loading data
   q, a, c = dt.load_data()
   query = q[query_id]
   query_emb = query[0].reshape(1, -1)
   query_correct_results = query[1]
   documents = list(c.keys())
   embeddings = np.vstack(list(c.values()))

   # All similarities at once
   sims = cosine_similarity(query_emb, embeddings).flatten()

   # Top k indices
   top_k_idx = np.argsort(sims)[::-1][:k]

   pred_results = [documents[i] for i in top_k_idx]
   pred_sim = sims[top_k_idx].tolist()
   return pred_results, query_correct_results, pred_sim


def personalised_pagerank(graph, ids, similarities, k):
    # Function that calculates the top-k nodes using personalised pagerank
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    adjacency = csr_matrix(nx.to_scipy_sparse_array(graph, format='csr'))

    # Build personalization vector: normalize similarities as seed weights
    total = sum(similarities)
    weights = {
        node_to_idx[node_id]: similarities[i] / total
        for i, node_id in enumerate(ids)
        if node_id in node_to_idx
    }

    # Running Pagerank algorithm
    pagerank = PageRank()
    scores_personal = pagerank.fit_predict(adjacency, weights)

    # Rank nodes by score and return top-k
    node_scores = zip(graph.nodes(), scores_personal)
    return [node for node, _ in sorted(node_scores, key=lambda x: x[1], reverse=True)[:k]]


def pagerank(graph, k):
    # Function that calculates the top-k nodes using pagerank
    adjacency = csr_matrix(nx.to_scipy_sparse_array(graph, format='csr'))
    # Running Pagerank algorithm
    pagerank_ = PageRank()
    scores = pagerank_.fit_predict(adjacency)

    # Rank nodes by score and return top-k
    node_labels = list(graph.nodes())
    node_scores = tuple(zip(node_labels, scores))
    topk_ = sorted(node_scores, key=lambda x: x[1], reverse=True)[:k]
    top_nodes= [node for node, score in topk_]
    return top_nodes
