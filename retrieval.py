from sknetwork.ranking import PageRank
import data_loading as dt
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.sparse import csr_matrix
from queue import PriorityQueue
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import CrossEncoder
from collections import defaultdict
from rank_bm25 import BM25Okapi
import transformers
import torch

"""-------------------------------------------------------------Help functions-------------------------------------------------------------"""

def normalize(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

def get_neighbors(graph, ids, hops, current_hop=1, visited=None,  hop_map=None):
    # Function that calculates the neighbors nodes. The exploration range depends on the hops

    if visited is None:
        visited = set(ids)
        hop_map = {node: 1 for node in ids}

    if hops <= 0:
        return hop_map

    next_ids = set()

    for node in ids:
        for neighbor in graph.neighbors(node):
            next_ids.add(neighbor)
            if neighbor not in hop_map:
                hop_map[neighbor] = 1/(current_hop+1)

    # remove already visited nodes
    next_ids -= visited

    visited.update(next_ids)

    return get_neighbors(graph, next_ids, hops - 1, current_hop + 1, visited, hop_map)


def get_nodes_in_shortest_paths(graph, nodes):
    all_path_nodes = set()

    for i in range(len(nodes)):
        all_path_nodes.add(nodes[i])
        for j in range(i + 1, len(nodes)):
            s, t = nodes[i], nodes[j]
            try:
                path = nx.shortest_path(graph, source=s, target=t)
                for node in path:
                    all_path_nodes.add(node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return list(all_path_nodes)

def ppr_graph_scores(graph, ids, similarities, nodes = None):
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
    node_scores = zip(graph.nodes(), scores_personal)
    node_scores = {node: score for node, score in node_scores if node in nodes}
    return node_scores


"""-------------------------------------------------------------Help functions-------------------------------------------------------------"""
"""-------------------------------------------------------------Rerankers-------------------------------------------------------------"""
def rerank_graph_aware(graph, query_id, ids, important_nodes, sims, alpha):
    sim_dict, query_correct_results = sim_scores_for_query(query_id, ids)
    final_scores = []
    # graph_scores = pagerank(graph, 0, ids)
    # Calculating graph score for the important nodes using personalised pagerank
    graph_scores = ppr_graph_scores(graph, important_nodes, sims, ids)
    # Calculating total scores for each node
    for i, node_id in enumerate(ids):
        # graph_score = neighbors[node_id]
        graph_score = graph_scores[node_id]
        score = alpha * sim_dict[node_id] + (1 - alpha) * graph_score
        final_scores.append((node_id, score))
    final_scores.sort(key=lambda x: -x[1])
    return final_scores, query_correct_results

def rerank_cross_encoder(query_id, ids):
    transformers.logging.set_verbosity_error()
    q_text, a_text, c_text = dt.load_texts()
    q, a, c = dt.load_data()
    query_emb = q[query_id]
    query_correct_results = query_emb[1]
    query_text = q_text[query_id]
    if torch.cuda.is_available():
         cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    else:
         cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query_text, c_text[id]) for id in ids]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    final_scores = sorted(
        zip(ids, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return final_scores, query_correct_results


def rerank_hybrid(query_id, ids, alpha=0.5):
    q, a, c = dt.load_data()
    q_text, a_text, c_text = dt.load_texts()
    query_emb = q[query_id]
    query_correct_results = query_emb[1]
    query_text = q_text[query_id]

    texts = [c_text[id][0] for id in ids]
    embs = [c[id][0] for id in ids]

    bm25 = BM25Okapi([t.split() for t in texts])
    bm25_scores = np.array(bm25.get_scores(query_text.split()))
    sim_dict, query_correct_results = sim_scores_for_query(query_id, ids)
    sem_scores = np.array([sim_dict[id] for id in ids])
    final_scores = []

    bm25_norm = normalize(bm25_scores)
    sem_norm = normalize(sem_scores)
    final_scores = [
        (node_id, alpha * sem_norm[i] + (1 - alpha) * bm25_norm[i])
        for i, node_id in enumerate(ids)
    ]
    final_scores.sort(key=lambda x: -x[1])


    return final_scores, query_correct_results
"""-------------------------------------------------------------Rerankers-------------------------------------------------------------"""

"""-------------------------------------------------------------Retrieval Techniques-------------------------------------------------------------"""

def sim_scores_for_query(query_id, ids=None):

    q, a, c = dt.load_data()
    query = q[query_id]
    query_emb = query[0].reshape(1, -1)
    query_correct_results = query[1]
    if ids is None:
        documents = list(c.keys())
        embeddings = np.vstack(list(c.values()))
    else:
        documents = list(ids)
        embeddings = np.vstack([c[id] for id in ids if id in c.keys()])

    # All similarities at once
    sims = cosine_similarity(query_emb, embeddings).flatten()
    sims = np.maximum(sims, 0)
    sim_dict = dict(zip(documents, sims))
    return sim_dict, query_correct_results

def top_k(query_id, k):
   # Given an list of ids using cosine similarity score find the top k most similar passages and then calcualting the metrics
   # Loading data

   sim_dict, query_correct_results = sim_scores_for_query(query_id)

   # Sort by similarity descending
   sorted_docs = sorted(
       sim_dict.items(),
       key=lambda x: x[1],
       reverse=True
   )

   top_k_docs = sorted_docs[:k]

   pred_results = [doc for doc, sim in top_k_docs]
   pred_sim = [sim for doc, sim in top_k_docs]

   return pred_results, query_correct_results, pred_sim


def personalised_pagerank(graph, ids, similarities, k, query_id, alpha):
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
    graph_scores = dict(zip(graph.nodes(), scores_personal))
    sim_dict, query_correct_results = sim_scores_for_query(query_id)
    final_scores = []
    # graph_scores = pagerank(graph, 0, ids)
    # Calculating graph score for the important nodes using personalised pagerank
    # Calculating total scores for each node
    for i, node_id in enumerate(ids):
        # graph_score = neighbors[node_id]
        #graph_score = graph_scores[node_id]
        score = alpha * sim_dict[node_id] + (1 - alpha) * graph_scores[node_id]
        final_scores.append((node_id, score))
    final_scores.sort(key=lambda x: -x[1])
    return [node for node, _ in final_scores[:k]]






def pagerank(graph, k, nodes=None):
    # Function that calculates the top-k nodes using pagerank
    adjacency = csr_matrix(nx.to_scipy_sparse_array(graph, format='csr'))
    # Running Pagerank algorithm
    pagerank_ = PageRank()
    scores = pagerank_.fit_predict(adjacency)
    node_labels = list(graph.nodes())
    node_scores = tuple(zip(node_labels, scores))
    if nodes is None:
        # Rank nodes by score and return top-k
        topk_ = sorted(node_scores, key=lambda x: x[1], reverse=True)[:k]
        top_nodes= [node for node, score in topk_]
        return top_nodes
    else:
        topk_ = sorted(node_scores, key=lambda x: x[1], reverse=True)
        top_nodes = {node: score for node, score in topk_ if node in nodes}
        return top_nodes



def k_step_neighborhood_expansion(graph, important_nodes, query_id, k, hops, alpha, sims, reranker_type):
    # Function that calculates the top-k nodes using k step neighborhood expansion
    neighbors = get_neighbors(graph, important_nodes, hops) # Finding the neighbors
    ids = list(neighbors.keys())
    # Calculating similarity scores
    final_scores = []
    if reranker_type == "graph_aware":
        final_scores, query_correct_results = rerank_graph_aware(graph, query_id, ids, important_nodes, sims, alpha)
    elif reranker_type == "cross_encoder":
        final_scores, query_correct_results = rerank_cross_encoder(query_id, ids)
    elif reranker_type == "BM25":
        final_scores, query_correct_results = rerank_hybrid(query_id, ids, alpha)
    top_k = final_scores[:k]
    pred_ids = [node_id for node_id, _ in top_k]
    pred_scores = [score for _, score in top_k]

    return pred_ids, pred_scores





def hits(graph, root_set, k, max_iter=300):

    base_set = set(root_set)
    D = graph.to_directed()
    for node in root_set:
        if node in graph:
            base_set.update(D.successors(node))
            base_set.update(D.predecessors(node))

    H = D.subgraph(base_set).copy()

    hubs, auth = nx.hits(H, max_iter=max_iter, normalized=True)
    alpha = 0.5
    score = {
        n: alpha * auth[n] + (1 - alpha) * hubs[n]
        for n in hubs
    }

    top = sorted(score.items(), key=lambda x: -x[1])

    return [n for n, _ in top[:k]]


def shortest_path(graph, query_id, important_nodes, k, alpha, sims, reranker_type):

    # Function that finds the shortest paths and the nodes in that path between all the important nodes
    nodes_in_path = get_nodes_in_shortest_paths(graph, important_nodes)
    # Calculating similarity scores
    sim_dict, query_correct_results = sim_scores_for_query(query_id, nodes_in_path)
    final_scores = []
    # graph_scores = pagerank(graph, 0, ids)
    # Calculating graph score for the important nodes using personalised pagerank
    final_scores = []
    if reranker_type == "graph_aware":
        final_scores, query_correct_results = rerank_graph_aware(graph, query_id, nodes_in_path, important_nodes, sims, alpha)
    elif reranker_type == "cross_encoder":
        final_scores, query_correct_results = rerank_cross_encoder(query_id, nodes_in_path)
    elif reranker_type == "BM25":
        final_scores, query_correct_results = rerank_hybrid(query_id, nodes_in_path, alpha)
    top_k = final_scores[:k]
    pred_ids = [node_id for node_id, _ in top_k]
    pred_scores = [score for _, score in top_k]

    return pred_ids, pred_scores

"""-------------------------------------------------------------Retrieval Techniques-------------------------------------------------------------"""

