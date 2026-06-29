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
import time

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


def ppr_for_given_nodes(graph, init_nodes, sims, retrieved_ids):
    """Given the retrieved ids using the personalised pagerank algorithm
       to calculate the graph score for them
    graph: The semantic graph
    init_nodes: The node to init personalised pagerank algorithm
    sims: Similarities for the personalised pagerank algorithm
    retrieved_ids: The ids that the retriever method retrieved
    """
    # Making the adjacency matrix and calculating the weights for the init nodes
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    adjacency = csr_matrix(nx.to_scipy_sparse_array(graph, format='csr'))
    # Safety check
    valid = [(node_id, sims[i]) for i, node_id in enumerate(init_nodes) if node_id in node_to_idx]
    total = sum(score for _, score in valid)
    if total == 0 or len(valid) == 0:
        raise Exception("Zero valid ids")

    weights = {
        node_to_idx[node_id]: sim / total
        for node_id, sim in valid
        if node_id in node_to_idx
    }

    # Running Pagerank algorithm
    pagerank = PageRank()
    # Calculating the graph score for all the nodes
    scores_personal = pagerank.fit_predict(adjacency, weights)
    # Choosing the nodes in the retrieved_ids
    node_scores = zip(graph.nodes(), scores_personal)
    node_scores = {node: score for node, score in node_scores if node in retrieved_ids}
    return node_scores

def sim_scores_for_query(query_id, retrieved_ids=None):

    q, _, c = dt.load_data()
    # Safety check
    if query_id not in q:
        raise Exception(f"Id not found {query_id}")
    query = q[query_id]
    query_emb = query[0].reshape(1, -1)
    query_correct_results = query[1]
    if retrieved_ids is None:
        documents = list(c.keys())
        embeddings = np.vstack(list(c.values()))
    else:
        documents = list(retrieved_ids)
        embeddings = np.vstack([c[id] for id in retrieved_ids if id in c.keys()])

    # All similarities at once
    sims = cosine_similarity(query_emb, embeddings).flatten()
    sims = np.maximum(sims, 0)
    sim_dict = dict(zip(documents, sims))

    return sim_dict, query_correct_results

"""-------------------------------------------------------------Help functions-------------------------------------------------------------"""
"""-------------------------------------------------------------Rerankers-------------------------------------------------------------"""
def rerank_graph_aware(graph, query_id, retrieved_ids, init_nodes, sims, alpha):
    """Given the retrieved ids using the similarity score and personalised pagerank score (graph score)
       to make a new rankings for them
       graph: The semantic graph
       query_id: The id of the query
       retrieved_ids: The ids that the retriever method retrieved
       init_nodes: The node to init personalised pagerank algorithm
       sims: Similarities for the personalised pagerank algorithm
       alpha: Variable used to calculate the final score
    """
    # Calculating cosine sim
    sim_dict, query_correct_results = sim_scores_for_query(query_id, retrieved_ids)
    final_scores = []

    # Calculating graph score using personalised pagerank
    graph_scores = ppr_for_given_nodes(graph, init_nodes, sims, retrieved_ids)
    # Calculating total scores for each node
    for i, node_id in enumerate(retrieved_ids):
        # graph_score = neighbors[node_id]
        graph_score = graph_scores[node_id]
        score = alpha * sim_dict[node_id] + (1 - alpha) * graph_score
        final_scores.append((node_id, score))

    final_scores.sort(key=lambda x: -x[1])
    return final_scores

def rerank_cross_encoder(query_id, retrieved_ids):
    """Given the retrieved ids using the cross encoder function to make a new rankings for them
           query_id: The id of the query
           retrieved_ids: The ids that the retriever method retrieved
    """
    transformers.logging.set_verbosity_error()
    # Loading the texts for the queries and the corpus
    q_text, _, c_text = dt.load_texts()
    # Safety check
    if query_id not in q_text:
        raise Exception(f"Id not found {query_id}")
    query_text = q_text[query_id]
    # Checking if cude exists
    if torch.cuda.is_available():
         cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    else:
         cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Making pairs for the cross encoder model
    pairs = [(query_text, c_text[id]) for id in retrieved_ids]

    scores = cross_encoder.predict(pairs, show_progress_bar=False)

    final_scores = sorted(
        zip(retrieved_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return final_scores


def rerank_bm25(query_id, retrieved_ids, alpha=0.5):
    """Given the retrieved ids using the BM25 function to make a new rankings for them
               query_id: The id of the query
               retrieved_ids: The ids that the retriever method retrieved
               alpha: Variable used to calculate the final score
    """
    # Loading the texts for the queries and the corpus
    q_text, _, c_text = dt.load_texts()
    # Safety check
    if query_id not in q_text:
        raise Exception(f"Id not found {query_id}")
    query_text = q_text[query_id]

    # Gathering the texts of the retrieved data
    texts = [c_text[id] for id in retrieved_ids]
    # Calculating the similarities and bm25 scores
    bm25 = BM25Okapi([t.split() for t in texts])
    bm25_scores = np.array(bm25.get_scores(query_text.split()))
    sim_dict, query_correct_results = sim_scores_for_query(query_id, retrieved_ids)
    sim_scores = np.array([sim_dict[id] for id in retrieved_ids])
    # Normalise the scores
    bm25_norm = normalize(bm25_scores)
    sim_norm = normalize(sim_scores)

    final_scores = [
        (node_id, alpha * sim_norm[i] + (1 - alpha) * bm25_norm[i])
        for i, node_id in enumerate(retrieved_ids)
    ]
    final_scores.sort(key=lambda x: -x[1])


    return final_scores
"""-------------------------------------------------------------Rerankers-------------------------------------------------------------"""

"""-------------------------------------------------------------Retrieval Techniques-------------------------------------------------------------"""


def top_k(query_id, k):
   """Given a query and the number of the documents to be retrieved,
      retrieve the topk documents using cosine similarity
      query_id : The id of the query
      k: number of items to be retrieved
   """
   # Calculating similarities
   sim_dict, query_correct_results = sim_scores_for_query(query_id)

   # Sort by similarity descending
   sorted_docs = sorted(
       sim_dict.items(),
       key=lambda x: x[1],
       reverse=True
   )
   # Get the topk and making two lists for the ids and the similarity score
   top_k_docs = sorted_docs[:k]
   pred_results = [doc for doc, sim in top_k_docs]
   pred_sim = [sim for doc, sim in top_k_docs]

   return pred_results, query_correct_results, pred_sim


def personalised_pagerank(graph, init_nodes, sims, k, query_id, alpha):
    """Given the init ids using the personalised pagerank algorithm
        to calculate the graph score for every node in the graph
        retrieve the topk nodes with the highest score
    graph: The semantic graph
    init_nodes: The node to init personalised pagerank algorithm
    sims: Similarities for the personalised pagerank algorithm
    retrieved_ids: The ids that the retriever method retrieved
    alpha: Variable used to calculate the final score
    k: number of items to be retrieved
    """
    # Making the adjacency matrix and calculating the weights for the init nodes
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    adjacency = csr_matrix(nx.to_scipy_sparse_array(graph, format='csr'))
    # Safety check
    valid = [(node_id, sims[i]) for i, node_id in enumerate(init_nodes) if node_id in node_to_idx]
    total = sum(score for _, score in valid)
    if total == 0 or len(valid) == 0:
        raise Exception("Zero valid ids")

   # Build personalization vector: normalize similarities as seed weights
    weights = {
        node_to_idx[node_id]: sim / total
        for node_id, sim in valid
    }

    # Running Pagerank algorithm
    pagerank = PageRank()

    # Calculating the graph score and similarity score for all the nodes
    scores_personal = pagerank.fit_predict(adjacency, weights)
    graph_scores = dict(zip(graph.nodes(), scores_personal))
    sim_dict, query_correct_results = sim_scores_for_query(query_id)
    final_scores = []
    for node_id in graph.nodes:
        score = alpha * sim_dict[node_id] + (1 - alpha) * graph_scores[node_id]
        final_scores.append((node_id, score))
    # Sorting based on final score and taking the topk
    final_scores.sort(key=lambda x: -x[1])
    return [node for node, _ in final_scores[:k]]


def k_step_neighborhood_expansion(graph, init_nodes, query_id, k, hops, alpha, sims, reranker_type):
    """Given the init ids and the query id retrieve the topk items using three different types of
       reranker function
        graph: The semantic graph
        init_nodes: The node to init the k steph algorithm
        sims: Similarities for the personalised pagerank algorithm
        reranker_type: Name of the reranker function to use
        hops: For the get_neighbors function. How far to expand through the graph
        alpha: Variable used to calculate the final score
        k: number of items to be retrieved
        """
    # Finding the neighbors
    neighbors = get_neighbors(graph, init_nodes, hops) # Finding the neighbors
    ids = list(neighbors.keys())
    if len(ids) <= 0:
        raise Exception(f"Neighbors not found {ids}")
    # Calculating similarity scores
    final_scores = []
    # Using a reranked function
    if reranker_type == "graph_aware":
        final_scores = rerank_graph_aware(graph, query_id, ids, init_nodes, sims, alpha)
    elif reranker_type == "cross_encoder":
        final_scores = rerank_cross_encoder(query_id, ids)
    elif reranker_type == "BM25":
        final_scores = rerank_bm25(query_id, ids, alpha)
    # Get the topk and making two lists for the ids and the  score
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
        final_scores = rerank_graph_aware(graph, query_id, nodes_in_path, important_nodes, sims, alpha)
    elif reranker_type == "cross_encoder":
        final_scores = rerank_cross_encoder(query_id, nodes_in_path)
    elif reranker_type == "BM25":
        final_scores = rerank_bm25(query_id, nodes_in_path, alpha)
    top_k = final_scores[:k]
    pred_ids = [node_id for node_id, _ in top_k]
    pred_scores = [score for _, score in top_k]

    return pred_ids, pred_scores

"""-------------------------------------------------------------Retrieval Techniques-------------------------------------------------------------"""

