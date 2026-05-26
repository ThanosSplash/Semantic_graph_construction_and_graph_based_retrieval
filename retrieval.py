from sknetwork.ranking import PageRank
import data_loading as dt
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.sparse import csr_matrix
from queue import PriorityQueue
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict


def sim_scores_for_query(query_id):
    q, a, c = dt.load_data()
    query = q[query_id]
    query_emb = query[0].reshape(1, -1)
    query_correct_results = query[1]
    documents = list(c.keys())
    embeddings = np.vstack(list(c.values()))

    # All similarities at once
    sims = cosine_similarity(query_emb, embeddings).flatten()
    sims = np.maximum(sims, 0)
    return sims, documents, query_correct_results

def top_k(query_id, k):
   # Given an list of ids using cosine similarity score find the top k most similar passages and then calcualting the metrics
   # Loading data

   sims, documents, query_correct_results = sim_scores_for_query(query_id)

   # Top k indices
   top_k_idx = np.argsort(sims)[::-1][:k]

   pred_results = [documents[i] for i in top_k_idx]
   pred_sim = sims[top_k_idx].tolist()
   return pred_results, query_correct_results, pred_sim


def personalised_pagerank(graph, ids, similarities, k, nodes=None):
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
    if nodes is None:
        # Rank nodes by score and return top-k

        return [node for node, _ in sorted(node_scores, key=lambda x: x[1], reverse=True)[:k]]
    else:
        topk_ = sorted(node_scores, key=lambda x: x[1], reverse=True)
        top_nodes = {node: score for node, score in topk_ if node in nodes}
        return top_nodes


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


def k_step_neighborhood_expansion(graph, important_nodes, query_id, k, hops, alpha, sims):
    # Function that calculates the top-k nodes using k step neighborhood expansion
    neighbors = get_neighbors(graph, important_nodes, hops) # Finding the neighbors
    ids = list(neighbors.keys())
    # Calculating similarity scores
    sims, documents, query_correct_results = sim_scores_for_query(query_id)
    final_scores = []
    #graph_scores = pagerank(graph, 0, ids)
    # Calculating graph score for the important nodes using personalised pagerank
    graph_scores = personalised_pagerank(graph, important_nodes, sims, 0, ids)
    # Calculating total scores for each node
    for i, node_id in enumerate(ids):
        #graph_score = neighbors[node_id]
        graph_score = graph_scores[node_id]
        score = alpha * sims[i] + (1 - alpha) * graph_score
        final_scores.append((node_id, score))

    # Finding the top-k nodes
    final_scores.sort(key=lambda x: -x[1])
    top_k = final_scores[:k]
    pred_ids = [node_id for node_id, _ in top_k]
    pred_scores = [score for _, score in top_k]

    return pred_ids, pred_scores


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


def shortest_path(graph, query_id, important_nodes, k, alpha):

    # Function that finds the shortest paths and the nodes in that path between all the important nodes
    nodes_in_path = get_nodes_in_shortest_paths(graph, important_nodes)
    # Calculating similarity scores
    sims, documents, query_correct_results = sim_scores_for_query(query_id)
    final_scores = []
    # graph_scores = pagerank(graph, 0, ids)
    # Calculating graph score for the important nodes using personalised pagerank
    graph_scores = personalised_pagerank(graph, important_nodes, sims, k, nodes_in_path)
    for i, node_id in enumerate(nodes_in_path):
        # graph_score = neighbors[node_id]
        graph_score = graph_scores[node_id]
        score = alpha * sims[i] + (1 - alpha) * graph_score
        final_scores.append((node_id, score))

    # Finding the top-k nodes
    final_scores.sort(key=lambda x: -x[1])
    top_k = final_scores[:k]
    pred_ids = [node_id for node_id, _ in top_k]
    pred_scores = [score for _, score in top_k]

    return pred_ids, pred_scores



def get_nodes_in_shortest_paths(graph, nodes):
    all_path_nodes = set()
    # Finding the nodes that are connected
    components = list(nx.connected_components(graph))
    node_to_component = {}
    for i, comp in enumerate(nx.connected_components(graph)):
        for n in comp:
            node_to_component[n] = i

    # Calculating the shortest paths
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            s, t = nodes[i], nodes[j]

            if node_to_component[s] != node_to_component[t]:
                # if two nodes are not connected there is no shortest path
                continue
            path = nx.shortest_path(graph, source=s, target=t)

            for node in path:
                all_path_nodes.add(node)

    return list(all_path_nodes)