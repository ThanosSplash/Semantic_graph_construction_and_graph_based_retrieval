import networkx as nx
from itertools import combinations
import numpy as np
import data_loading as dt
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import clustering as cl




def build_threshold_graph(corpus):
    # Building a graph where the edges are formed if two passages (nodes) have cosine score higher than a threshold
    G = nx.Graph()
    threshold = 0.3
    for id in corpus:
        G.add_node(id)
    # Constructing the graph
    for (id1, emb1), (id2, emb2) in combinations(corpus.items(), 2):
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        sim = float(cosine_similarity(emb1, emb2))

        if sim >= threshold:
            G.add_edge(id1, id2, weight=sim)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"betweenness:{nx.betweenness_centrality(G, normalized=False, endpoints=False)} | closeness: {nx.closeness_centrality(G)} | degree: {nx.degree_centrality(G)}")

    dt.save_graph(G, "Graph_Baseline.gpickle")
    plot_graph(G, "Threshold")


    return G


def build_knn_graph(corpus):
    # Building a graph where the edges are formed using the knn algorithm
    G = nx.Graph()
    k = 5

    for id in corpus:
        G.add_node(id)
    ids = list(corpus.keys())
    embeddings = np.array(list(corpus.values()))
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    data = cl.pipeline(embeddings, scaler)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

    # Constructing the graph
    for i, neighbors in enumerate(indices):
        node_id = ids[i]
        for j, nearest in enumerate(neighbors):
            if i != nearest:
                G.add_edge(node_id, ids[nearest], weight=distances[i][j])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"betweenness:{ nx.betweenness_centrality(G, normalized=False, endpoints=False)} | closeness: {nx.closeness_centrality(G)} | degree: { nx.degree_centrality(G)}")




    dt.save_graph(G, "Graph_KNN.gpickle")

    plot_graph(G, "KNN")
    return G



def build_mutual_knn_graph(corpus):
    # Building a graph where the edges are formed using the mutual knn algorithm
    G = nx.Graph()
    k = 5
    for id in corpus:
        G.add_node(id)

    ids = list(corpus.keys())
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    embeddings = np.array(list(corpus.values()))
    data = cl.pipeline(embeddings, scaler)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(data)
    distances, indices = nn.kneighbors(data)
    # Constructing the graph
    d = {}
    for i, neighbors in enumerate(indices):
        node_id = ids[i]
        d[i] = (set(neighbors[1:]), distances[i][1:])
    for id in d.keys():
        for i, neighbor in enumerate(list(d[id][0])):
            if id < neighbor and id in d[neighbor][0]:  # mutual check
                G.add_edge(ids[id], ids[neighbor], weight=d[id][1][i])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"betweenness:{nx.betweenness_centrality(G, normalized=False, endpoints=False)} | closeness: {nx.closeness_centrality(G)} | degree: {nx.degree_centrality(G)}")
    plot_graph(G, "Mutual KNN")
    return G



def build_clustering_graph(data, threshold_distance):
    # Building multiples graphs using clustering
    unique_clusters, clustering_labels, ids, embeddings = cl.kmeans(data, threshold_distance)


    graphs_knn = []
    graphs_mutual_knn = []
    graphs_threshold = []
    # Constructing the graph
    for cluster_id in range(len(unique_clusters)):
        cluster_nodes = {
            ids[i]: embeddings[i]
            for i, label in enumerate(clustering_labels)
            if label == cluster_id
        }
        graphs_knn.append(build_knn_graph(cluster_nodes))
        graphs_mutual_knn.append(build_mutual_knn_graph(cluster_nodes))
        graphs_threshold.append(build_threshold_graph(cluster_nodes))




    knn_g = nx.compose_all(graphs_knn)
    mutual_knn_g = nx.compose_all(graphs_mutual_knn)
    threshold_g = nx.compose_all(graphs_threshold)
    plot_graph(knn_g, "Kmeans and KNN")
    plot_graph(mutual_knn_g, "Kmeans and mututal KNN")
    plot_graph(threshold_g, "Kmeans and threshold")
    return



def plot_graph(G, name):

    plt.figure(figsize=(12, 10))

    components = list(nx.connected_components(G))
    pos = {}

    # Spread components horizontally
    offset = 0
    spacing = 5  # increase for more separation

    for comp in components:
        subgraph = G.subgraph(comp)
        sub_pos = nx.spring_layout(subgraph, seed=42, k=0.9)

        # shift positions
        for node, (x, y) in sub_pos.items():
            pos[node] = (x + offset, y)

        offset += spacing

    labels = nx.get_edge_attributes(G, 'label')

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    plt.title(name + ' Knowledge Graph')
    plt.axis('off')
    plt.show()



def plot_subgraph(G, nodes):

    S = G.subgraph(nodes)

    # Sanitize edge weights
    S = S.copy()
    for u, v, data in S.edges(data=True):
        if 'weight' in data:
            w = data['weight']
            while hasattr(w, '__len__'):
                w = w[0]
            data['weight'] = float(w)

    pos = nx.spring_layout(S, seed=42, k=0.9)
    labels = nx.get_edge_attributes(S, 'label')
    plt.figure(figsize=(12, 10))

    nx.draw_networkx_nodes(S, pos, node_size=700, node_color='lightblue', alpha=0.6)
    nx.draw_networkx_edges(S, pos, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(S, pos, font_size=10)
    nx.draw_networkx_edge_labels(S, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')

    plt.title('Knowledge Subgraph')
    plt.axis('off')
    plt.show()

def plot_data(data, labels):
    plt.figure(figsize=(20, 10))

    # Plot each unique label with a different color
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(data[mask, 0], data[mask, 1],
                    label=f"Cluster {label}",
                    alpha=0.6,
                    s=10)
    plt.title("Data Clusters")
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
