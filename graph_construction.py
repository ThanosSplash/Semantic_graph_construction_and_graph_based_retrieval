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

def pipeline(data, scaler):
    pca = PCA()
    data_scaled = scaler.fit_transform(data)
    pca.fit(data_scaled)

    variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance)
    threshold_variance = 0.85
    n_components = np.argmax(cumulative_variance >= threshold_variance) + 1

    pca_plot = PCA(n_components=n_components)
    data_pca = pca_plot.fit_transform(data_scaled)
    return data_pca



def build_thershold_graph(corpus):
    """
    Building a
    :param corpus:
    :return:
    """
    G = nx.Graph()
    threshold = 0.2
    for id in corpus:
        G.add_node(id)
    for (id1, emb1), (id2, emb2) in combinations(corpus.items(), 2):
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        sim = cosine_similarity(emb1, emb2)

        if sim >= threshold:
            G.add_edge(id1, id2, weight=sim)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"betweenness:{nx.betweenness_centrality(G, normalized=False, endpoints=False)} | closeness: {nx.closeness_centrality(G)} | degree: {nx.degree_centrality(G)}")

    dt.save_graph(G, "Graph_Baseline.gpickle")
    plot_graph(G, "Baseline")


    return


def build_knn_graph(corpus):
    G = nx.Graph()
    k = 5

    for id in corpus:
        G.add_node(id)
    ids = list(corpus.keys())
    embeddings = np.array(list(corpus.values()))
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    data = pipeline(embeddings, scaler)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(data)
    distances, indices = nn.kneighbors(data)


    for i, neighbors in enumerate(indices):
        node_id = ids[i]
        for j, nearest in enumerate(neighbors):
            if i != nearest:
                G.add_edge(node_id, ids[nearest], weight=distances[i][j])

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"betweenness:{ nx.betweenness_centrality(G, normalized=False, endpoints=False)} | closeness: {nx.closeness_centrality(G)} | degree: { nx.degree_centrality(G)}")



    nx.degree_centrality(G)
    dt.save_graph(G, "Graph_KNN.gpickle")

    plot_graph(G, "KNN")



def build_mutual_knn_graph(corpus):
    G = nx.Graph()
    k = 5
    for id in corpus:
        G.add_node(id)

    ids = list(corpus.keys())
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    embeddings = np.array(list(corpus.values()))
    data = pipeline(embeddings, scaler)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

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




def build_clustering_graph(data, threshold_distance):

    unique_clusters, clustering_labels, ids = cl.kmeans_pipeline(data, threshold_distance)

    G = nx.Graph()
    for id in data:
        G.add_node(id)
    for cluster_id in range(len(unique_clusters)):
        cluster_nodes = [ids[i] for i, label in enumerate(clustering_labels) if label == cluster_id]

        for i, node_a in enumerate(cluster_nodes):
            for node_b in cluster_nodes[i + 1:]:  # avoid duplicate edges
                G.add_edge(node_a, node_b)

    plot_graph(G, "Kmeans")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"betweenness:{nx.betweenness_centrality(G, normalized=False, endpoints=False)} | closeness: {nx.closeness_centrality(G)} | degree: {nx.degree_centrality(G)}")

    return



def plot_graph(G, name):
    # Flatten any malformed edge weights before layout
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            w = data['weight']
            while hasattr(w, '__len__'):
                w = w[0]
            data['weight'] = float(w)

    pos = nx.spring_layout(G, seed=42, k=0.9)
    labels = nx.get_edge_attributes(G, 'label')
    plt.figure(figsize=(12, 10))

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')

    plt.title(name + ' Knowledge Graph')
    plt.axis('off')
    plt.show()



def plot_subgraph(G, nodes):
    """
    nodes: list of node names/ids you want to plot
    Example: plot_subgraph(G, ['node1', 'node2', 'node3'])
    """
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
