import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np
import networkx as nx
from networkx.algorithms import community
import data_loading as dt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import retrieval as rt
import graph_construction as gc
import nx_parallel
import igraph as ig

def plot_cluster_with_silhouette(data, data_2d, centers, n_clusters, clustering_labels):
    # Function that plots the data and the clusters in 2d and also the silhouette score of each cluster
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])
    if n_clusters > 1:
       silhouette_avg = silhouette_score(data, clustering_labels)
       sample_silhouette_values = silhouette_samples(data, clustering_labels)
    else:
       silhouette_avg = 0.0
       sample_silhouette_values = sample_silhouette_values = np.zeros(len(data))
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clustering_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(clustering_labels.astype(float) / n_clusters)
    ax2.scatter(
        data_2d[:, 0], data_2d[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

    return fig


def plot_k_distance_graph(X, k_values):
    plt.figure(figsize=(12, 7))

    for k in k_values:
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)

        distances, _ = neigh.kneighbors(X)

        # k-th nearest neighbor distances
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)

        plt.plot(k_distances, label=f"k={k}")

    plt.xlabel('Points (sorted)')
    plt.ylabel('k-th nearest neighbor distance')
    plt.title('K-distance Graph (multiple k)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_graph(G, name):
    # Function that plots the entire given graph
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

    plt.title(name + ' Semantic Graph')
    plt.axis('off')

def plot_graph_stats(info_knn,  info_mutual_knn, clustering = "", clusters = 0):
    n_neighbors_vals = [x[0] for x in info_knn]
    n_neighbors_vals_mutual_knn = [x[0] for x in info_mutual_knn]

    n_components_vals = [x[1] for x in info_knn]
    n_components_vals_mutual_knn = [x[1] for x in info_mutual_knn]

    n_communities_vals = [x[2] for x in info_knn]
    n_communities_vals_mutual_knn = [x[2] for x in info_mutual_knn]

    density_vals = [x[3] for x in info_knn]
    density_vals_mutual_knn = [x[3] for x in info_mutual_knn]

    avg_degree_vals = [x[4] for x in info_knn]
    avg_degree_vals_mutual_knn = [x[4] for x in info_mutual_knn]

    metrics = [
        (n_components_vals, n_components_vals_mutual_knn, "Number of Connected Components"),
        (n_communities_vals, n_communities_vals_mutual_knn, "Number of Communities"),
        (density_vals, density_vals_mutual_knn, "Graph Density"),
        (avg_degree_vals, avg_degree_vals_mutual_knn, "Average Degree"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for ax, (values, values_mutual, label) in zip(axes, metrics):
        ax.plot(n_neighbors_vals, values, marker='o', color='g', label="knn")
        ax.plot(n_neighbors_vals_mutual_knn, values_mutual, marker='o', color='r', label="mutual knn")
        ax.set_xlabel("Number of Neighbors (k)")
        ax.set_ylabel(label)
        ax.legend()
        if clustering == "kmeans":
         ax.set_title(f"{label} Clustering method {clustering}, {clusters} (kNN graph) vs (mutual kNN graph)")
        else:
         ax.set_title(f"{label} (kNN graph) vs (mutual kNN graph)")
        ax.grid(False)

    plt.tight_layout()
    plt.show()

def getting_plot_graph_stats():
    files = dt.get_files()
    info_to_plot_knn = []
    info_to_plot_mutual_knn = []
    info_to_plot_clustering_knn = {}
    info_to_plot_clustering_mutual_knn = {}

    for file in files:
        params, info = dt.load_graph_parameters(file)
        if params['graph_type'] == "knn graph":
            info_to_plot_knn.append((
                params['knn']['n_neighbors'],
                info['Number of connected components'],
                info['Number of communities'][0],
                info['Graph Density'],
                info['Average Degree'],
            ))
        elif params['graph_type'] == "mutual knn graph":
            info_to_plot_mutual_knn.append((
                params['knn']['n_neighbors'],
                info['Number of connected components'],
                info['Number of communities'][0],
                info['Graph Density'],
                info['Average Degree'],))
        elif params['graph_type'] == "clustering knn graph":
            if params['clustering']['n_clusters'] not in info_to_plot_clustering_knn:
                info_to_plot_clustering_knn[params['clustering']['n_clusters']] = [(
                    params['Graph building algorithm params']['n_neighbors'],
                    info['Number of connected components'],
                    info['Number of communities'][0],
                    info['Graph Density'],
                    info['Average Degree'],
                )]
            else:
                info_to_plot_clustering_knn[params['clustering']['n_clusters']].append((
                    params['Graph building algorithm params']['n_neighbors'],
                    info['Number of connected components'],
                    info['Number of communities'][0],
                    info['Graph Density'],
                    info['Average Degree'],
                ))
        elif params['graph_type'] == "clustering mutual knn graph":
            if params['clustering']['n_clusters'] not in info_to_plot_clustering_mutual_knn:
                info_to_plot_clustering_mutual_knn[params['clustering']['n_clusters']] = [(
                    params['Building graph algorithm params']['n_neighbors'],
                    info['Number of connected components'],
                    info['Number of communities'][0],
                    info['Graph Density'],
                    info['Average Degree'],
                )]
            else:
                info_to_plot_clustering_mutual_knn[params['clustering']['n_clusters']].append((
                    params['Building graph algorithm params']['n_neighbors'],
                    info['Number of connected components'],
                    info['Number of communities'][0],
                    info['Graph Density'],
                    info['Average Degree'],
                ))

    info_to_plot_knn.sort(key=lambda x: x[0])
    info_to_plot_mutual_knn.sort(key=lambda x: x[0])
    #info_to_plot_clustering_knn.sort(key=lambda x: x[0])
    #info_to_plot_clustering_mutual_knn.sort(key=lambda x: x[0])
    plot_graph_stats(info_to_plot_knn, info_to_plot_mutual_knn)
    print(info_to_plot_clustering_knn.keys())
    for cl in info_to_plot_clustering_knn.keys():
      test_knn = info_to_plot_clustering_knn[cl]
      test_knn.sort(key=lambda x: x[0])
      test_mutual_knn = info_to_plot_clustering_mutual_knn[cl]
      test_mutual_knn.sort(key=lambda x: x[0])
      plot_graph_stats(test_knn, test_mutual_knn, "kmeans", cl)

def eval_re(eval_results, sample_type):
    dataset = [(entry['final scores']['recallk'], entry['final scores']['mrr'],
                entry['final scores']['ndcg'], entry['final scores']['mapk'], entry['k'], entry['alpha'], entry['init']) for entry in eval_results if entry['sample_type'] == sample_type]
    dataset.sort(key=lambda x: x[0], reverse=True)
    return dataset
def plot_eval_results(info_knn,  info_mutual_knn, sample_type):

    recall_vals =  [x[0][0][0] for x in info_knn[sample_type]]
    recall_vals_mutual_knn = [x[0][0][0] for x in info_mutual_knn[sample_type]]

    mrr_vals = [x[0][0][1] for x in  info_knn[sample_type]]
    mrr_vals_mutual_knn = [x[0][0][1] for x in info_mutual_knn[sample_type]]

    ndcg_vals = [x[0][0][2] for x in info_knn[sample_type]]
    ndcg_vals_mutual_knn = [x[0][0][2] for x in info_mutual_knn[sample_type]]

    map_vals = [x[0][0][3] for x in info_knn[sample_type]]
    map_vals_mutual_knn = [x[0][0][3] for x in info_mutual_knn[sample_type]]

    k_vals = [x[1]['knn']['n_neighbors'] for x in info_knn[sample_type]]
    k_vals_mutual_knn = [x[1]['knn']['n_neighbors'] for x in info_mutual_knn[sample_type]]

    metrics = [
        (recall_vals, recall_vals_mutual_knn, "recall"),
        (mrr_vals, mrr_vals_mutual_knn, "mrr"),
        (ndcg_vals, ndcg_vals_mutual_knn, "ndcg"),
        (map_vals, map_vals_mutual_knn, "map"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for ax, (values, values_mutual, label) in zip(axes, metrics):
        ax.plot(k_vals, values, marker='o', color='g', label="knn")
        ax.plot(k_vals_mutual_knn, values_mutual, marker='o', color='r', label="mutual knn")
        ax.set_xlabel("Number of Neighbors (k)")
        ax.set_ylabel(label)
        ax.legend()
        ax.set_title(f"{label} (kNN graph) vs (mutual kNN graph)")
        ax.grid(False)

    plt.tight_layout()
    plt.show()
def plot_eval_results_of_a_graph(info):
    results = info[0]

    init_groups = {}
    for entry in results:
        init_val = entry[6]
        init_groups.setdefault(init_val, []).append(entry)

    fig, ax = plt.subplots(figsize=(8, 6))

    for init_val, group in sorted(init_groups.items()):
        group_sorted = sorted(group, key=lambda x: x[5])
        alphas = [x[5] for x in group_sorted]
        recalls = [x[0] for x in group_sorted]
        ax.plot(alphas, recalls, marker='o', label=f"init={init_val}")

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall vs Alpha for different init values\n({info[2]})")
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    plt.show()

    alpha_groups = {}
    for entry in results:
        alpha_val = entry[5]
        alpha_groups.setdefault(alpha_val, []).append(entry)

    fig, ax = plt.subplots(figsize=(8, 6))

    for alpha_val, group in sorted(alpha_groups.items()):
        group_sorted = sorted(group, key=lambda x: x[6])
        init = [x[6] for x in group_sorted]
        recalls = [x[0] for x in group_sorted]
        ax.plot(init, recalls, marker='o', label=f"alpha={alpha_val}")

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall vs Alpha for different init values\n({info[2]})")
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    plt.show()



def getting_plot_retrieval_stats():
    files = dt.get_files()
    info_to_plot_knn = {}
    info_to_plot_knn['small'] = []
    info_to_plot_knn['medium'] = []
    info_to_plot_knn['long'] = []
    info_to_plot_knn['all_samples'] = []
    info_to_plot_mutual_knn = {}
    info_to_plot_mutual_knn['small'] = []
    info_to_plot_mutual_knn['medium'] = []
    info_to_plot_mutual_knn['long'] = []
    info_to_plot_mutual_knn['all_samples'] = []
    for file in files:
        eval_results = dt.load_eval_results(file)
        params, info = dt.load_graph_parameters(file)
        if params['graph_type'] == "knn graph":
            print(file)
            info_to_plot_knn['small'].append((eval_re(eval_results, 'small'), params, file))
            info_to_plot_knn['medium'].append((eval_re(eval_results, 'medium'), params, file))
            info_to_plot_knn['long'].append((eval_re(eval_results, 'long'), params, file))
            info_to_plot_knn['all_samples'].append((eval_re(eval_results, 'all_samples'), params, file))

        elif params['graph_type'] == "mutual knn graph":
            info_to_plot_mutual_knn['small'].append((eval_re(eval_results, 'small'), params, file))
            info_to_plot_mutual_knn['medium'].append((eval_re(eval_results, 'medium'), params, file))
            info_to_plot_mutual_knn['long'].append((eval_re(eval_results, 'long'), params, file))
            info_to_plot_mutual_knn['all_samples'].append((eval_re(eval_results, 'all_samples'), params, file))


    plot_eval_results_of_a_graph(info_to_plot_knn['small'][0])
    plot_eval_results_of_a_graph(info_to_plot_knn['medium'][0])
    plot_eval_results_of_a_graph(info_to_plot_knn['long'][0])
    plot_eval_results_of_a_graph(info_to_plot_knn['all_samples'][0])
    #print(info_to_plot_knn['small'][0][1]['knn']['n_neighbors'])
    #info_to_plot_knn['small'].sort(key=lambda x: x[1]['knn']['n_neighbors'])
    #info_to_plot_knn['medium'].sort(key=lambda x: x[1]['knn']['n_neighbors'])
    #info_to_plot_knn['long'].sort(key=lambda x: x[1]['knn']['n_neighbors'])
    #info_to_plot_knn['all_samples'].sort(key=lambda x: x[1]['knn']['n_neighbors'])

    #info_to_plot_mutual_knn['small'].sort(key=lambda x: x[1]['knn']['n_neighbors'])
    #info_to_plot_mutual_knn['medium'].sort(key=lambda x: x[1]['knn']['n_neighbors'])
    #info_to_plot_mutual_knn['long'].sort(key=lambda x: x[1]['knn']['n_neighbors'])
    #info_to_plot_mutual_knn['all_samples'].sort(key=lambda x: x[1]['knn']['n_neighbors'])
    #plot_eval_results(info_to_plot_knn, info_to_plot_mutual_knn, "small")
    #plot_eval_results(info_to_plot_knn, info_to_plot_mutual_knn, "medium")
    #plot_eval_results(info_to_plot_knn, info_to_plot_mutual_knn, "long")
    #plot_eval_results(info_to_plot_knn, info_to_plot_mutual_knn, "all_samples")






def plot_subgraph(G, nodes):
    # Function that plots a part of the given graph
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

    plt.title('Semantic Graph')
    plt.axis('off')
    plt.show()


def convert(obj):
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, dict):        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
    return obj


def print_graph_stats(graph):

    graph_info = {}

    if graph.is_directed():
        components = list(nx.strongly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    # Louvain and average_clustering only support undirected graphs
    undirected = graph.to_undirected() if graph.is_directed() else graph

    graph_info["Nodes"] = graph.number_of_nodes()
    graph_info["Edges"] = graph.number_of_edges()
    graph_info["Pagerank Top Nodes"] = rt.pagerank(graph, 5)
    graph_info["Graph Density"] = nx.density(graph)

    if graph.number_of_nodes() > 0:
        graph_info["Average Degree"] = 2 * graph.number_of_edges() / graph.number_of_nodes()
    else:
        graph_info["Average Degree"] = 0

    graph_info["Number of connected components"] = len(components)

    community_list = list(
        nx.community.louvain_communities(undirected, weight="weight")
    )
    mod = nx.community.modularity(undirected, community_list)
    graph_info["Number of communities"] = (len(community_list), mod)

    graph_info = {k: convert(v) for k, v in graph_info.items()}
    graph_info["largest_component_size"] = max((len(c) for c in components), default=0)
    graph_info["avg_clustering"] = nx.average_clustering(undirected, weight="weight")
    graph_info["top_degree_nodes"] = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:10]

    return graph_info




