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
from collections import defaultdict
from scipy import stats



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
        neigh = NearestNeighbors(n_neighbors=k, metric = "cosine")
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

"""-------------------------------------------------------------Make plot functions-------------------------------------------------------------"""

def make_plot_for_eval_metric_and_graph_stat(eval_metric, graph_stat, method, sample_type, query_type, file):
    graphs_best_perf = dt.get_each_graph_best_perf(method, sample_type, file, query_type)
    data_to_plot = []
    for entry in graphs_best_perf:
        _, graph_info = dt.load_graph_parameters(entry['graph'])
        if graph_stat == "Number of communities":
           data_to_plot.append((entry[eval_metric], graph_info[graph_stat][0]))
        else:
           data_to_plot.append((entry[eval_metric], graph_info[graph_stat]))
    data_to_plot.sort(
        key=lambda x: x[1],
        reverse=False
    )

    x_vals = [d[1] for d in data_to_plot]
    y_vals = [d[0] for d in data_to_plot]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'o', alpha=0.4, label='data')


    coeffs = np.polyfit(x_vals, y_vals, deg=1)
    trend_line = np.poly1d(coeffs)
    x_sorted = np.linspace(min(x_vals), max(x_vals), 100)
    plt.plot(x_sorted, trend_line(x_sorted), '-', color='red', linewidth=2, label='trend')

    plt.xlabel(graph_stat)
    plt.ylabel(eval_metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'Outputs/runs/{file}/plots/{eval_metric}_vs_{graph_stat}_{sample_type}_{method}_{query_type}.png', dpi=300, bbox_inches='tight')
def make_plot_performance_of_different_graph_types(eval_metric, method, sample_type, query_type, on_avg, file):
    graphs_best_perf = dt.get_each_graph_best_perf(method, sample_type, file, query_type)
    data_to_plot = defaultdict(list)
    for entry in graphs_best_perf:
        graph_params, _ = dt.load_graph_parameters(entry['graph'])
        data_to_plot[graph_params["graph_type"]].append(entry[eval_metric])
    if on_avg is False:
        x = data_to_plot.keys()
        y = [max(data_to_plot[graph_type]) for graph_type in data_to_plot.keys()]
    else:
        x = data_to_plot.keys()
        y = [np.mean(data_to_plot[graph_type]) for graph_type in data_to_plot.keys()]
    plt.figure(figsize=(16, 12))
    plt.xlabel("Graph Types")
    plt.ylabel(eval_metric)
    plt.bar(x, y)
    plt.savefig(f'Outputs/runs/{file}/plots/{eval_metric}_vs_graph_types_{sample_type}_{method}_{query_type}.png',
                dpi=300, bbox_inches='tight')


def forest_plot(data, sample_type, method, query_type, file):
    """Function that makes a forest plot for paired bootstrap"""
    metrics = ["recall", "mrr", "ndcg", "map"]
    y_pos = np.arange(len(metrics))
    for graph, stats in data.items():
        # For each graph collects the diffs , ci_lower, ci_higher for each metric
        diffs = [stats[metric]["observed_diff"] for metric in metrics]
        ci_lower = [stats[metric]["ci"][0] for metric in metrics]
        ci_higher = [stats[metric]["ci"][1] for metric in metrics]
        errors = [np.array(diffs) - np.array(ci_lower), np.array(ci_higher) - np.array(diffs)]

        plt.figure(figsize=(7, 4))
        plt.errorbar(diffs, y_pos, xerr=errors, fmt='o', capsize=5, color='steelblue')
        plt.axvline(0, color='red', linestyle='--', alpha=0.6, label='no difference')
        plt.yticks(y_pos, metrics)
        plt.xlabel("Graph method − Baseline (Δ)")
        plt.title("Paired bootstrap: Graph method vs Cosine baseline")
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'Outputs/runs/{file}/plots/paired_bootstrap_{graph}_{sample_type}_{method}_{query_type}.png',
                    dpi=300, bbox_inches='tight')


def make_spearmanr_plots(file):
    correlations = dt.get_spearmanr_freeze()
    for name, stats in correlations:
        if stats["significant"] == "not significant":
            continue
        make_plot_for_eval_metric_and_graph_stat(stats["metric"], stats["graph_stat"], "PPR", "all_samples", "dev", file)

"""-------------------------------------------------------------Make plot functions-------------------------------------------------------------"""
"""-------------------------------------------------------------Data gathetring functions-------------------------------------------------------------"""


"""-------------------------------------------------------------Data gathetring functions-------------------------------------------------------------"""
























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

def calc_graph_stats(graph):

    graph_info = {}


    graph_info["Nodes"] = graph.number_of_nodes()
    graph_info["Edges"] = graph.number_of_edges()
    graph_info["Pagerank Top Nodes"] = rt.pagerank(graph, 5)
    graph_info["Graph Density"] = nx.density(graph)
    graph_info["top_degree_nodes"] = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:10]

    is_directed = graph.is_directed()
    if is_directed:
        g_ig = ig.Graph.from_networkx(graph)
        g_ig = g_ig.as_undirected(mode="collapse", combine_edges="max")
    else:
        g_ig = ig.Graph.from_networkx(graph)
    graph = []
    if is_directed:
        mode_cc = "strong"
    else:
        mode_cc = "weak"

    components = g_ig.connected_components(mode=mode_cc)
    graph_info["Number of connected components"] = len(components)
    graph_info["largest_component_size"] = max((len(c) for c in components), default=0)
    partition = g_ig.community_multilevel(weights="weight")
    node_names = g_ig.vs["_nx_name"]
    community_list = [set(node_names[idx] for idx in cluster) for cluster in partition]


    mod = partition.modularity
    graph_info["Number of communities"] = (len(community_list), mod)

    local_cluster_coefficients = g_ig.transitivity_local_undirected(
        weights="weight"
    )

    graph_info["avg_clustering"] = float(np.nanmean(local_cluster_coefficients))


    graph_info = {k: convert(v) for k, v in graph_info.items()}
    return graph_info

def print_graph_stats_(graph):

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




