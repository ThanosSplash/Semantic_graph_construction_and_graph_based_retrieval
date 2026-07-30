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

"""-------------------------------------------------------------Make plot functions-------------------------------------------------------------"""

def make_plot_for_best_eval_results(info_knn, info_mutual_knn, info_kmeans_knn, info_kmeans_mutual_knn, sample_type):
    def extract(info, sample_type):
        recall = [x[0][0]['recall'] for x in info[sample_type]]
        mrr = [x[0][0]['mrr'] for x in info[sample_type]]
        ndcg = [x[0][0]['ndcg'] for x in info[sample_type]]
        map_ = [x[0][0]['map'] for x in info[sample_type]]
        k = [x[1] for x in info[sample_type]]
        return k, recall, mrr, ndcg, map_

    k_vals, recall_vals, mrr_vals, ndcg_vals, map_vals = extract(info_knn, sample_type)
    k_vals_mutual, recall_vals_mutual, mrr_vals_mutual, ndcg_vals_mutual, map_vals_mutual = extract(info_mutual_knn, sample_type)

    metrics = [
        (recall_vals, recall_vals_mutual, "recall"),
        (mrr_vals, mrr_vals_mutual, "mrr"),
        (ndcg_vals, ndcg_vals_mutual, "ndcg"),
        (map_vals, map_vals_mutual, "map"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for ax, (values, values_mutual, label) in zip(axes, metrics):
        ax.plot(k_vals, values, marker='o', color='g', label="knn")
        ax.plot(k_vals_mutual, values_mutual, marker='o', color='r', label="mutual knn")
        ax.set_xlabel("Number of Neighbors (k)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} - {sample_type}")
        ax.grid(False)

    clusters = list(info_kmeans_knn.keys())
    print(clusters)
    colors_knn = ['blue', 'orange', 'purple', 'brown', 'magenta', 'cyan', 'olive', 'navy']
    colors_mutual = ['gold', 'teal', 'pink', 'gray', 'indigo', 'coral', 'lime', 'chocolate']

    for i, cluster in enumerate(clusters):
        color = colors_knn[i % len(colors_knn)]
        color_ = colors_mutual[i % len(colors_mutual)]
        k_k, recall_k, mrr_k, ndcg_k, map_k = extract(info_kmeans_knn[cluster], sample_type)
        k_km, recall_km, mrr_km, ndcg_km, map_km = extract(info_kmeans_mutual_knn[cluster], sample_type)

        cluster_metrics = [recall_k, mrr_k, ndcg_k, map_k]
        cluster_metrics_mutual = [recall_km, mrr_km, ndcg_km, map_km]

        for ax, vals, vals_mutual in zip(axes, cluster_metrics, cluster_metrics_mutual):
            ax.plot(k_k, vals, marker='o', color=color, label=f"kmeans knn ({cluster})")
            ax.plot(k_km, vals_mutual, marker='o', color=color_, label=f"kmeans mutual knn ({cluster})")

    for ax in axes:
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def make_plot_for_eval_results(info_knn, info_mutual_knn, info_kmeans_knn, info_kmeans_mutual_knn, sample_type):
    def extract(info, sample_type):
        recall = [x[0][0]['recall'] for x in info[sample_type]]
        mrr = [x[0][0]['mrr'] for x in info[sample_type]]
        ndcg = [x[0][0]['ndcg'] for x in info[sample_type]]
        map_ = [x[0][0]['map'] for x in info[sample_type]]
        k = [x[1] for x in info[sample_type]]
        return k, recall, mrr, ndcg, map_

    k_vals, recall_vals, mrr_vals, ndcg_vals, map_vals = extract(info_knn, sample_type)
    k_vals_mutual, recall_vals_mutual, mrr_vals_mutual, ndcg_vals_mutual, map_vals_mutual = extract(info_mutual_knn, sample_type)
    baseline = dt.load_results_file(f"Outputs/Baseline-RAG/baseline_results.json")['Baseline'][sample_type]
    baseline_recall = baseline[0]['recall']
    baseline_mrr = baseline[0]['mrr']
    baseline_ndcg = baseline[0]['ndcg']
    baseline_map = baseline[0]['map']
    metrics = [
        (recall_vals, recall_vals_mutual, baseline_recall,"recall"),
        (mrr_vals, mrr_vals_mutual,baseline_mrr,"mrr"),
        (ndcg_vals, ndcg_vals_mutual,baseline_ndcg ,"ndcg"),
        (map_vals, map_vals_mutual,baseline_map ,"map"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for ax, (values, values_mutual, baseline_val, label) in zip(axes, metrics):
        ax.plot(k_vals, values, marker='o', color='g', label="knn")
        ax.plot(k_vals_mutual, values_mutual, marker='o', color='purple', label="mutual knn")
        ax.axhline(
            y=baseline_val,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Baseline RAG'
        )
        ax.set_ylabel(label)
        ax.set_title(f"{label} - {sample_type}")
        ax.grid(False)

    clusters = list(info_kmeans_knn.keys())
    print(clusters)
    colors_knn = ['blue', 'orange', 'brown', 'magenta', 'cyan', 'olive', 'navy']
    colors_mutual = ['gold', 'teal', 'pink', 'gray', 'indigo', 'coral', 'lime', 'chocolate']

    for i, cluster in enumerate(clusters):
        color = colors_knn[i % len(colors_knn)]
        color_ = colors_mutual[i % len(colors_mutual)]
        k_k, recall_k, mrr_k, ndcg_k, map_k = extract(info_kmeans_knn[cluster], sample_type)
        k_km, recall_km, mrr_km, ndcg_km, map_km = extract(info_kmeans_mutual_knn[cluster], sample_type)

        cluster_metrics = [recall_k, mrr_k, ndcg_k, map_k]
        cluster_metrics_mutual = [recall_km, mrr_km, ndcg_km, map_km]

        for ax, vals, vals_mutual in zip(axes, cluster_metrics, cluster_metrics_mutual):
            ax.plot(k_k, vals, marker='o', color=color, label=f"kmeans knn ({cluster})")
            ax.plot(k_km, vals_mutual, marker='o', color=color_, label=f"kmeans mutual knn ({cluster})")

    for ax in axes:
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def make_plot_of_eval_results_of_a_graph_type(info_to_plot, sample_type):
    def extract(info, sample_type):
        recall = [x[0][0]['recall'] for x in info[sample_type]]
        mrr = [x[0][0]['mrr'] for x in info[sample_type]]
        ndcg = [x[0][0]['ndcg'] for x in info[sample_type]]
        map_ = [x[0][0]['map'] for x in info[sample_type]]
        k = [x[1] for x in info[sample_type]]
        return k, recall, mrr, ndcg, map_

    k_vals_all = []
    recall_vals_all = []
    mrr_vals_all = []
    ndcg_vals_all = []
    map_vals_all = []

    for method in info_to_plot.keys():
        k_vals, recall_vals, mrr_vals, ndcg_vals, map_vals = extract(info_to_plot[method], sample_type)
        k_vals_all.append((k_vals, method))
        recall_vals_all.append((recall_vals, method))
        mrr_vals_all.append((mrr_vals, method))
        ndcg_vals_all.append((ndcg_vals, method))
        map_vals_all.append((map_vals, method))
    metrics = [
        (recall_vals_all, "recall"),
        (mrr_vals_all, "mrr"),
        (ndcg_vals_all, "ndcg"),
        (map_vals_all, "map"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    colors_knn = ['blue', 'orange', 'purple', 'brown', 'magenta', 'cyan', 'olive', 'navy']
    for ax, (values, label) in zip(axes, metrics):
        for i in range(len(values)):
            print(values[i][0])
            ax.plot(k_vals_all[i][0], values[i][0], marker='o', color=colors_knn[i], label=values[i][1])

        ax.set_xlabel("Number of Neighbors (k)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} - {sample_type}")
        ax.grid(False)
        ax.legend()

    plt.tight_layout()
    plt.show()
def make_plot_eval_results_of_a_graph(info):
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
"""-------------------------------------------------------------Make plot functions-------------------------------------------------------------"""
"""-------------------------------------------------------------Data gathetring functions-------------------------------------------------------------"""

def weighted_vs_unweighted_undirected(method, sample_type):
    weighted_unweighted = dt.get_data_weighted_unweighted_undirected(method, sample_type)
    baseline = dt.load_results_file(f"Outputs/Baseline-RAG/baseline_results.json")['Baseline'][sample_type][0]['recall']
    categories = weighted_unweighted.keys()
    weighted = []
    unweighted = []
    for graph in weighted_unweighted.keys():
        weighted.append(weighted_unweighted[graph]['weighted'][1])
        unweighted.append(weighted_unweighted[graph]['unweighted'][1])


    x = np.arange(len(categories))
    plt.figure(figsize=(18, 6))
    plt.bar(x - 0.2, weighted, width=0.4, label='Weighted')
    plt.bar(x + 0.2, unweighted, width=0.4, label='Unweighted')
    plt.xticks(x + 0.6, categories, rotation=10, ha='right', fontsize=7)

    plt.axhline(
        y=baseline,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Baseline RAG'
    )
    plt.xlabel('Graphs')
    plt.ylabel('Recall')
    plt.title('Weighted vs Unweighted')
    plt.legend()
    plt.show()


def undirected_vs_directed_weighted(method, sample_type):
    directed_undirected = dt.get_data_undirected_directed_weighted(method, sample_type)
    baseline = dt.load_results_file(f"Outputs/Baseline-RAG/baseline_results.json")['Baseline'][sample_type][0]['recall']
    categories = directed_undirected.keys()
    directed = []
    undirected = []
    for graph in directed_undirected.keys():
        directed.append(directed_undirected[graph]['directed'][1])
        undirected.append(directed_undirected[graph]['undirected'][1])

    x = np.arange(len(categories))
    plt.figure(figsize=(18, 6))
    plt.bar(x - 0.2, directed, width=0.4, label='Directed')
    plt.bar(x + 0.2, undirected, width=0.4, label='Undirected')
    plt.xticks(x + 0.6, categories, rotation=10, ha='right', fontsize=7)
    plt.axhline(
        y=baseline,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Baseline RAG'
    )
    plt.xlabel('Graphs')
    plt.ylabel('Recall')
    plt.title('Directed vs Undirected')
    plt.legend()
    plt.show()

def undirected_vs_directed_unweighted(method, sample_type):
    directed_undirected = dt.get_data_undirected_directed_unweighted(method, sample_type)
    baseline = dt.load_results_file(f"Outputs/Baseline-RAG/baseline_results.json")['Baseline'][sample_type][0]['recall']
    categories = directed_undirected.keys()
    directed = []
    undirected = []
    for graph in directed_undirected.keys():
        directed.append(directed_undirected[graph]['directed'][1])
        undirected.append(directed_undirected[graph]['undirected'][1])
    x = np.arange(len(categories))
    plt.figure(figsize=(18, 6))
    plt.bar(x - 0.2, directed, width=0.4, label='Directed')
    plt.bar(x + 0.2, undirected, width=0.4, label='Undirected')
    plt.xticks(x + 0.6, categories, rotation=10, ha='right', fontsize=7)
    plt.axhline(
        y=baseline,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Baseline RAG'
    )
    plt.xlabel('Graphs')
    plt.ylabel('Recall')
    plt.title('Directed Unweighted vs Undirected')
    plt.legend()
    plt.show()




def clustering_vs_not_clustering(method, sample_type):
    clustering_not_clustering = dt.get_data_clustering_not_clustering(method, sample_type)
    baseline = dt.load_results_file(f"Outputs/Baseline-RAG/baseline_results.json")['Baseline'][sample_type][0]['recall']
    categories = list(clustering_not_clustering.keys())

    clustering_vals = []
    no_clustering_vals = []

    for graph_type in categories:
        clustering_avg = max(clustering_not_clustering[graph_type]["clustering"])
        no_clustering_avg = max(clustering_not_clustering[graph_type]["no_clustering"])

        clustering_vals.append(clustering_avg)
        no_clustering_vals.append(no_clustering_avg)

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(x - width/2, clustering_vals, width, label="Clustering")
    plt.bar(x + width/2, no_clustering_vals, width, label="No Clustering")
    plt.axhline(
        y=baseline,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Baseline RAG'
    )

    plt.xticks(x, categories)
    plt.xlabel("Graph Type")
    plt.ylabel("Recall")
    plt.title("Clustering vs No Clustering")
    plt.legend()

    plt.tight_layout()
    plt.show()


def ppr_vs_k_steph_graph_aware(sample_type):
        ppr_k_steph_graph_aware = dt.get_data_ppr_vs_ksteph(sample_type)
        baseline = dt.load_results_file(f"Outputs/Baseline-RAG/baseline_results.json")['Baseline'][sample_type][0][
            'recall']
        categories = ppr_k_steph_graph_aware.keys()
        #print(categories)
        ppr_vals = []
        k_steph_vals = []
        for graph_type in ppr_k_steph_graph_aware.keys():
            kmeans_avg = max(ppr_k_steph_graph_aware[graph_type]['PPR'])
            not_clustering_avg = max(ppr_k_steph_graph_aware[graph_type]['k-steph_graph_aware'])
            ppr_vals.append(kmeans_avg)
            k_steph_vals.append(not_clustering_avg)

        x = np.arange(len(categories))
        plt.figure(figsize=(18, 6))
        plt.bar(x - 0.2, k_steph_vals, width=0.4, label='graph_aware')
        plt.bar(x + 0.2, ppr_vals, width=0.4, label='PPR')
        plt.axhline(
            y=baseline,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Baseline RAG'
        )

        plt.xticks(x + 0.6, categories, rotation=10, ha='right', fontsize=7)
        plt.xlabel('Graphs')
        plt.ylabel('Recall')
        plt.title('PPR vs K-steph graph aware')
        plt.legend()
        plt.show()

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

def print_graph_stats(graph):

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

    #closeness_values = g_ig.closeness(weights="weight")
    #closeness_pairs = list(zip(node_names, closeness_values))
    #graph_info["top_closeness_nodes"] = sorted(
        #closeness_pairs, key=lambda x: x[1], reverse=True
    #)[:10]


    #betweenness_values = g_ig.betweenness(weights="weight")
    #betweenness_pairs = list(zip(node_names, betweenness_values))
    #graph_info["top_betweenness_nodes"] = sorted(
    #    betweenness_pairs, key=lambda x: x[1], reverse=True
    #)[:10]

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




