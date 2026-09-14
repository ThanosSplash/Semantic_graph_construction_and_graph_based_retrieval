import os
import json
from collections import defaultdict
from scipy.stats import spearmanr
import numpy as np
import evaluation as ev
import data_loading as dt
import plotting as pt
import pickle
COL_WIDTHS = {
    "method": 25,
    "density": 25,
    "components": 15,
    "recall": 25,
    "mrr": 25,
    "ndcg": 25,
    "map": 25,
    "graph": 120,
    "graph_name": 90,
    "graph_name_": 70,
    "hops": 25,
    "init": 25,
    "id": 25,
    "baseline_rank": 25,
    "baseline_recall": 25,
    "rank": 25,
    "difference": 25,
    "case": 25,
    "alpha": 20,
    "k": 15,
    "reranker": 25,
    "weighted_score": 20,
    "sample_type":20,
    "norm": 20,
    "seed_selection": 25,
    "multi-hop": 25,
    'latency': 25,
    'random_state': 20,
     'metric' : 25,
     'n': 25,
      "pos": 25,
      "neg": 25 ,
      "same": 25,
      "sum_delta": 30,
      "sum_abs": 30,
       "scaler":20,
       "pca": 20,
     "dist_metric":20,
     "Directed": 20,
     "Weighted": 20,
     "std_ndcg": 25,
     "mean_ndcg": 25,
     "std_recall": 25,
     "mean_recall": 25,
     "std_mrr": 25,
     "mean_mrr": 25,
     "std_map": 25,
     "mean_map": 25,
     "max_ndcg":25,
     "max_recall": 25,
     "max_mrr": 25,
     "max_map": 25,

}


def leaderboard_header_(headers):
    header_row = ""
    for header in headers:
        header_row += f"{header:<{COL_WIDTHS[header]}}"

    return header_row
def leaderboard_row_(row_content, headers):
    row_ = ""
    for header in headers:
        if header in row_content:
            row_ += f"{row_content[header]:<{COL_WIDTHS[header]}}"
        else:
            blank = "-"
            row_ += f"{blank:<{COL_WIDTHS[header]}}"


    return row_


def save_queries_rank_diff_table(data, dir, name, sort_key="difference"):
    headers = ['method', 'baseline_rank', 'rank', 'difference', 'case', 'multi-hop']
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph in data.keys():
        rows = sorted(data[graph][0].items(), key=lambda x: x[1]["difference"], reverse=True)
        with open(f"{dir}/{name}", "a") as f:
            f.write(f"----------{graph}----------------------\n\n")
            f.write(f"{'id':<{COL_WIDTHS['id']}}")
            f.write(leaderboard_header_(headers) + "\n")
            for id_, row in rows:
                f.write(f"{id_:<{COL_WIDTHS['id']}}")
                f.write(leaderboard_row_(row, headers) + "\n")
            f.write(
                f"\n\nImproved: {data[graph][1]['Improved']}, Same: {data[graph][1]['Same']}, Worst: {data[graph][1]['Worst']}")
            f.write(f"\n\n--------------------------------\n\n")

def save_queries_recall_diff_table(data, dir, name, sort_key="difference"):
    headers = ['method', 'baseline_recall', 'recall', 'difference', 'case', 'multi-hop']
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph in data.keys():
        rows = sorted(data[graph][0].items(), key=lambda x: x[1]["difference"], reverse=True)
        with open(f"{dir}/{name}", "a") as f:
            f.write(f"----------{graph}----------------------\n\n")
            f.write(f"{'id':<{COL_WIDTHS['id']}}")
            f.write(leaderboard_header_(headers) + "\n")
            for id_, row in rows:
                f.write(f"{id_:<{COL_WIDTHS['id']}}")
                f.write(leaderboard_row_(row, headers) + "\n")
            f.write(
                f"\n\n All types")
            f.write(
                f"\n\nImproved: {data[graph][1]['Improved']}, Same: {data[graph][1]['Same']}, Worst: {data[graph][1]['Worst']}")
            f.write(
                f"\n\n Multi Hop")
            f.write(f"\n\n--------------------------------\n\n")
def save_ppr_alpha_sensitivity_table(data, dir, name, sort_key="init"):
    # Clear txt file
    headers = ['graph_name', 'sample_type', 'init', 'k', 'alpha', 'recall', 'mrr', 'ndcg', 'map',"weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph in data.keys():
        rows = sorted(data[graph], key=lambda x: (x["graph_name"], x["k"], x["init"], x["alpha"]), reverse=False)
        with open(f"{dir}/{name}", "a") as f:
                f.write(leaderboard_header_(headers) + "\n")
                for row in rows:
                    f.write(leaderboard_row_(row, headers) + "\n")

def save_ppr_init_sensitivity_table(data, dir, name):
    headers = ['graph_name', 'sample_type', 'alpha', 'init', 'k', 'recall', 'mrr', 'ndcg', 'map', "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph in data.keys():
        rows = sorted(data[graph], key=lambda x: (x["graph_name"], x["k"], x["alpha"], x["init"]), reverse=False)
        with open(f"{dir}/{name}", "a") as f:
            f.write(leaderboard_header_(headers) + "\n")
            for row in rows:
                f.write(leaderboard_row_(row, headers) + "\n")


def save_graph_sensitivity_table(data, dir, name):
    headers = ['graph_name','sample_type',"density", "components", "recall", "mrr", "ndcg", "map", "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph in data.keys():
        rows = sorted(data[graph], key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"], x["density"]), reverse=True)
        with open(f"{dir}/{name}", "a") as f:
            f.write(leaderboard_header_(headers) + "\n")
            for row in rows:
                f.write(leaderboard_row_(row, headers) + "\n")
            f.write(f"---------------------------------  \n\n")

def save_hop_sensitivity_table(data, dir, name):
  headers = ['graph_name', 'sample_type', 'hops', "init", "alpha", "reranker", "k", "recall", "mrr", "ndcg", "map"]
  with open(f"{dir}/{name}", "w") as f:
        f.write("")
  #rows = sorted(data, key=lambda x: x['hops'], reverse=False)
  for graph in data.keys():
      rows = sorted(data[graph], key=lambda x: (x["graph_name"], x["k"], x["alpha"], x["init"], x["hops"]), reverse=False)
      with open(f"{dir}/{name}", "a") as f:
          f.write(leaderboard_header_(headers) + "\n")
          for row in rows:
              f.write(leaderboard_row_(row, headers) + "\n")
          f.write(f"---------------------------------  \n\n")


def save_norm_table(data_min_max, data_z_score, data_log_min_max, data_raw, avgs, data_raw_ppr, data_raw_cosine, dir, name):
    headers = ['graph_name', 'norm', 'sample_type', 'alpha', 'init', 'k', 'recall', 'mrr', 'ndcg', 'map', "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    with open(f"{dir}/{name}", "a") as f:
        f.write("With Min_Max\n\n")
        f.write(leaderboard_header_(headers) + "\n")
        for row in data_min_max:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"\nAvg ndcg {avgs['Min_Max']}\n")
        f.write("\n---------------------------------------\n\n")

    with open(f"{dir}/{name}", "a") as f:
        f.write("With z_score\n\n")
        f.write(leaderboard_header_(headers) + "\n")
        for row in data_z_score:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"\nAvg ndcg {avgs['z_score']}\n")
        f.write("\n---------------------------------------\n\n")

    with open(f"{dir}/{name}", "a") as f:
        f.write("With log min_max\n\n")
        f.write(leaderboard_header_(headers) + "\n")
        for row in data_log_min_max:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"\nAvg ndcg {avgs['log + Min_Max']}\n")
        f.write("\n---------------------------------------\n\n")

    with open(f"{dir}/{name}", "a") as f:
        f.write("Raw\n\n")
        f.write(leaderboard_header_(headers) + "\n")
        for row in data_raw:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"\nAvg ndcg {avgs['raw']}\n")
        f.write("\n---------------------------------------\n\n")

    with open(f"{dir}/{name}", "a") as f:
        f.write("Raw PPR score\n\n")
        f.write(leaderboard_header_(headers) + "\n")
        for row in data_raw_ppr:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write("\n---------------------------------------\n\n")

    with open(f"{dir}/{name}", "a") as f:
        f.write("Raw cosine score\n\n")
        f.write(leaderboard_header_(headers) + "\n\n")
        for row in data_raw_cosine:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write("\n---------------------------------------\n\n")

def save_seed_selection_table(data_cosine, data_bm25, data_fusion, avgs, dir, name):
    headers = ['graph_name', 'seed_selection', 'norm', 'sample_type', 'alpha', 'init', 'k', 'recall', 'mrr', 'ndcg', 'map', "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    with open(f"{dir}/{name}", "a") as f:
        f.write("With cosine\n\n")
        f.write(leaderboard_header_(headers) + "\n")
        for row in data_cosine:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"\nAvg ndcg {avgs['cosine'][0]}\n")
        f.write(f"\nstd {avgs['cosine'][1]}\n")
        f.write("\n---------------------------------------\n\n")

    with open(f"{dir}/{name}", "a") as f:
        f.write("With bm25\n\n")
        f.write(leaderboard_header_(headers) + "\n")
        for row in data_bm25:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"\nAvg ndcg {avgs['bm25'][0]}\n")
        f.write(f"\nstd {avgs['bm25'][1]}\n")
        f.write("\n---------------------------------------\n\n")

    with open(f"{dir}/{name}", "a") as f:
        f.write("With fusion\n\n")
        f.write(leaderboard_header_(headers) + "\n")
        for row in data_fusion:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"\nAvg ndcg {avgs['fusion'][0]}\n")
        f.write(f"\nstd {avgs['fusion'][1]}\n")
        f.write("\n---------------------------------------\n\n")
def save_leaderboard(data, dir, name):
    headers = ['graph_name',"sample_type", "Directed", "Weighted", "dist_metric", "pca", "scaler","seed_selection", "norm", "alpha", "init", "hops", "k", "recall", "mrr", "ndcg", "map", "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph in data.keys():
        rows = sorted(data[graph], key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)
        with open(f"{dir}/{name}", "a") as f:
            f.write(leaderboard_header_(headers) + "\n")
            for row in rows:
                f.write(leaderboard_row_(row, headers) + "\n")
            f.write(f"---------------------------------  \n\n")

def save_leaderboard_(data, dir, name):
    headers = ['graph_name_', "mean_ndcg", "std_ndcg", "max_ndcg", "mean_mrr", "std_mrr", "max_mrr", "mean_recall", "std_recall", "max_recall", "mean_map", "std_map", "max_map"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    rows = sorted(data, key=lambda x: (x["mean_ndcg"], -x["std_ndcg"]), reverse=True)
    with open(f"{dir}/{name}", "a") as f:
            f.write(leaderboard_header_(headers) + "\n")
            for row in rows:
                f.write(leaderboard_row_(row, headers) + "\n")
            f.write(f"---------------------------------  \n\n")


def save_random_state_table(data, configs ,dir, name):
    headers = ['random_state', 'graph_name', 'seed_selection', 'norm', 'sample_type', 'alpha', 'init', 'k', 'recall', 'mrr', 'ndcg',
               'map', "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for random_state in data.keys():
        with open(f"{dir}/{name}", "a") as f:
                f.write(f"Random state: {random_state}\n\n")
                f.write(leaderboard_header_(headers) + "\n")
                for entry in data[random_state]:
                    f.write(leaderboard_row_(entry, headers) + "\n")


                f.write(f"\nMean: {configs[random_state]['mean']['ndcg']}\n")
                f.write(f"Stds: {configs[random_state]['std']['ndcg']}\n")
                f.write(f"Ndcg: {configs[random_state]['mean']['ndcg'] + configs[random_state]['std']['ndcg']} , {configs[random_state]['mean']['ndcg'] - configs[random_state]['std']['ndcg']}\n")
                f.write(f"----------------------------------\n\n")

def save_param_sensitivity_stats_table(data, dir, name):
    headers = ["metric", "n", "pos", "neg", "same", "sum_delta", "sum_abs"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph, labels in data.items():
        with open(f"{dir}/{name}", "a") as f:
            f.write(f"Graph : {graph}\n\n")
        for label, metrics in labels.items():
          with open(f"{dir}/{name}", "a") as f:
            f.write(f"Transition : {label}\n\n")
            f.write(leaderboard_header_(headers) + "\n")
            for metric, stats in metrics.items():
                    headers_row = ["n", "pos", "neg", "same", "sum_delta", "sum_abs"]
                    f.write(f"{metric:<{COL_WIDTHS[metric]}}")
                    f.write(leaderboard_row_(stats, headers_row) + "\n")

            f.write(f"\n\n----------------\n\n")
#------------------------------------------------------------------------------------
METRICS = {
    "recall": "recall",
    "mrr": "mrr",
    "ndcg": "ndcg",
    "map": "map",
}

HOP_TRANSITIONS = [(1, 2), (2, 3), (3, 4)]
ALPHA_TRANSITIONS = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
INIT_TRANSITIONS = [(2, 5), (5, 10), (10, 20), (10, 50), (20, 50)]

CONFIG_KEYS_HOPS = ("graph_name", "graph_family","sample_type", "init", "alpha", "k", "reranker")
CONFIG_KEYS_ALPHA = ("graph_name", "graph_family","sample_type", "init", "k", "reranker", "hops")
CONFIG_KEYS_INIT = ("graph_name", "graph_family","sample_type", "alpha", "k", "reranker", "hops")


def compute_param_transition_stats(data, transition_type, mode):
    "Function that calcualtes transition over the changes of a param for each graph family"
    transition = []
    param = ""
    # For each param choosing the correct format
    if transition_type == "hops":
        transition = HOP_TRANSITIONS
        config_keys = CONFIG_KEYS_HOPS
        param = "hops"
    elif transition_type == "alpha":
        transition = ALPHA_TRANSITIONS
        config_keys = CONFIG_KEYS_ALPHA
        param = "alpha"
    elif transition_type == "init":
        transition = INIT_TRANSITIONS
        config_keys = CONFIG_KEYS_INIT
        param = "init"
    # Where the data will be saved  graph_stats[graph_family][transition][metric]
    # transition example for alpha 0.2 - 0.5
    graph_stats = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {"n": 0, "pos": 0, "neg": 0, "same": 0,
                         "sum_delta": 0.0, "sum_abs": 0.0}
            )
        )
    )
    # Group entries that belong to the same "config" same values across all config keys different
    # at the param we want to compare
    configs = defaultdict(dict)
    for entry in data:
        key = tuple(entry[k] for k in config_keys if k in entry)
        configs[key][entry[param]] = entry
    # Calculating graph_stats
    for key, hop_data in configs.items():
        graph = key[config_keys.index(mode)]
        for h1, h2 in transition:
            if h1 not in hop_data or h2 not in hop_data:
                continue
            trans_label = f"{h1}->{h2}"
            for raw_metric, display_metric in METRICS.items():
                v1 = hop_data[h1][raw_metric]
                v2 = hop_data[h2][raw_metric]
                delta = v2 - v1
                s = graph_stats[graph][trans_label][display_metric]
                s["n"] += 1
                s["sum_delta"] += delta
                s["sum_abs"] += abs(delta)
                if delta > 1e-12:
                    s["pos"] += 1
                elif delta < -1e-12:
                    s["neg"] += 1
                else:
                    s["same"] += 1

    return graph_stats
def make_query_table(file, query_type, query_category):
    """Fucntion that creates the query table. For each query showcases if graph method improves its recall, rank
       query_category takes values "all", "one hop", "multi hop"
    """
    # Load baseline and file queries
    queries = dt.load_queries_results(f"Outputs/runs/{file}", query_type)
    baseline_queries = dt.load_queries_results(f"Outputs/Baseline-RAG", query_type)
    # Finding the best performed params and method for mrr
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = []
    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            for sample_type, results in sample_types.items():
                params = results[0].pop("params", {})
                params['method'] = method
                params['graph'] = graph
                all_data.append((results[0]['graph_name'], params))

    all_queries_rank = {}
    for graph, params in all_data:
        queries_rank, diff_summary_rank = dt.extract_best_run_and_compare_to_baseline(queries, baseline_queries, params,'rank', query_category)
        all_queries_rank[graph] = (queries_rank, diff_summary_rank)
    all_queries_recall = {}
    for graph, params in all_data:
        queries_recall, diff_summary_recall = dt.extract_best_run_and_compare_to_baseline(queries, baseline_queries, params, 'recall', query_category)
        all_queries_recall[graph] = (queries_recall, diff_summary_recall)


    save_queries_rank_diff_table(all_queries_rank, f"Outputs/runs/{file}/tables", f"queries_rank_{query_category}_diff.txt")
    save_queries_recall_diff_table(all_queries_recall, f"Outputs/runs/{file}/tables", f"queries_recall_{query_category}_diff.txt")

def make_alpha_sensitivity_table(file):
    # Loading values for each init
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = defaultdict(list)

    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            for sample_type, results in sample_types.items():
                    for result in results:
                        params = result.pop("params", {})
                        graph_params, _ = dt.load_graph_parameters(graph)
                        flat_entry = {
                            "graph_family": graph_params["graph_type"],
                            "method": method,
                            "sample_type": sample_type,
                            **result,
                            **params
                        }

                        all_data[sample_type].append(flat_entry)

    # For each init make a seperate table that track changes in recall , mrr as alpha changes
    graph_stats_alpha = compute_param_transition_stats(all_data["all_samples"], "alpha", "graph_family")
    save_param_sensitivity_stats_table(graph_stats_alpha, f"Outputs/runs/{file}/tables", f"graph_family_alpha_sensitivity_stats.txt")

    graph_stats_alpha = compute_param_transition_stats(all_data["all_samples"], "alpha", "graph_name")
    save_param_sensitivity_stats_table(graph_stats_alpha, f"Outputs/runs/{file}/tables", f"each_graph_alpha_sensitivity_stats.txt")
    save_ppr_alpha_sensitivity_table(all_data, f"Outputs/runs/{file}/tables", f"ppr_alpha_sensitivity_init.txt")


def make_init_sensitivity_table(file):
    # Loading values for each alpha
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data= defaultdict(list)
    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            for sample_type, results in sample_types.items():
                    for result in results:
                        params = result.pop("params", {})
                        graph_params, _ = dt.load_graph_parameters(graph)
                        flat_entry = {
                            "graph_family": graph_params["graph_type"],
                            "method": method,
                            "sample_type": sample_type,
                            **result,
                            **params
                        }

                        all_data[sample_type].append(flat_entry)


    graph_stats_init = compute_param_transition_stats(all_data["all_samples"], "init", "graph_family")
    save_param_sensitivity_stats_table(graph_stats_init, f"Outputs/runs/{file}/tables", f"graph_family_init_sensitivity_stats.txt")
    graph_stats_init = compute_param_transition_stats(all_data["all_samples"], "init", "graph_name")
    save_param_sensitivity_stats_table(graph_stats_init, f"Outputs/runs/{file}/tables",
                                       f"each_graph_init_sensitivity_stats.txt")
    # For each alpha make a seperate table that track changes in recall , mrr as init changes
    save_ppr_init_sensitivity_table(all_data, f"Outputs/runs/{file}/tables", f"ppr_init_sensitivity_alpha.txt")


def make_graph_construction_sensitivity_table(file):
      grouped = dt.load_grouped(f"Outputs/runs/{file}")
      all_data = defaultdict(list)
      for graph, methods in grouped.items():
          for method, sample_types in methods.items():
              for sample_type, results in sample_types.items():
                      params = results[0].pop("params", {})

                      flat_entry = {
                          "method": method,
                          "sample_type": sample_type,
                          **results[0],
                          **params
                      }

                      all_data[sample_type].append(flat_entry)

      save_graph_sensitivity_table(all_data, f"Outputs/runs/{file}/tables", "graph_sensitivity.txt")


def make_hop_sensitivity_table(file):
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = defaultdict(list)

    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            if method == "k-steph":
                for sample_type, results in sample_types.items():
                        for result in results:
                            params = result.pop("params", {})
                            graph_params, _ = dt.load_graph_parameters(graph)
                            flat_entry = {
                                "graph_family": graph_params["graph_type"],
                                "method": method,
                                "sample_type": sample_type,
                                **result,
                                **params
                            }
                            all_data[sample_type].append(flat_entry)



    graph_stats_hops = compute_param_transition_stats(all_data["all_samples"], "hops", "graph_family")
    save_param_sensitivity_stats_table(graph_stats_hops, f"Outputs/runs/{file}/tables", f"graph_family_graph_families_hop_sensitivity_stats.txt")

    graph_stats_hops = compute_param_transition_stats(all_data["all_samples"], "hops", "graph_name")
    save_param_sensitivity_stats_table(graph_stats_hops, f"Outputs/runs/{file}/tables", f"each_graph_hop_sensitivity_stats.txt")

    save_hop_sensitivity_table(all_data, f"Outputs/runs/{file}/tables", f"hop_sensitivity.txt")


def make_leaderboard_table(file):
    # Loading values for each alpha
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = defaultdict(list)
    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            for sample_type, results in sample_types.items():
                    for result in results:
                        params = result.pop("params", {})
                        graph_params, _ = dt.load_graph_parameters(graph)
                        if "graph_params" in graph_params:
                            construction_key = "graph_params"
                        else:
                            construction_key = "graph_construction"

                        graph_type = graph_params["graph_type"]
                        graph_building_algo = graph_params["Graph building algorithm"]

                        if graph_building_algo == "Threshold":
                            dist_metric = "cosine"
                        else:
                            dist_metric = graph_params["Graph building algorithm params"]["metric"],
                        graph_type = graph_params["graph_type"]
                        flat_entry = {
                            "method": method,
                            "sample_type": "all_samples",
                            **result,
                            **params,
                            "pca": graph_params["preprocess"][0],
                            "scaler": graph_params["preprocess"][1],
                            "dist_metric": dist_metric,
                            "Weighted": graph_params[construction_key]["Weighted"],
                            "Directed": graph_params[construction_key]["Directed"]
                        }

                        all_data[sample_type].append(flat_entry)
    save_leaderboard(all_data, f"Outputs/runs/{file}/tables", "leaderboard.txt")

def make_mean_std_leaderboard(file):
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    graph_family = defaultdict(list)
    graph_each = defaultdict(list)
    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            results = grouped[graph][method]['all_samples']
            for result in results:
                    params = result.pop("params", {})
                    if params["alpha"] == 1:
                        continue
                    graph_params, _ = dt.load_graph_parameters(graph)
                    if "graph_params" in graph_params:
                        construction_key = "graph_params"
                    else:
                        construction_key = "graph_construction"


                    graph_building_algo = graph_params["Graph building algorithm"]

                    if graph_building_algo == "Threshold":
                        dist_metric = "cosine"
                    else:
                        dist_metric = graph_params["Graph building algorithm params"]["metric"],
                    graph_type = graph_params["graph_type"]
                    flat_entry = {
                        "method": method,
                        "sample_type": "all_samples",
                        **result,
                        **params,
                        "pca": graph_params["preprocess"][0],
                        "scaler": graph_params["preprocess"][1],
                        "dist_metric": dist_metric,
                        "Weighted": graph_params[construction_key]["Weighted"],
                        "Directed": graph_params[construction_key]["Directed"]
                    }
                    graph_family[graph_type].append(flat_entry)
                    graph_each[graph].append(flat_entry)

    metrics = ["recall", "mrr", "ndcg", "map"]
    graph_family_results = []
    for graph_type in graph_family.keys():
        means_stds = {}
        max_scores = {}
        for metric in metrics:
            scores = [entry[metric] for entry in graph_family[graph_type]]
            means_stds[metric] = (np.mean(scores), np.std(scores))
            max_scores[metric] = max(scores)
        flat_entry = {
            "graph_name_": graph_type,
            "mean_recall": means_stds["recall"][0],
            "mean_mrr": means_stds["mrr"][0],
            "mean_ndcg": means_stds["ndcg"][0],
            "mean_map": means_stds["map"][0],
            "std_recall": means_stds["recall"][1],
            "std_mrr": means_stds["mrr"][1],
            "std_ndcg": means_stds["ndcg"][1],
            "std_map": means_stds["map"][1],
            "max_recall": max_scores["recall"],
            "max_mrr": max_scores["mrr"],
            "max_ndcg": max_scores["ndcg"],
            "max_map": max_scores["map"],
        }
        graph_family_results.append(flat_entry)
    save_leaderboard_(graph_family_results, f"Outputs/runs/{file}/tables", "leaderboard_graph_family.txt")



def make_norm_not_norm_table(file):
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    raw_entries = []
    min_max_entries = []
    log_min_max_entries = []
    z_score_entries = []
    for graph, methods in grouped.items():
        results = grouped[graph]['PPR']['all_samples']
        for result in results:
            params = result.pop("params", {})
            flat_entry = {
                "graph": graph,
                "method": 'PPR',
                "sample_type": 'all_samples',
                **result,
                **params
            }
            if flat_entry['norm'] == "":
                raw_entries.append(flat_entry)
            elif flat_entry['norm'] == "z_score":
                z_score_entries.append(flat_entry)
            elif flat_entry['norm'] == "Min_Max":
                min_max_entries.append(flat_entry)
            elif flat_entry['norm'] == "log + Min_Max":
                log_min_max_entries.append(flat_entry)

    entry_only_cosine = (
            [e for e in min_max_entries if e['alpha'] == 1.0] +
            [e for e in z_score_entries if e['alpha'] == 1.0] +
            [e for e in log_min_max_entries if e['alpha'] == 1.0] +
            [e for e in raw_entries if e['alpha'] == 1.0]
    )
    entry_ppr_raw_score = (
            [e for e in min_max_entries if e['alpha'] == 0.0] +
            [e for e in z_score_entries if e['alpha'] == 0.0] +
            [e for e in log_min_max_entries if e['alpha'] == 0.0] +
            [e for e in raw_entries if e['alpha'] == 0.0]
    )

    min_max_entries = sorted(min_max_entries, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)
    z_score_entries = sorted(z_score_entries, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)
    log_min_max_entries = sorted(log_min_max_entries, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)
    raw_entries = sorted(raw_entries, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)

    min_max_entries = [e for e in min_max_entries if e['alpha'] not in (0.0, 1.0)]
    z_score_entries = [e for e in z_score_entries if e['alpha'] not in (0.0, 1.0)]
    log_min_max_entries = [e for e in log_min_max_entries if e['alpha'] not in (0.0, 1.0)]
    raw_entries = [e for e in raw_entries if e['alpha'] not in (0.0, 1.0)]
    avgs = {}
    avgs["Min_Max"] = np.mean([e['ndcg'] for e in min_max_entries])
    avgs["z_score"] = np.mean([e['ndcg'] for e in z_score_entries])
    avgs["log + Min_Max"] = np.mean([e['ndcg'] for e in log_min_max_entries])
    avgs["raw"] = np.mean([e['ndcg'] for e in raw_entries])

    best_norm = max(avgs, key=avgs.get)
    best_score = avgs[best_norm]
    dt.freeze_norm(best_norm, best_score)

    save_norm_table(min_max_entries, z_score_entries, log_min_max_entries, raw_entries, avgs, entry_ppr_raw_score, entry_only_cosine,f"Outputs/runs/{file}/tables", "norm_table")
def make_seed_selection_table(file):
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    cosine_entries = []
    bm25_entries = []
    fusion_entries = []
    for graph, methods in grouped.items():
        results = grouped[graph]['PPR']['all_samples']
        for result in results:
            params = result.pop("params", {})
            flat_entry = {
                "graph": graph,
                "method": 'PPR',
                "sample_type": 'all_samples',
                **result,
                **params
            }
            if flat_entry["seed_selection"] == "cosine":
                cosine_entries.append(flat_entry)
            elif flat_entry['seed_selection'] == "bm25":
                bm25_entries.append(flat_entry)
            elif flat_entry['seed_selection'] == "fusion":
                fusion_entries.append(flat_entry)

    cosine_entries = sorted(cosine_entries, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)
    bm25_entries = sorted(bm25_entries, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)
    fusion_entries = sorted(fusion_entries, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)
    avgs = {}
    avgs["cosine"] = (np.mean([e['ndcg'] for e in cosine_entries]), np.std([e['ndcg'] for e in cosine_entries]))
    avgs["bm25"] = (np.mean([e['ndcg'] for e in bm25_entries]), np.std([e['ndcg'] for e in bm25_entries]))
    avgs["fusion"] = (np.mean([e['ndcg'] for e in fusion_entries]), np.std([e['ndcg'] for e in fusion_entries]))
    best_seed = max(avgs, key=avgs.get)
    best_score = avgs[best_seed]
    dt.freeze_seed(best_seed, best_score)
    save_seed_selection_table(cosine_entries, bm25_entries, fusion_entries, avgs, f"Outputs/runs/{file}/tables", "seed_selection_table")

def make_random_state_table(file):

    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    entries_scores = defaultdict(dict)
    entries = defaultdict(list)
    for graph, methods in grouped.items():
        results = grouped[graph]['PPR']['all_samples']
        for result in results:
            graph_params = dt.load_graph_parameters(graph)
            random_state = graph_params[0]["clustering"]["random_state"]
            graph_name = graph_params[0]["graph_name"]
            params = result.pop("params", {})
            flat_entry = {
                "graph": graph,
                "method": 'PPR',
                "sample_type": 'all_samples',
                **result,
                **params,
                "random_state": random_state
            }

            entries[random_state].append(flat_entry)
            entries_scores[random_state][graph_name] = (flat_entry["ndcg"], flat_entry['recall'], flat_entry['mrr'])

    config_stats = {}
    for random_state, scores_by_graph in entries_scores.items():

        arr = np.array(list(scores_by_graph.values()))

        mean_ndcg, mean_recall, mean_mrr = arr.mean(axis=0)
        std_ndcg, std_recall, std_mrr = arr.std(axis=0)

        config_stats[random_state] = {
            "mean": {"ndcg": float(mean_ndcg), "recall": float(mean_recall), "mrr": float(mean_mrr)},
            "std": {"ndcg": float(std_ndcg), "recall": float(std_recall), "mrr": float(std_mrr)},
            "scores": scores_by_graph,
        }

    for random_state in entries:
        entries[random_state].sort(
            key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]),
            reverse=True
        )

    save_random_state_table(entries, config_stats, f"Outputs/runs/{file}/tables", "random_state_table")


def make_paired_bootstrap_table(file, method, sample_type, query_type,
                                  n_boot=10000, seed=42, alpha=0.05):
    """Fucntion that saves in the table file the results of paired bootstrap. The pairs are
       the queries results of baseline and graph methods. The results are evaluation metrics
    """
    # Getting the top 3 best performing graphs results
    top_3 = dt.get_top_3_best_performing_graphs(method, sample_type, file, query_type)
    eval_metrics = ["mrr", "recall", "ndcg", "map"]
    table = {}
    # Gathering the valid baseline query results
    for graph_info in top_3:
        graph_name = graph_info["graph_name"]
        queries = dt.get_query_metrics_for_a_graph(file, graph_info, query_type)
        queries_baseline = dt.find_matching_k_queries_in_baseline(
            graph_info["params"]["k"], query_type
        )

        eval_dict_graph = defaultdict(list)
        eval_dict_baseline = defaultdict(list)
        # Filling the graph and baseline dictionaries for each metric
        for query in queries.keys():
            if query not in queries_baseline:
                continue
            for metric in eval_metrics:
                if metric in queries[query] and metric in queries_baseline[query]:
                    eval_dict_graph[metric].append(queries[query][metric])
                    eval_dict_baseline[metric].append(queries_baseline[query][metric])

        # Paired Bootstrap
        results_recall = ev.paired_bootstrap_ci(eval_dict_baseline['recall'],eval_dict_graph['recall'],n_boot,seed,alpha)
        results_mrr = ev.paired_bootstrap_ci(eval_dict_baseline['mrr'], eval_dict_graph['mrr'], n_boot, seed, alpha)
        results_ndcg = ev.paired_bootstrap_ci(eval_dict_baseline['ndcg'], eval_dict_graph['ndcg'], n_boot, seed, alpha)
        results_map = ev.paired_bootstrap_ci(eval_dict_baseline['map'], eval_dict_graph['map'], n_boot, seed, alpha)
        table[graph_name] = {"recall": results_recall, "mrr": results_mrr, "ndcg":results_ndcg, "map":results_map}

    with open(f"Outputs/runs/{file}/tables/Paired_bootstrap.json", "w") as f:
        f.write(json.dumps(table) + "\n")
    # Forest plot
    pt.forest_plot(table, "all_samples", "PPR", "dev", file)

def make_spearmanr_table(file, method, sample_type, query_type):
     """Function that uses spearmanr to find if correlation exists between evaluation metrics and graph stats"""
     # Gathering best performing graphs results for each graph
     best_perf_graph = dt.get_each_graph_best_perf(method, sample_type, file, query_type)
     metrics = ["recall", "mrr", "ndcg", "map"]
     graph_stats = ["Edges", "Graph Density", "Number of connected components", "largest_component_size",
                    "Number of communities", "avg_clustering"]
     all_metrics = defaultdict(list)
     all_graph_stats = defaultdict(list)
     correlations = {}
     # Gathering all the values in all_metrics, all_graph_stats to use them later
     for graph in best_perf_graph:
         _, graph_info = dt.load_graph_parameters(graph["graph"])

         for metric in metrics:
             all_metrics[metric].append(graph[metric])

         for graph_stat in graph_stats:
             if graph_stat == "Number of communities":
                 all_graph_stats[graph_stat].append(graph_info[graph_stat][0])
             else:
                 all_graph_stats[graph_stat].append(graph_info[graph_stat])

     # spearmanr
     for graph_stat in graph_stats:
         for metric in metrics:
             rho, p_value = spearmanr(all_graph_stats[graph_stat], all_metrics[metric])
             correlations[f"{graph_stat}_{metric}"] = {"rho": rho, "p_value": p_value, "significant": "significant" if p_value < 0.05 else "not significant",
                                                       "metric": metric, "graph_stat": graph_stat}
     # Sorting and saving
     sorted_correlations = sorted(correlations.items(), key=lambda x: (x[1]["rho"], -x[1]["p_value"]), reverse=False)
     dt.freeze_ppr_spearmanr(sorted_correlations)
     with open(f"Outputs/runs/{file}/tables/spearmanr_correlations.txt", "w") as f:
         f.write("")
     with open(f"Outputs/runs/{file}/tables/spearmanr_correlations.txt", "a") as f:
         for name, stats in sorted_correlations:
             rho = stats["rho"]
             p = stats["p_value"]
             sig = stats["significant"]
             f.write(f"{name:35s} rho={rho:+.3f}  p={p:.5f}  ({sig})\n")


def make_ppr_tables(file, query_type):
    dt.seperate_results(file, query_type)
    make_query_table(file, query_type, "all")
    make_query_table(file, query_type, "one hop")
    make_query_table(file, query_type, "multi hop")
    make_leaderboard_table(file)
    make_alpha_sensitivity_table(file)
    make_init_sensitivity_table(file)
    make_graph_construction_sensitivity_table(file)
    make_spearmanr_table(file, "PPR", "all_samples", query_type)
    make_paired_bootstrap_table(file, "PPR", "all_samples", query_type)
def make_k_steph_tables(file, query_type):
    dt.seperate_results(file, query_type)
    make_query_table(file, query_type, "all")
    make_query_table(file, query_type, "one hop")
    make_query_table(file, query_type, "multi hop")
    make_leaderboard_table(file)
    make_alpha_sensitivity_table(file)
    make_init_sensitivity_table(file)
    make_graph_construction_sensitivity_table(file)
    make_paired_bootstrap_table(file, "k-steph", "all_samples", query_type)