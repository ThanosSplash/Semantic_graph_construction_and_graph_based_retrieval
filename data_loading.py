import json

import pandas as pd
import pickle
import networkx as nx
import os
import pyarrow
import numpy as np
import plotting
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import copy
import re
from collections import defaultdict

import tables

"""-----------------------------------------------------------------------------Dataset preperation-----------------------------------------------------------------------------"""

def read_dataset_bioasq(path):

    # Loading questions and answers
    dataset = pd.read_parquet(path + "question-answer-passages/train-00000-of-00001.parquet", engine='pyarrow')

    # Loading text corpus
    dataset_corpus = pd.read_parquet(path + "text-corpus/train-00000-of-00001.parquet", engine='pyarrow')

    # Checking for nan and na values
    print(dataset_corpus.loc[(dataset_corpus["passage"] == "nan") | dataset_corpus["passage"].isna(), "id"])
    return dataset, dataset_corpus


def save_data(question_emb, answer_emb, corpus_emb, questions_text, answer_text, corpus_text):

    data = (question_emb, answer_emb, corpus_emb)
    file_path = "Data/embeddings.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    data = (questions_text, answer_text, corpus_text)
    file_path = "Data/texts.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)



def save_samples(small, medium, long):
    file_path = "Data/small_samples" + ".pkl"
    with open(file_path, "wb") as f:
        pickle.dump(small, f)

    file_path = "Data/medium_samples" + ".pkl"
    with open(file_path, "wb") as f:
        pickle.dump(medium, f)

    file_path = "Data/long_samples" + ".pkl"
    with open(file_path, "wb") as f:
        pickle.dump(long, f)

def get_files(directory_name):
    dir_list = os.listdir(directory_name)
    print("Files and directories in '", len(dir_list), "' :")
    # prints all files
    return dir_list
def clear_eval(dir):
    if os.path.exists(dir):
        with open(dir, "w") as f:
            f.write("")
"""-----------------------------------------------------------------------------Dataset preperation-----------------------------------------------------------------------------"""

"""-----------------------------------------------------------------------------Graph Related-----------------------------------------------------------------------------"""
def save_graph(G, name):
    directory_name = f"Outputs/graphs/{name}/graph.gpickle"
    node_list = list(G.nodes())  # 1. Build the global list

    cache_obj = {
        "global_node_list": node_list,
        "node_to_idx": {node: idx for idx, node in enumerate(node_list)},
        "adjacency": nx.to_scipy_sparse_array(G, format='csr')
    }

    with open(directory_name, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    directory_name = f"Outputs/graphs/{name}/graph_cache.pkl"
    with open(directory_name, "wb") as f:
        pickle.dump(cache_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    directory_name = f"Outputs/graphs/{name}/graph_info.json"
    graph_info = plotting.print_graph_stats(G)

    json_str = json.dumps(graph_info, indent=len(graph_info.keys()))
    with open(directory_name, "w") as f:
        f.write(json_str)

def save_graph_data(data, graph_id, fig = None):
    directory_name = f"Outputs/graphs/{graph_id}"

    os.makedirs(directory_name, exist_ok=True)

    path = directory_name + "/graph_parameters" + ".json"
    if fig is not None:
        fig.savefig(f"{directory_name }/plot.png", dpi=300, bbox_inches="tight")
    json_str = json.dumps(data, indent=len(data.keys()))
    with open(path, "w") as f:
        f.write(json_str)


"""-----------------------------------------------------------------------------Graph Related-----------------------------------------------------------------------------"""








"""-----------------------------------------------------------------------------Load graph info-----------------------------------------------------------------------------"""
def load_graph(name):

    G = nx.Graph()
    directory_name = f"Outputs/graphs/{name}/graph.gpickle"
    with open(directory_name, "rb") as f:
        G = pickle.load(f)

    return G
def load_graph_cache(name):
    directory_name = f"Outputs/graphs/{name}/graph_cache.pkl"
    with open(directory_name, "rb") as f:
        graph_cache = pickle.load(f)
    return graph_cache
def load_graph_parameters(name):
    directory_name = f"Outputs/graphs/{name}/graph_parameters.json"
    with open(directory_name, "r", encoding="utf-8") as f:
        params = json.load(f)
    directory_name = f"Outputs/graphs/{name}/graph_info.json"
    with open(directory_name, "r", encoding="utf-8") as f:
        info = json.load(f)
    return params, info
"""-----------------------------------------------------------------------------Load graph info-----------------------------------------------------------------------------"""

"""-----------------------------------------------------------------------------Extract from retrieval results------------------------------------------------------------"""
def get_top_5_best_performing_graphs(method, sample_type, dir):
    # For a given method and sample type return the top best performing graphs
    files = get_files(dir)
    graphs_perf = []
    for file in files:
        seperate_results(f"{dir}/{file}/")
        grouped_results = load_grouped_results(f"{dir}/{file}/", 'by_recall')
        graphs_perf.append((file, grouped_results[method][sample_type][0]['recall']))
    graphs_perf.sort(key=lambda x: x[1], reverse=True)
    top_5_graphs = graphs_perf[:5]
    return [graph_name for graph_name, score in top_5_graphs]

def get_best_performing_graphs_by_method(method, sample_type, dir):
    # For a given method and a sample type gets for each graph the best performance based on recall
    files = get_files(dir)
    graphs_perf = []
    for file in files:
        seperate_results(f"{dir}/{file}/")
        grouped_results = load_grouped_results(f"{dir}/{file}/", 'by_recall')
        graphs_perf.append((file, grouped_results[method][sample_type][0]['recall']))
    graphs_perf.sort(key=lambda x: x[1], reverse=True)
    return dict(graphs_perf)

def get_best_performing_graphs_and_params_by_method(method, sample_type, dir):
    # For a given method and a sample type gets for each graph the best performance based on recall
    files = get_files(dir)
    graphs_perf = []
    for file in files:
        with open(f"{dir}/{file}/graph_info.json", "r", encoding="utf-8") as f:
            graph_info = json.load(f)
        with open(f"{dir}/{file}/graph_parameters.json", "r", encoding="utf-8") as f:
            graph_params = json.load(f)
        print(file)
        density = graph_info['Graph Density']
        components = graph_info['Number of connected components']
        graph_type = graph_params['graph_type']

        if 'clustering' in graph_params:
           if graph_params['Graph building algorithm'] == "Threshold":
               threshold = graph_params['Graph building algorithm params']['threshold_distance']
               clusters = graph_params['clustering']['n_clusters']
               graph = f"{graph_type} threshold = {threshold} clusters = {clusters}"
           else:
               clusters = graph_params['clustering']['n_clusters']
               neighbors = graph_params['Graph building algorithm params']['n_neighbors']
               graph = f"{graph_type} n = {neighbors} clusters = {clusters}"
        else:
          if graph_params['Graph building algorithm'] == "Threshold":
              threshold = graph_params['Graph building algorithm params']['threshold_distance']
              graph = f"{graph_type} threshold = {neighbors}"
          else:
               neighbors = graph_params['Graph building algorithm params']['n_neighbors']
               graph = f"{graph_type} n = {neighbors}"

        seperate_results(f"{dir}/{file}/")
        grouped_results = load_grouped_results(f"{dir}/{file}/", 'by_recall')[method][sample_type][0]
        graphs_perf.append((graph, {'recall':grouped_results['recall'], 'mrr':grouped_results['mrr'],
                                    'ndcg':grouped_results['ndcg'], 'map':grouped_results['map'] ,'density':density,'components':components}))

    return dict(graphs_perf)
def get_best_performnaces_ppr_k_steph(sample_type, dir):
    files = get_files(dir)
    graphs_perf = []
    for file in files:
        seperate_results(f"{dir}/{file}/")
        grouped_results = load_grouped_results(f"{dir}/{file}/", 'by_recall')
        scores_by_method = [grouped_results[method][sample_type][0]['recall'] for method in grouped_results.keys()]
        if len(scores_by_method) > 1:
            graphs_perf.append((file, scores_by_method))
    return dict(graphs_perf)

def get_best_performing_method_parms_from_a_graph(sample_type, dir, sorted_by):
    best_params = {}
    seperate_results(f"{dir}/")
    grouped_results = load_grouped_results(f"{dir}/", sorted_by)
    best_perf = grouped_results['PPR'][sample_type][0]
    #print(best_perf)
    best_params['method'] = 'PPR'
    best_params['params'] = best_perf['params']


    return best_params
def get_data_weighted_unweighted_undirected(method, sample_type):
    # Saves in a dictionary how weighted and unweighted graphs performed
    weighted_graphs_perf = get_best_performing_graphs_by_method(method, sample_type, f"Outputs/graphs")
    unweighted_graphs_perf = get_best_performing_graphs_by_method(method, sample_type, f"Outputs/Undirected_unweighted_graphs")

    graphs = list(weighted_graphs_perf.keys()) + list(unweighted_graphs_perf.keys())
    pairs = defaultdict(dict)

    for g in graphs:
        is_weighted = "Weighted_True" in g

        key = re.sub(r"_Directed_(True|False)_Weighted_(True|False)", "", g)

        if is_weighted:
            pairs[key]["weighted"] = (g, weighted_graphs_perf[g])
        else:
           pairs[key]["unweighted"] = (g, unweighted_graphs_perf[g])


    pairs = {
        k: v
        for k, v in pairs.items()
        if "weighted" in v and "unweighted" in v
    }

    return pairs


def get_data_undirected_directed_unweighted(method, sample_type):
    # Saves in a dictionary how directed and undirected graphs performed
    undirected_graphs_perf = get_best_performing_graphs_by_method(method, sample_type, f"Outputs/graphs")
    directed_graphs_perf = get_best_performing_graphs_by_method(method, sample_type, f"Outputs/Directed_unweighted_graphs")
    graphs = list(undirected_graphs_perf.keys()) + list(directed_graphs_perf.keys())
    pairs = defaultdict(dict)

    for g in graphs:
        is_directed = "Directed_True" in g

        key = re.sub(r"_Directed_(True|False)_Weighted_(True|False)", "", g)
        if is_directed:
               pairs[key]["directed"] = (g, directed_graphs_perf[g])
        else:
               pairs[key]["undirected"] = (g, undirected_graphs_perf[g])

    pairs = {
        k: v
        for k, v in pairs.items()
        if "directed" in v and "undirected" in v
    }


    return pairs

def get_data_undirected_directed_weighted(method, sample_type):
    # Saves in a dictionary how directed and undirected graphs performed
    undirected_graphs_perf = get_best_performing_graphs_by_method(method, sample_type, f"Outputs/graphs")
    directed_graphs_perf = get_best_performing_graphs_by_method(method, sample_type, f"Outputs/Directed_weighted_graphs")
    graphs = list(undirected_graphs_perf.keys()) + list(directed_graphs_perf.keys())
    pairs = defaultdict(dict)

    for g in graphs:
        is_directed = "Directed_True" in g
        key = re.sub(r"_Directed_(True|False)_Weighted_(True|False)", "", g)
        if is_directed:
                pairs[key]["directed"] = (g, directed_graphs_perf[g])
        else:
                pairs[key]["undirected"] = (g, undirected_graphs_perf[g])

    pairs = {
        k: v
        for k, v in pairs.items()
        if "directed" in v and "undirected" in v
    }

    return pairs
def get_data_clustering_not_clustering(method, sample_type):
    # For each graph type saves in a dictionary how clustering and not clustering performed
    graph_perf = get_best_performing_graphs_by_method(method, sample_type, f"Outputs/graphs")

    pairs = {
        'knn': {
            'clustering': [],
            'no_clustering': []
        },
        'mutual knn': {
            'clustering': [],
            'no_clustering': []
        },
        'threshold': {
            'clustering': [],
            'no_clustering': []
        }
    }

    for g, perf in graph_perf.items():
        params, _ = load_graph_parameters(g)
        graph_type = params['graph_type']

        if graph_type == 'kmeans knn graph':
            pairs['knn']['clustering'].append(perf)

        elif graph_type == 'knn graph':
            pairs['knn']['no_clustering'].append(perf)

        elif graph_type == 'kmeans mutual knn graph':
            pairs['mutual knn']['clustering'].append(perf)

        elif graph_type == 'mutual knn graph':
            pairs['mutual knn']['no_clustering'].append(perf)

        elif graph_type == 'kmeans threshold graph':
            pairs['threshold']['clustering'].append(perf)

        elif graph_type == 'threshold graph':
            pairs['threshold']['no_clustering'].append(perf)

    return pairs

def get_data_ppr_vs_ksteph(sample_type):
    # For each graph type saves in a dictionary how ppr and k-steph performed
    graph_perf = get_best_performnaces_ppr_k_steph(sample_type, f"Outputs/graphs")
    pairs = {}

    for g, perf in graph_perf.items():
        params, _ = load_graph_parameters(g)
        graph_type = params['graph_type']
        pairs [g] = {
                  'PPR': [],
                  'k-steph_graph_aware': []
             }
        pairs[g]['PPR'].append(perf[0])
        pairs[g]['k-steph_graph_aware'].append(perf[1])


    return pairs

def extract_best_run_and_compare_to_baseline(queries, baseline_queries, best_params, eval):
    if eval == 'rank':
        row_1 = 'rank'
        row_2 = 'Baseline_rank'
    elif eval == 'recall':
        row_1 = 'recall'
        row_2 = 'Baseline_recall'

    queries_vs_baseline = {}
    worst_count = 0
    same_count = 0
    improved_count = 0
    for query in queries.keys():
        results = queries[query]['results']
        for result in results:
            flag = True
            if result['method']!= best_params['method']:
                flag = False
                continue
            for param in best_params['params'].keys():
                if param not in result:
                    flag = False
                    break
                if result[param] != best_params['params'][param]:
                    flag = False
                    break
            if flag == True:
                # For each query keeps track of the baseline rank,recall,mrr , the difference and if the query had a better performance
                baseline = baseline_queries[query]['results'][0][row_1]

                if baseline < 0:
                    baseline = 0

                if result[row_1] < 0:
                    result[row_1] = 0
                if row_1 == 'recall':
                    diff = result[row_1] - baseline
                elif row_1 == 'rank':
                    diff = baseline - result[row_1]
                if diff < 0:
                    Case = 'Worst'
                    worst_count += 1
                elif diff == 0:
                    Case = 'Same'
                    same_count += 1
                else:
                    Case = 'Improved'
                    improved_count += 1
                queries_vs_baseline[query] = {'Method': result['method'], row_2: baseline, row_1: result[row_1], 'Difference': diff,
                                          "Case": Case}
    return queries_vs_baseline,{'Improved': improved_count, 'Same': same_count, 'Worst': worst_count}
"""-----------------------------------------------------------------------------Extract from retrieval results------------------------------------------------------------"""

"""-----------------------------------------------------------------------------Load from dataset-----------------------------------------------------------------------------"""

def load_data():

    file_path = "Data/embeddings.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)


    question, answer, corpus = data

    return question, answer, corpus

def load_texts():
    file_path = "Data/texts.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    question, answer, corpus = data
    return question, answer, corpus
def load_samples():
    file_path = "Data/small_samples" + ".pkl"
    with open(file_path, "rb") as f:
        small = pickle.load(f)

    file_path = "Data/medium_samples" + ".pkl"
    with open(file_path, "rb") as f:
        medium = pickle.load(f)

    file_path = "Data/long_samples" + ".pkl"
    with open(file_path, "rb") as f:
        long = pickle.load(f)

    return small, medium, long

def load_queries_results(dir):

    with open(f"{dir}/queries.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_grouped_results(dir, sorted_by):
    with open(f"{dir}/grouped_results.pkl", "rb") as f:
        grouped = pickle.load(f)
    return grouped[sorted_by]

def load_results_file(dir):
    # Read the eval_results.json file and saves them in a dictionary
    grouped = {}
    with open(dir) as f:
         for line in f:
            r = json.loads(line)
            sample_type = r["sample_type"]
            final = r["final scores"]
            params = {}
            method = r["method"]

            if "k" in r:
                params["k"] = int(r['k'])

            if "alpha" in r and r["alpha"] != "":
                params["alpha"] = float(r['alpha'])

            if "reranker" in r and r["reranker"] != "":
                params["reranker"] = r['reranker']
                method =f"{method}_{r['reranker']}"

            if "init" in r and r["init"] != "":
                params["init"] = int(r['init'])

            if "hops" in r and r["hops"] != "":
                params["hops"] = int(r['hops'])

            grouped.setdefault(method, {}).setdefault(sample_type, []).append({
                "recall": float(final["recallk"]),
                "mrr": float(final["mrr"]),
                "ndcg": float(final["ndcg"]),
                "map": float(final["mapk"]),
                "params": params
            })
    return grouped

def load_eval_results(name):
    directory_name = f"Outputs/graphs/{name}/eval_results.json"
    eval_results = []
    with open(directory_name) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eval_results.append(r)
    return eval_results
"""-----------------------------------------------------------------------------Load from dataset-----------------------------------------------------------------------------"""
"""-----------------------------------------------------------------------------Leaderboard-----------------------------------------------------------------------------"""

def convert_to_dict(obj):
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(v) for v in obj]
    else:
        return obj

def seperate_results(dir):
    grouped = load_results_file(f"{dir}/eval_results.json")
    grouped_by_recall = copy.deepcopy(grouped)
    grouped_by_mrr = copy.deepcopy(grouped)
    by_init = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_alpha = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for method, sample_types in grouped_by_mrr.items():
        for sample_type, rows in sample_types.items():
            rows.sort(key=lambda x: x["mrr"], reverse=True)


    for method, sample_types in grouped_by_recall.items():
        for sample_type, rows in sample_types.items():
            rows.sort(key=lambda x: x["recall"], reverse=True)

    for method, sample_types in grouped.items():
        for sample_type, results in sample_types.items():
            for r in results:
                init = r["params"]["init"]
                by_init[method][sample_type][init].append(r)


    for method, sample_types in grouped.items():
        for sample_type, results in sample_types.items():
            for r in results:
                alpha = r["params"]["alpha"]
                by_alpha[method][sample_type][alpha].append(r)

    results = {
        "by_recall": convert_to_dict(grouped_by_recall),
        "by_mrr": convert_to_dict(grouped_by_mrr),
        "by_init": convert_to_dict(by_init),
        "by_alpha": convert_to_dict(by_alpha)
    }
    with open(f"{dir}/grouped_results.pkl", "wb") as f:
        pickle.dump(results, f)


def save_avg_eval_results(method, final_scores, params, dir):
    record = {
        "method": method,
        "final scores": final_scores,
        **{k: v for k, v in params.items() if k != "graph"}
    }
    if method.lower() != "baseline":
        dir += "/eval_results.json"
    else:
        dir += "/baseline_results.json"
    with open(dir, "a") as f:
        f.write(json.dumps(record) + "\n")
def save_eval_results_for_each_query(params, eval_results_for_each_query, method, relevant_passage_ids_per_query,dir):
    directory_name = dir
    os.makedirs(directory_name, exist_ok=True)
    queries_file = f"{directory_name}/queries.json"

    if params["sample_type"] == "all_samples":
        return

    # Load existing data if file exists
    if os.path.exists(queries_file) and os.path.getsize(queries_file) != 0:
        with open(queries_file, "r") as f:
            all_queries = json.load(f)
    else:
        all_queries = {}

    # Update each query
    q, a, c = load_texts()
    for query_id, data in eval_results_for_each_query.items():
        if str(query_id) not in all_queries:
            all_queries[str(query_id)] = {"query_id": query_id, "results": [], "text": q[query_id],
                                          ",relevant_passage_ids": relevant_passage_ids_per_query[query_id]}

        all_queries[str(query_id)]["results"].append({
            "method": method,
            **params,
            **data
        })

    # Save back
    with open(queries_file, "w") as f:
        json.dump(all_queries, f, indent=2)
def save_eval_results(eval_results_for_each_query, params, method, final_scores, dir, relevant_passage_ids_per_query):

    save_avg_eval_results(method, final_scores, params, dir)

    save_eval_results_for_each_query(params, eval_results_for_each_query, method, relevant_passage_ids_per_query, dir)
"""-----------------------------------------------------------------------------Leaderboard-----------------------------------------------------------------------------"""
