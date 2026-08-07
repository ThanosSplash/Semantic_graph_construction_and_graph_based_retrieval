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
from datetime import datetime

import tables

"""-----------------------------------------------------------------------------Dataset preperation-----------------------------------------------------------------------------"""


def is_corpus_in_relevant(dataset, corpus):
    corpus_ids_set = set(corpus["id"])
    total_del = 0
    updated_lists = []
    for row in tqdm(dataset.itertuples(), total=len(dataset)):
            ids_to_check = set(row.relevant_passage_ids)
            not_found = ids_to_check - corpus_ids_set
            total_del += len(not_found)
            relevant_passage_ids_updated = [x for x in row.relevant_passage_ids if x not in not_found]
            updated_lists.append(relevant_passage_ids_updated)

    dataset["relevant_passage_ids"] = updated_lists
    print(f"Deleted {total_del}")
    return dataset



def read_dataset_bioasq(path):

    # Loading questions and answers
    dataset = pd.read_parquet(path + "question-answer-passages/train-00000-of-00001.parquet", engine='pyarrow')
    dataset_test = pd.read_parquet("Datasets/rag-mini-bioasq/question-answer-passages/train-00000-of-00001.parquet",
                                   engine='pyarrow')

    # Loading text corpus
    dataset_corpus = pd.read_parquet(path + "text-corpus/train-00000-of-00001.parquet", engine='pyarrow')


    # Checking for nan and na values
    na_corpus = dataset_corpus.loc[dataset_corpus["passage"].isna()
            | (dataset_corpus["passage"].astype(str).str.strip().str.lower() == "nan")
            | (dataset_corpus["passage"].astype(str).str.strip() == ""), "id"]

    na_questions = dataset.loc[dataset["question"].isna()
                       | (dataset["question"].astype(str).str.strip().str.lower() == "nan")
                       | (dataset["question"].astype(str).str.strip() == ""), "id"]
    na_questions_test = dataset_test.loc[dataset_test["question"].isna()
                       | (dataset_test["question"].astype(str).str.strip().str.lower() == "nan")
                       | (dataset_test["question"].astype(str).str.strip() == ""), "id"]

    print(f"na corpus {na_corpus} , na_questions {na_questions} ,  na_questions_test {na_questions_test}")
    dataset_corpus = dataset_corpus[~dataset_corpus["id"].isin(na_corpus)]
    dataset = dataset[~dataset["id"].isin(na_questions)]
    dataset_test = dataset_test[~dataset_test["id"].isin(na_questions_test)]


    is_corpus_in_relevant(dataset, dataset_corpus)
    is_corpus_in_relevant(dataset_test, dataset_corpus)
    return dataset, dataset_corpus, dataset_test


def save_data(dataset_embs, dataset_text, dataset_test_emb, dataset_test_text):

    data = (dataset_embs[0], dataset_embs[1], dataset_embs[2])
    file_path = "Data/embeddings.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    data = (dataset_test_emb[0], dataset_test_emb[1])
    file_path = "Data/embeddings_test.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    data = (dataset_text[0], dataset_text[1], dataset_text[2])
    file_path = "Data/texts.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    data = (dataset_test_text[0], dataset_test_text[1])
    file_path = "Data/texts_test.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)



def save_split(small, medium, long, dir, type):
    file_path = f"{dir}/small_split_{type}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(small, f)

    file_path = f"{dir}/medium_split_{type}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(medium, f)

    file_path = f"{dir}/long_split_{type}.pkl"
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
        grouped_results = load_grouped(f"{dir}/{file}/")
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
        grouped_results = load_grouped(f"{dir}/{file}/")
        if method in grouped_results:
           graphs_perf.append((file, grouped_results[method][sample_type][0]['recall']))
    graphs_perf.sort(key=lambda x: x[1], reverse=True)
    return dict(graphs_perf)



    return dict(graphs_perf)
def get_best_performnaces_ppr_k_steph(sample_type, dir):
    files = get_files(dir)
    graphs_perf = []
    for file in files:
        seperate_results(f"{dir}/{file}/")
        grouped_results = load_grouped(f"{dir}/{file}/", 'by_recall')
        scores_by_method = [grouped_results[method][sample_type][0]['recall'] for method in grouped_results.keys()]
        if len(scores_by_method) > 1:
            graphs_perf.append((file, scores_by_method))
    return dict(graphs_perf)


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


"""-----------------------------------------------------------------------------Extract from retrieval results------------------------------------------------------------"""

"""-----------------------------------------------------------------------------Load from dataset-----------------------------------------------------------------------------"""

def load_data():

    file_path = "Data/embeddings.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)


    question, answer, corpus = data

    return question, answer, corpus

def load_data_tests():
    file_path = "Data/embeddings_test.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    question, answer = data

    return question, answer

def load_texts():
    file_path = "Data/texts.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    question, answer, corpus = data
    return question, answer, corpus

def load_texts_tests():
    file_path = "Data/texts_test.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    question, answer = data
    return question, answer
def load_splits_dev():
    file_path = f"Data/splits/small_split_dev.pkl"
    with open(file_path, "rb") as f:
        small = pickle.load(f)

    file_path = f"Data/splits/medium_split_dev.pkl"
    with open(file_path, "rb") as f:
        medium = pickle.load(f)

    file_path = f"Data/splits/long_split_dev.pkl"
    with open(file_path, "rb") as f:
        long = pickle.load(f)

    return small, medium, long

def load_splits_test():
    file_path = f"Data/splits/small_split_test.pkl"
    with open(file_path, "rb") as f:
        small = pickle.load(f)

    file_path = f"Data/splits/medium_split_test.pkl"
    with open(file_path, "rb") as f:
        medium = pickle.load(f)

    file_path = f"Data/splits/long_split_test.pkl"
    with open(file_path, "rb") as f:
        long = pickle.load(f)

    return small, medium, long
def load_queries_results(dir):

    with open(f"{dir}/queries.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_grouped(dir):
    with open(f"{dir}/grouped.pkl", "rb") as f:
        grouped = pickle.load(f)
    return grouped

def load_results_file(dir):
    # Read the eval_results.json file and saves them in a dictionary
    grouped = {}
    with open(dir) as f:
         for line in f:
            r = json.loads(line)
            sample_type = r["sample_type"]
            params = {}
            method = r["method"]
            graph = r['graph']
            graph_params, graph_info = load_graph_parameters(graph)
            graph_name = graph_params['graph_name']
            density = graph_info["Graph Density"]
            components = graph_info["Number of connected components"]
            if "k" in r:
                params["k"] = int(r['k'])

            if "alpha" in r and r["alpha"] != "":
                params["alpha"] = float(r['alpha'])

            if "reranker" in r and r["reranker"] != "":
                params["reranker"] = r['reranker']

            if "init" in r and r["init"] != "":
                params["init"] = int(r['init'])

            if "hops" in r and r["hops"] != "":
                params["hops"] = int(r['hops'])

            grouped.setdefault(graph, {}).setdefault(method, {}).setdefault(sample_type, []).append({
                "recall": float(r["recallk"]),
                "mrr": float(r["mrr"]),
                "ndcg": float(r["ndcg"]),
                "map": float(r["mapk"]),
                "graph_name": graph_name,
                "density": density,
                "components": components,
                "weighted_score": 0.4*float(r["ndcg"]) + 0.3*float(r["recallk"]) + 0.3*float(r["mrr"]),
                "params": params
            })
    return grouped

"""-----------------------------------------------------------------------------Load from dataset-----------------------------------------------------------------------------"""


"""-----------------------------------------------------------load from records-----------------------------------------------------------"""
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
def seperate_results(dir):
    grouped = load_results_file(f"Outputs/runs/{dir}/aggregate_results.jsonl")
    grouped_by_weighted_score = copy.deepcopy(grouped)
    for graph, methods in grouped_by_weighted_score.items():
        for method, sample_types in methods.items():
            for sample_type, rows in sample_types.items():
                rows.sort(key=lambda x: x["weighted_score"], reverse=True)

    with open(f"Outputs/runs/{dir}/grouped.pkl", "wb") as f:
        pickle.dump(grouped_by_weighted_score, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_aggregate_record(method, final_scores, params, dir):
    record = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        **final_scores,
        **{k: v for k, v in params.items()}
    }
    dir += "/aggregate_results.jsonl"
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
                                          "relevant_passage_ids": relevant_passage_ids_per_query[query_id]}

        all_queries[str(query_id)]["results"].append({
            "method": method,
            **params,
            **data
        })

    # Save back
    with open(queries_file, "w") as f:
        json.dump(all_queries, f, indent=2)
"""-----------------------------------------------------------load from records-----------------------------------------------------------"""

"""-----------------------------------------------------------------------------Experiments setup-----------------------------------------------------------------------------"""
def make_run_id(experiment_name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    parts = [date_str, experiment_name]
    return "_".join(parts)

def setup_run_dir(run_id, notes, splits_used, params, base_dir = "Outputs/runs"):
    run_dir = f"{base_dir}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    info = {}
    info["notes"] = notes
    info["splits"] = splits_used
    info['params'] = params
    with open(f"{run_dir}/config.json", "w") as f:
        f.write(json.dumps(info) + "\n")

    return run_dir

def save_coverage_report(coverage_report, dir, name):
    with open(f"{dir}/{name}", "w") as f:
        f.write(json.dumps(coverage_report) + "\n")

def sanity_check(queries_ids, dir):
    info = {}
    queries, _, corpus = load_data()

    all_gold_ids = queries_ids
    all_ids_in_corpus = set(corpus.keys())
    all_relevant_ids = set()
    for id in all_gold_ids:
        all_relevant_ids.update(queries[id][1])


    gold_in_corpus = [id for id in all_relevant_ids if id in all_ids_in_corpus]
    gold_missing = [id for id in all_relevant_ids if id not in all_ids_in_corpus]

    info["number of evaluation queries"] = len(all_gold_ids)
    info["total unique gold passage ids"] = len(all_relevant_ids)
    #info["gold ids present in corpus"] =  gold_in_corpus
    info["number gold ids present in corpus"] = len(gold_in_corpus)
    #info["gold ids missing from corpus"] =  gold_missing
    info["number gold ids missing from corpus"] = len(gold_missing)
    if len(all_relevant_ids) == 0:
        info["coverage percentage"] = 0
    else:
        info["coverage percentage"] = len(gold_in_corpus)/len(all_relevant_ids)

    zero_reachable = 0
    for id in all_gold_ids:
        g_gold = set(queries[id][1])
        if len(g_gold.intersection(corpus.keys())) == 0:
            zero_reachable += 1

    info["queries with zero reachable gold passages"] = zero_reachable
    save_coverage_report(info, dir, "coverage_report.json")
"""-----------------------------------------------------------------------------Experiments setup-----------------------------------------------------------------------------"""
"""-----------------------------------------------------------------------------For query table------------------------------------------------------------------------------------------------------------------------------"""
def extract_best_run_and_compare_to_baseline(queries, baseline_queries, best_params, eval):
    row_1 = 'rank'
    row_2 = 'baseline_rank'

    queries_vs_baseline = {}
    worst_count = 0
    same_count = 0
    improved_count = 0
    print(queries.keys())
    for query in queries.keys():
        print(query)
        results = queries[query]['results']
        for result in results:
            flag = True
            for param in best_params.keys():
                if param not in result:
                    flag = False
                    break
                if result[param] != best_params[param]:
                    flag = False
                    break
            if flag == True:
                # For each query keeps track of the baseline rank,recall,mrr , the difference and if the query had a better performance
                baseline_results = sorted(baseline_queries[query]['results'], key=lambda x: x["rank"], reverse=True)
                baseline = baseline_results[0]['rank']
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
                queries_vs_baseline[query] = {'method': result['method'], row_2: baseline, row_1: result[row_1], 'difference': diff,
                                          "case": Case}
    return queries_vs_baseline,{'Improved': improved_count, 'Same': same_count, 'Worst': worst_count}
def extract_best_run_and_compare_to_baseline_recall(queries, baseline_queries, best_params, eval):
    row_1 = 'recall'
    row_2 = 'baseline_recall'

    queries_vs_baseline = {}
    worst_count = 0
    same_count = 0
    improved_count = 0
    print(queries.keys())
    for query in queries.keys():
        print(query)
        results = queries[query]['results']
        for result in results:
            flag = True
            for param in best_params.keys():
                if param not in result:
                    flag = False
                    break
                if result[param] != best_params[param]:
                    flag = False
                    break
            if flag == True:
                # For each query keeps track of the baseline rank,recall,mrr , the difference and if the query had a better performance
                baseline_results = sorted(baseline_queries[query]['results'], key=lambda x: x[row_1], reverse=True)
                baseline = baseline_results[0][row_1]
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
                queries_vs_baseline[query] = {'method': result['method'], row_2: baseline, row_1: result[row_1], 'difference': diff,
                                          "case": Case}
    return queries_vs_baseline,{'Improved': improved_count, 'Same': same_count, 'Worst': worst_count}
"""-----------------------------------------------------------------------------For query table------------------------------------------------------------------------------------------------------------------------------"""