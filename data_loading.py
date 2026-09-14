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
from pathlib import Path
import tables
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
"""-----------------------------------------------------------------------------Dataset preperation-----------------------------------------------------------------------------"""



def read_dataset_bioasq(path):
    # Loading questions and answers
    dataset = pd.read_parquet(path + "question-answer-passages/train-00000-of-00001.parquet", engine='pyarrow')
    dataset_test = pd.read_parquet("Datasets/rag-mini-bioasq/question-answer-passages/test-00000-of-00001.parquet",
                                   engine='pyarrow')
    # Loading text corpus
    dataset_corpus = pd.read_parquet(path + "text-corpus/train-00000-of-00001.parquet", engine='pyarrow')
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

def get_files_for_param(directory_name, param):
    dir_list = os.listdir(directory_name)
    if param == "preprocess":
        to_search = ["PCA()", "StandardScaler()"]
        dir_list_found = [s for s in dir_list if any(term in s for term in to_search)]
    elif param == "graph_construction":
        to_search = ["Directed_True_Weighted_True", "Directed_True_Weighted_False", "Directed_False_Weighted_False"]
        dir_list_found = [s for s in dir_list if any(term in s for term in to_search)]
    elif param == "metric":
        to_search = ["euclidean", "manhattan", "chebyshev"]
        dir_list_found = [s for s in dir_list if any(term in s for term in to_search)]
    else:
        raise ValueError(f"Wrong param {param}")

    return dir_list_found

def get_files_with_random_state(directory_name, random_state):
    dir_list = os.listdir(directory_name)
    dir_list_without_kmeans = [s for s in dir_list if "kmeans" not in s.lower()]
    dir_kmeans = get_kmeans_files_for_random_state(directory_name, random_state)
    final_dir_list = dir_list_without_kmeans + dir_kmeans
    print("Files and directories in '", len(final_dir_list), "' :")
    # prints all files
    return final_dir_list
def get_kmeans_files(directory_name):
    dir_list = os.listdir(directory_name)
    dir_list_kmeans = [s for s in dir_list if "kmeans" in s.lower() and "agglo_kmeans" not in s.lower()]
    print("Files and directories in '", len(dir_list), "' :")
    # prints all files
    return dir_list_kmeans

def get_agglo_files(directory_name):
    dir_list = os.listdir(directory_name)
    dir_list_agglo_kmeans = [s for s in dir_list if "agglo_kmeans" in s.lower()]
    print("Files and directories in '", len(dir_list), "' :")
    # prints all files
    return dir_list_agglo_kmeans
def get_kmeans_files_for_random_state(directory_name, random_state):
    dir_list = get_kmeans_files(directory_name)
    dir_list_seed = [s for s in dir_list if f"seed{random_state}" in s.lower()]
    return dir_list_seed
def clean_baseline(query_type):
    with open(f"Outputs/Baseline-RAG/queries_{query_type}.json", "w") as f:
        f.write("")
    with open(f"Outputs/Baseline-RAG/aggregate_results_{query_type}.jsonl", "w") as f:
        f.write("")
    with open(f"Outputs/Baseline-RAG/queries_{query_type}.jsonl", "w") as f:
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
    graph_info = plotting.calc_graph_stats(G)

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




def make_dev_test_splits():
    """Function that makes the dev and test splits"""
    # Make dev split
    query, _, _ = load_texts()
    make_split(query, 0.02, "dev")
    # Make test split
    query, _ = load_texts_tests()
    make_split(query, 0.7, "test")

def make_split(query, split_percentage, split_type):
    """Making splits for a given set of queries. The splits are based on lengths and the three groups are
       small, medium, large. After labeling all the query with the labels small, medium, long the splits are made
       by taking ids randomly.
    """
    vectorizer = TfidfVectorizer()
    analyzer = vectorizer.build_analyzer()
    query_labels = {}
    lengths = [len(analyzer(text)) for text in query.values()]
    p33, p66 = np.percentile(lengths, [33, 66])

    for qid, text in query.items():
        # Labeling the queries to small, medium , long
        query_length = len(analyzer(text))

        if query_length <= p33:
            label = "small"
        elif query_length <= p66:
            label = "medium"
        else:
            label = "long"

        query_labels[qid] = label

    # Determining each sample size
    total_sample = int(len(query_labels) * split_percentage)
    each_sample_size = total_sample // 3
    # Randomly taking the ids for sample
    rng = np.random.default_rng(42)
    small_pool = [qid for qid, l in query_labels.items() if l == "small"]
    medium_pool = [qid for qid, l in query_labels.items() if l == "medium"]
    long_pool = [qid for qid, l in query_labels.items() if l == "long"]
    small_queries = rng.choice(
        small_pool,
        size=min(each_sample_size, len(small_pool)),
        replace=False
    ).tolist()

    medium_queries = rng.choice(
        medium_pool,
        size=min(each_sample_size, len(medium_pool)),
        replace=False
    ).tolist()

    long_queries = rng.choice(
        long_pool,
        size=min(each_sample_size, len(long_pool)),
        replace=False
    ).tolist()

    # Saving the samples
    save_split(small_queries, medium_queries, long_queries, f"Data/splits", split_type)




"""-----------------------------------------------------------------------------Load graph info-----------------------------------------------------------------------------"""
def load_graph(name):

    G = nx.Graph()
    directory_name = f"{BASE_DIR}/Outputs/graphs/{name}/graph.gpickle"
    with open(directory_name, "rb") as f:
        G = pickle.load(f)

    return G
def load_graph_cache(name):
    directory_name = f"{BASE_DIR}/Outputs/graphs/{name}/graph_cache.pkl"
    with open(directory_name, "rb") as f:
        graph_cache = pickle.load(f)
    return graph_cache
def load_graph_parameters(name):
    directory_name = f"{BASE_DIR}/Outputs/graphs/{name}/graph_parameters.json"
    with open(directory_name, "r", encoding="utf-8") as f:
        params = json.load(f)
    directory_name = f"Outputs/graphs/{name}/graph_info.json"
    with open(directory_name, "r", encoding="utf-8") as f:
        info = json.load(f)
    return params, info
"""-----------------------------------------------------------------------------Load graph info-----------------------------------------------------------------------------"""

"""-----------------------------------------------------------------------------Extract from retrieval results------------------------------------------------------------"""
def get_top_3_best_performing_graphs(method, sample_type, file, query_type):
    # For a given method and sample type return the top best performing graphs

    graphs_perf = []
    seperate_results(file, query_type)
    grouped_results = load_grouped(f"Outputs/runs/{file}")
    for graph, methods in grouped_results.items():
        params = grouped_results[graph][method][sample_type][0].pop("params", {})
        flat_entry = {
            "graph": graph,
            "method": method,
            "sample_type": sample_type,
            **grouped_results[graph][method][sample_type][0],
            "params":params
        }
        graphs_perf.append(flat_entry)

    top_3_graphs = sorted(graphs_perf, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)[:3]
    return top_3_graphs
def get_top_3_best_performing_graphs_for_alpha(method, sample_type, file, query_type, alpha):
    # For a given method and sample type return the top best performing graphs

    graphs_perf = []
    seperate_results(file, query_type)
    grouped_results = load_grouped(f"Outputs/runs/{file}")
    for graph, methods in grouped_results.items():
        entries = grouped_results[graph][method][sample_type]

        # find the entry that matches the requested alpha
        entry = next(
            (e for e in entries if e.get("params", {}).get("alpha") == alpha),
            None
        )
        if entry is None:
            continue

        params = entry.pop("params", {})

        flat_entry = {
            "graph": graph,
            "method": method,
            "sample_type": sample_type,
            **entry,
            "params": params,
        }
        graphs_perf.append(flat_entry)

    top_3_graphs = sorted(graphs_perf, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)[:3]
    return top_3_graphs
def get_each_graph_best_perf(method, sample_type, file, query_type):
    # For a given method and sample type return the top best performing graphs

    graphs_perf = []
    seperate_results(file, query_type)
    grouped_results = load_grouped(f"Outputs/runs/{file}")
    for graph, methods in grouped_results.items():
        params = grouped_results[graph][method][sample_type][0].pop("params", {})
        flat_entry = {
            "graph": graph,
            "method": method,
            "sample_type": sample_type,
            **grouped_results[graph][method][sample_type][0],
            "params":params
        }
        graphs_perf.append(flat_entry)

    graphs_best_perf = sorted(graphs_perf, key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]), reverse=True)
    return graphs_best_perf

def get_query_metrics_for_a_graph(file, graph, query_type):
    queries = load_queries_results(f"Outputs/runs/{file}", query_type)

    best_params = graph["params"]
    valid_queries = {}
    for query in queries.keys():
        multi_hop = False
        results = queries[query]['results']
        relevant_passage_ids = queries[query]['relevant_passage_ids']
        for result in results:
            flag = True
            if result["graph"] == graph["graph"]:
                for param in best_params.keys():
                    if param not in result:
                        flag = False
                        break
                    if result[param] != best_params[param]:
                        flag = False
                        break
                if flag == True:
                    valid_queries[query] = result

    return valid_queries

"""-----------------------------------------------------------------------------Extract from retrieval results------------------------------------------------------------"""

"""-----------------------------------------------------------------------------Load from dataset-----------------------------------------------------------------------------"""

def load_data():

    file_path = f"{BASE_DIR}/Data/embeddings.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)


    question, answer, corpus = data

    return question, answer, corpus

def load_data_tests():
    file_path = f"{BASE_DIR}/Data/embeddings_test.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    question, answer = data

    return question, answer

def load_texts():
    file_path = f"{BASE_DIR}/Data/texts.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    question, answer, corpus = data
    return question, answer, corpus

def load_texts_tests():
    file_path = f"{BASE_DIR}/Data/texts_test.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    question, answer = data
    return question, answer
def load_splits_dev():
    file_path = f"{BASE_DIR}/Data/splits/small_split_dev.pkl"
    with open(file_path, "rb") as f:
        small = pickle.load(f)

    file_path = f"{BASE_DIR}/Data/splits/medium_split_dev.pkl"
    with open(file_path, "rb") as f:
        medium = pickle.load(f)

    file_path = f"{BASE_DIR}/Data/splits/long_split_dev.pkl"
    with open(file_path, "rb") as f:
        long = pickle.load(f)

    return small, medium, long

def load_splits_test():
    file_path = f"{BASE_DIR}/Data/splits/small_split_test.pkl"
    with open(file_path, "rb") as f:
        small = pickle.load(f)

    file_path = f"{BASE_DIR}/Data/splits/medium_split_test.pkl"
    with open(file_path, "rb") as f:
        medium = pickle.load(f)

    file_path = f"{BASE_DIR}/Data/splits/long_split_test.pkl"
    with open(file_path, "rb") as f:
        long = pickle.load(f)

    return small, medium, long
def load_queries_results(dir, query_type):

    with open(f"{dir}/queries_{query_type}.json", "r", encoding="utf-8") as f:
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
            latency = r['latency']
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
            if "norm" in r:
                params["norm"] = r['norm']
            if "seed_selection" in r:
                params["seed_selection"] = r['seed_selection']

            grouped.setdefault(graph, {}).setdefault(method, {}).setdefault(sample_type, []).append({
                "recall": float(r["recallk"]),
                "mrr": float(r["mrr"]),
                "ndcg": float(r["ndcg"]),
                "map": float(r["mapk"]),
                "graph_name": graph_name,
                "density": density,
                "components": components,
                "latency": latency,
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
def seperate_results(dir, query_type):
    grouped = load_results_file(f"Outputs/runs/{dir}/aggregate_results_{query_type}.jsonl")
    grouped_by_weighted_score = copy.deepcopy(grouped)
    for graph, methods in grouped_by_weighted_score.items():
        for method, sample_types in methods.items():
            for sample_type, rows in sample_types.items():
                rows.sort(
                    key=lambda x: (x["ndcg"], x["recall"], x["mrr"], -x["latency"]),
                    reverse=True
                )

    with open(f"Outputs/runs/{dir}/grouped.pkl", "wb") as f:
        pickle.dump(grouped_by_weighted_score, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_aggregate_record(method, final_scores, params, dir, query_type):
    record = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        **final_scores,
        **{k: v for k, v in params.items()}
    }
    dir += f"/aggregate_results_{query_type}.jsonl"
    with open(dir, "a") as f:
        f.write(json.dumps(record) + "\n")
def save_eval_results_for_each_query(params, eval_results_for_each_query, method,
                                     relevant_passage_ids_per_query, dir, query_type):
    directory_name = dir
    os.makedirs(directory_name, exist_ok=True)
    queries_file = f"{directory_name}/queries_{query_type}.json"

    if params["sample_type"] == "all_samples":
        return

    # Load existing data if file exists
    if os.path.exists(queries_file) and os.path.getsize(queries_file) != 0:
        with open(queries_file, "r") as f:
            all_queries = json.load(f)
    else:
        all_queries = {}

    # Update each query
    if query_type == "dev":
        q, _, _ = load_texts()
    elif query_type == "test":
        q, _ = load_texts_tests()
    else:
        raise ValueError(f"Wrong query type {query_type}")
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
    suffix = uuid.uuid4().hex[:8]
    parts = [date_str, experiment_name, suffix]
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

def sanity_check(queries_ids, dir, query_type):
    info = {}
    if query_type == "dev":
        queries, _, corpus = load_data()
    elif query_type == "test":
        _, _, corpus = load_data()
        queries, _ = load_data_tests()
    else:
        raise ValueError(f"Wrong type of query {query_type}")

    all_ids_in_corpus = set(corpus.keys())
    all_gold_ids = set()
    zero_reachable = 0
    for id in queries_ids:
        relevant_id = queries[id][1]
        all_gold_ids.update(relevant_id)
        if len(set(relevant_id) & all_ids_in_corpus) == 0:
            zero_reachable += 1


    gold_in_corpus = [id for id in all_gold_ids if id in all_ids_in_corpus]
    gold_missing = [id for id in all_gold_ids if id not in all_ids_in_corpus]

    info["number of evaluation queries"] = len(queries_ids)
    info["total unique gold passage ids"] = len(all_gold_ids)
    info["number gold ids present in corpus"] = len(gold_in_corpus)
    info["number gold ids missing from corpus"] = len(gold_missing)
    if len(all_gold_ids) == 0:
        info["coverage percentage"] = 0
    else:
        info["coverage percentage"] = len(gold_in_corpus)/len(all_gold_ids)

    info["queries with zero reachable gold passages"] = zero_reachable
    save_coverage_report(info, dir, f"coverage_report_{query_type}.json")
"""-----------------------------------------------------------------------------Experiments setup-----------------------------------------------------------------------------"""
"""-----------------------------------------------------------------------------For query table------------------------------------------------------------------------------------------------------------------------------"""
def find_matching_k_in_baseline(baseline_results, k, query_id):
    """Function that for a given query finds the baseline query result where k is equal to a given k"""
    for result in baseline_results:
        if result['k'] == k:
            return result
    raise ValueError(
        f"Expected one baseline for query={query_id}, k={k}"
    )

def find_matching_k_queries_in_baseline(k, query_type):
    """Function that return for each query the results where k is equal to given k"""
    queries = load_queries_results(f"Outputs/Baseline-RAG", query_type)

    valid_queries = {}
    for query in queries.keys():
        results = queries[query]['results']
        for result in results:
            flag = True
            if result['k'] != k:
                flag = False
            if flag == True:
                    valid_queries[query] = result
    return valid_queries
def extract_best_run_and_compare_to_baseline(queries, baseline_queries, best_params, eval, query_category):
    """Fucntion that collects for each query their best performance for the graph retrieval method and matches it with the baseline
       using the function find_matching_k_baseline_result
    """
    row_1 = eval
    row_2 = f'baseline_{eval}'

    queries_vs_baseline = {}
    worst_count = 0
    same_count = 0
    improved_count = 0


    for query in queries.keys():
        results = queries[query]['results']
        relevant_passage_ids = queries[query]['relevant_passage_ids']
        # Collects only valid queries for the given category
        if query_category == "one hop" and len(relevant_passage_ids) != 1:
            continue
        elif query_category == "multi hop" and len(relevant_passage_ids) <= 1:
            continue
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
                baseline_results = find_matching_k_in_baseline(baseline_queries[query]['results'], best_params['k'], query)
                baseline = baseline_results[row_1]


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

def freeze_norm(best_norm, score):
    with open(f"{BASE_DIR}/Outputs/freeze/norm_freeze.json", "w") as f:
        f.write(json.dumps({"Norm": best_norm, "ndcg_score": score}) + "\n")
def freeze_seed(best_seed, score):
    with open(f"{BASE_DIR}/Outputs/freeze/seed_freeze.json", "w") as f:
        f.write(json.dumps({"Seed selection": best_seed, "ndcg_score": score}) + "\n")

def freeze_random_state(random_state, scores):
    with open(f"{BASE_DIR}/Outputs/freeze/random_state_freeze.json", "w") as f:
        f.write(json.dumps({"random state": random_state, "ndcg_score": scores}) + "\n")

def freeze_ppr_configs(file, sample_type, query_type):
    top_3_fusion = get_top_3_best_performing_graphs("PPR", sample_type, file, query_type)
    ppr_config_fusion = {
        "graph" : top_3_fusion[0]["graph"],
        "method": top_3_fusion[0]["method"],
        "sample_type": top_3_fusion[0]["sample_type"],
        "params": top_3_fusion[0]["params"]
    }
    with open(f"{BASE_DIR}/Outputs/freeze/ppr_fusion_configs_freeze_{sample_type}.json", "w") as f:
        f.write(json.dumps(ppr_config_fusion) + "\n")

    top_3 = get_top_3_best_performing_graphs_for_alpha("PPR", sample_type, file, query_type, 0.0)
    ppr_config = {
        "graph": top_3[0]["graph"],
        "method": top_3[0]["method"],
        "sample_type": top_3[0]["sample_type"],
        "params": top_3[0]["params"]
    }
    with open(f"{BASE_DIR}/Outputs/freeze/ppr_configs_freeze_{sample_type}.json", "w") as f:
        f.write(json.dumps(ppr_config) + "\n")

def freeze_k_steph_configs(file, sample_type, query_type):
    top_5_fusion = get_top_3_best_performing_graphs("k-steph", sample_type, file, query_type)
    k_steph_config = {
        "graph": top_5_fusion[0]["graph"],
        "method": top_5_fusion[0]["method"],
        "sample_type": top_5_fusion[0]["sample_type"],
        "params": top_5_fusion[0]["params"]
    }
    with open(f"{BASE_DIR}/Outputs/freeze/k_steph_configs_freeze_{sample_type}.json", "w") as f:
        f.write(json.dumps(k_steph_config) + "\n")
def freeze_ppr_spearmanr(spearman_correlation):

    with open(f"{BASE_DIR}/Outputs/freeze/spearmanr_correlation_freeze.json", "w") as f:
        f.write(json.dumps(spearman_correlation) + "\n")
def freeze_dbscan_configs(min_samples, eps, sl_score, num_of_clusters):
    with open(f"{BASE_DIR}/Outputs/freeze/dbscan_config.json", "w") as f:
        f.write(json.dumps({"eps": eps, "min_samples": min_samples, "sl_score": sl_score, "clusters": num_of_clusters}) + "\n")
def get_freeze_norm():
    with open(f"{BASE_DIR}/Outputs/freeze/norm_freeze.json", "r", encoding="utf-8") as f:
            norm = json.load(f)
    return norm["Norm"]
def get_freeze_seed():
    with open(f"{BASE_DIR}/Outputs/freeze/seed_freeze.json", "r", encoding="utf-8") as f:
            norm = json.load(f)
    return norm["Seed selection"]

def get_freeze_random_state():
    with open(f"{BASE_DIR}/Outputs/freeze/random_state_freeze.json", "r", encoding="utf-8") as f:
            norm = json.load(f)
    return norm["random state"]
def get_freeze_ppr_fusion_configs(sample_type):
    with open(f"{BASE_DIR}/Outputs/freeze/ppr_fusion_configs_freeze_{sample_type}.json", "r", encoding="utf-8") as f:
            ppr_configs = json.load(f)
    return ppr_configs
def get_freeze_ppr_configs(sample_type):
    with open(f"{BASE_DIR}/Outputs/freeze/ppr_configs_freeze_{sample_type}.json", "r", encoding="utf-8") as f:
            ppr_configs = json.load(f)
    return ppr_configs
def get_freeze_k_steph_configs(sample_type):
    with open(f"{BASE_DIR}/Outputs/freeze/k_steph_configs_freeze_{sample_type}.json", "r", encoding="utf-8") as f:
            k_steph_configs = json.load(f)
    return k_steph_configs

def get_dbscan_configs():
    with open(f"{BASE_DIR}/Outputs/freeze/dbscan_config.json", "r", encoding="utf-8") as f:
            dbscan_configs = json.load(f)
    return dbscan_configs

def get_spearmanr_freeze():
    with open(f"{BASE_DIR}/Outputs/freeze/spearmanr_correlation_freeze.json", "r") as f:
        spearmanr_correlation = json.load(f)
    return spearmanr_correlation