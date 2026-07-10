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

"""-----------------------------------------------------------------------------Dataset preperation-----------------------------------------------------------------------------"""

"""-----------------------------------------------------------------------------Graph Related-----------------------------------------------------------------------------"""
def save_graph(G, name):
    directory_name = f"Outputs/graphs/{name}/graph.gpickle"
    node_list = list(G.nodes())  # 1. Build the global list

    cache_obj = {
        # 1. The list preserves the exact order of the matrix rows/columns
        "global_node_list": node_list,

        # 2. The dictionary maps Node ID -> Matrix Index Position
        "node_to_idx": {node: idx for idx, node in enumerate(node_list)},

        # 3. The sparse matrix stores the edges using those Index Positions
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


def save_eval_results(indexes, params, method, final_scores, results_file):

    record = {
        "method": method,
        "final scores": final_scores,
        **params
    }
    if method.lower() != "baseline":
        results_file += "/eval_results.json"
    else:
        print("global")
        results_file += "/global_eval_records.json"
    with open(results_file, "a") as f:
        f.write(json.dumps(record) + "\n")


    directory_name = "Outputs/Queries"
    os.makedirs(directory_name, exist_ok=True)
    queries_file = f"{directory_name}/queries.json"

    # Load existing data if file exists
    if os.path.exists(queries_file) and os.path.getsize(queries_file) != 0:
        with open(queries_file, "r") as f:
            all_queries = json.load(f)
    else:
        all_queries = {}


        # Update each query
    for query_id, data in indexes.items():
        if str(query_id) not in all_queries:

            all_queries[str(query_id)] = {"query_id": query_id, "results": []}

        all_queries[str(query_id)]["results"].append({
            "method": method,
            "model": results_file,
            **params,
            **data
        })

    # Save back
    with open(queries_file, "w") as f:
        json.dump(all_queries, f, indent=2)





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


def get_files():
    directory_name = "Outputs/graphs"
    dir_list = os.listdir(directory_name)
    print("Files and directories in '", len(dir_list), "' :")
    # prints all files
    return dir_list

"""-----------------------------------------------------------------------------Graph Related-----------------------------------------------------------------------------"""
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



"""-----------------------------------------------------------------------------Load from dataset-----------------------------------------------------------------------------"""
def clear_eval(path):
    if os.path.exists(path):
        with open(path, "w") as f:
            f.write("")
        print(f"file {path} cleared")
    else:

        print(f"file not found: {os.path.abspath(path)}")

def graph_perf(file):
    # Opening the queries json file
    path = f"Outputs/Queries/queries.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Turn the json file into the dataframe
    rows = []
    for query_id_str, entry in data.items():
        query_id = entry.get("query_id", query_id_str)
        for result in entry.get("results", []):
            row = {"query_id": query_id}
            row.update(result)
            rows.append(row)
    df = pd.DataFrame(rows)
    # Find all the entries for the specific file (method) and sort it based on rank
    model = f"Outputs/graphs/{file}/eval_results.json"
    ny_rows = df[df['model'] == model].drop('model', axis=1)
    sorted_df = ny_rows.sort_values(by='recall', ascending=False)
    print(sorted_df)
    return sorted_df



"""-----------------------------------------------------------------------------Leaderboard-----------------------------------------------------------------------------"""


COL_WIDTHS = {
    "method": 65,
    "recall": 12,
    "mrr": 10,
    "ndcg": 10,
    "map": 10,
    "graph": 50,
}

def leaderboard_header():
    return (
        f"{'Method':<{COL_WIDTHS['method']}}"
        f"{'Recall':<{COL_WIDTHS['recall']}}"
        f"{'MRR':<{COL_WIDTHS['mrr']}}"
        f"{'nDCG':<{COL_WIDTHS['ndcg']}}"
        f"{'MAP':<{COL_WIDTHS['map']}}\n"
        f"{'Graph':<{COL_WIDTHS['graph']}}\n"
        + "-" * sum(COL_WIDTHS.values())
    )

def leaderboard_row(row):
    return (
        f"{row['method']:<{COL_WIDTHS['method']}}"
        f"{row['recall']:<{COL_WIDTHS['recall']}.4f}"
        f"{row['mrr']:<{COL_WIDTHS['mrr']}.4f}"
        f"{row['ndcg']:<{COL_WIDTHS['ndcg']}.4f}"
        f"{row['map']:<{COL_WIDTHS['map']}.4f}"
        f"{row.get('graph', ''):<{COL_WIDTHS['graph']}}"
    )
def save_leaderboard(eval_file, output_dir, glob=False):
    grouped = {}
    if glob == False:
        eval_dir = f"{eval_file}/eval_results.json"
    else:
        eval_dir = f"{eval_file}/global_eval_records.json"
    with open(eval_dir) as f:
        for line in f:
            r = json.loads(line)

            sample_type = r["sample_type"]
            final = r["final scores"]

            parts = [r["method"]]

            if "k" in r:
                parts.append(f"k={r['k']}")

            if "alpha" in r and r["alpha"] != "":
                parts.append(f"a={r['alpha']}")

            if "reranker" in r and r["reranker"] != "":
                parts.append(f"reranker={r['reranker']}")

            if "init" in r and r["init"] != "":
                parts.append(f"init={r['init']}")

            method = " | ".join(parts)

            grouped.setdefault(sample_type, []).append({
                "method": method,
                "recall": final["recallk"],
                "mrr": final["mrr"],
                "ndcg": final["ndcg"],
                "map": final["mapk"],
                "graph": r.get("graph", ""),
            })

    out_path = os.path.join(eval_file, output_dir)
    os.makedirs(out_path, exist_ok=True)

    for sample_type, rows in grouped.items():
        rows.sort(key=lambda x: x["recall"], reverse=True)

        safe_name = sample_type.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(out_path, f"{safe_name}.txt")

        with open(file_path, "w") as f:
            f.write(f"=== {sample_type.upper()} ===\n\n")
            f.write(leaderboard_header() + "\n")

            for row in rows:
                f.write(leaderboard_row(row) + "\n")

            f.write("\n")


def make_global_leaderboard():
    out_path = "Outputs/Global leaderboard/global_eval_records.json"
    #Gathering all the eval files from the graphs
    files = get_files()
    with open(out_path, "a") as f_out:  # fresh file each run
        f_out.write("\n")
        for file in files:
            eval_path = f"Outputs/graphs/{file}/eval_results.json"
            if not os.path.exists(eval_path):
                continue
            with open(eval_path) as f_in:
                for line in f_in:
                    line = line
                    if not line:
                        continue
                    r = json.loads(line)
                    r["graph"] = file
                    f_out.write(json.dumps(r) + "\n")
    save_leaderboard("Outputs/Global leaderboard", "leaderboards", True)




"""-----------------------------------------------------------------------------Leaderboard-----------------------------------------------------------------------------"""
