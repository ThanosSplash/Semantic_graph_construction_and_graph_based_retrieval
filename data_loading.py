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

    with open(directory_name, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
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
    results_file += "/eval_results.json"
    with open(f"Outputs/{results_file}", "a") as f:
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


    return

def load_graph(name):

    G = nx.Graph()
    directory_name = f"Outputs/graphs/{name}/graph.gpickle"
    with open(directory_name, "rb") as f:
        G = pickle.load(f)

    return G

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

"""-----------------------------------------------------------------------------Leaderboard-----------------------------------------------------------------------------"""


COL_WIDTHS = {
    "method": 65,
    "recall": 12,
    "mrr": 10,
    "ndcg": 10,
    "map": 10,
}

def leaderboard_header():
    return (
        f"{'Method':<{COL_WIDTHS['method']}}"
        f"{'Recall':<{COL_WIDTHS['recall']}}"
        f"{'MRR':<{COL_WIDTHS['mrr']}}"
        f"{'nDCG':<{COL_WIDTHS['ndcg']}}"
        f"{'MAP':<{COL_WIDTHS['map']}}\n"
        + "-" * sum(COL_WIDTHS.values())
    )

def leaderboard_row(row):
    return (
        f"{row['method']:<{COL_WIDTHS['method']}}"
        f"{row['recall']:<{COL_WIDTHS['recall']}.4f}"
        f"{row['mrr']:<{COL_WIDTHS['mrr']}.4f}"
        f"{row['ndcg']:<{COL_WIDTHS['ndcg']}.4f}"
        f"{row['map']:<{COL_WIDTHS['map']}.4f}"
    )
def save_leaderboard(eval_file, output_dir):
    grouped = {}

    with open(eval_file + "/eval_results.json") as f:
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
"""-----------------------------------------------------------------------------Leaderboard-----------------------------------------------------------------------------"""
