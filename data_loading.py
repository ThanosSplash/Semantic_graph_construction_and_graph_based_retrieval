import json

import pandas as pd
import pickle
import networkx as nx
import os
import pyarrow

import plotting


def read_dataset_bioasq(path):

    # Loading questions and answers
    dataset = pd.read_parquet(path + "question-answer-passages/train-00000-of-00001.parquet", engine='pyarrow')

    # Loading text corpus
    dataset_corpus = pd.read_parquet(path + "text-corpus/train-00000-of-00001.parquet", engine='pyarrow')

    # Checking for nan and na values
    print(dataset_corpus.loc[(dataset_corpus["passage"] == "nan") | dataset_corpus["passage"].isna(), "id"])
    return dataset, dataset_corpus


def save_data(question, answer, corpus):

    data = (question, answer, corpus)
    file_path = "Data/embeddings.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)



def save_samples(corpus, name):
    file_path = "Data/embedding_sample_" + name + ".pkl"
    with open(file_path, "wb") as f:
        pickle.dump(corpus, f)



def save_graph(G, name):

    file_path = "Outputs/" + name + "/graph.gpickle"

    with open(file_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_path = "Outputs/" + name + "/graph_info" + ".json"
    graph_info = plotting.print_graph_stats(G)

    json_str = json.dumps(graph_info, indent=len(graph_info.keys()))
    with open(file_path, "w") as f:
        f.write(json_str)


def load_data():

    file_path = "Data/embeddings.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    question, answer, corpus = data
    return question, answer, corpus


def load_sample(name):
    file_path = "Data/" + name
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def load_graph(name):

    G = nx.Graph()
    file_path = "Outputs/" + name + "/graph.gpickle"
    with open(file_path, "rb") as f:
        G = pickle.load(f)

    return G



def save_graph_data(data, graph_id, fig = None):
    directory_name = "Outputs/" + graph_id

    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    path = directory_name + "/graph_parameters" + ".json"
    if fig is not None:
        fig.savefig(f"{directory_name }/plot.png", dpi=300, bbox_inches="tight")
    json_str = json.dumps(data, indent=len(data.keys()))
    with open(path, "w") as f:
        f.write(json_str)



def get_files():
    path = "Outputs/"
    dir_list = os.listdir(path)
    print("Files and directories in '", len(dir_list), "' :")
    # prints all files
    return dir_list



COL_WIDTHS = {"method": 35, "recall": 12, "mrr": 10, "ndcg": 10, "map": 10}


def _header(k):
    return (
        f"{'Method':<{COL_WIDTHS['method']}}"
        f"{'Recall@'+str(k) :<{COL_WIDTHS['recall']}}"
        f"{'MRR':<{COL_WIDTHS['mrr']}}"
        f"{'nDCG@'+str(k):<{COL_WIDTHS['ndcg']}}"
        f"{'MAP@'+str(k):<{COL_WIDTHS['map']}}\n"
        + "-" * (sum(COL_WIDTHS.values()))
    )

def _row(method, recall, mrr, ndcg, map_score):
    return (
        f"{method:<{COL_WIDTHS['method']}}"
        f"{recall:<{COL_WIDTHS['recall']}.4f}"
        f"{mrr:<{COL_WIDTHS['mrr']}.4f}"
        f"{ndcg:<{COL_WIDTHS['ndcg']}.4f}"
        f"{map_score:<{COL_WIDTHS['map']}.4f}"
    )

def save_result(method: str, recallk, mrr_score, ndcg_score, mapk, k, result_file):
    # write header once, then append rows
    result_file += "/eval_results"
    file_exists = os.path.isfile(result_file)
    with open(result_file, "a") as f:
        if not file_exists:
            f.write(_header(k) + "\n")
        f.write(_row(method, recallk, mrr_score, ndcg_score, mapk) + "\n")