import pandas as pd
import pickle
import networkx as nx
import os
import pyarrow


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

    file_path = "Data/" + name
    with open(file_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    file_path = "Data/" + name
    with open(file_path, "rb") as f:
        G = pickle.load(f)

    return G




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
    file_exists = os.path.isfile(result_file)
    with open(result_file, "a") as f:
        if not file_exists:
            f.write(_header(k) + "\n")
        f.write(_row(method, recallk, mrr_score, ndcg_score, mapk) + "\n")