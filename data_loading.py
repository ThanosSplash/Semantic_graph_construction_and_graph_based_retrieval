import pandas as pd
import pickle
import networkx as nx
import pyarrow


def read_dataset_bioasq(path):
    """
    Loading the data from the dataset bioasq and checking for nan or na values
    :param path: path of the dataset biosaq
    :return: Two dataframes one for the questions and answer and one for the passages
    """
    # Loading questions and answers
    dataset = pd.read_parquet(path + "question-answer-passages/train-00000-of-00001.parquet", engine='pyarrow')

    # Loading text corpus
    dataset_corpus = pd.read_parquet(path + "text-corpus/train-00000-of-00001.parquet", engine='pyarrow')

    # Checking for nan and na values
    print(dataset_corpus.loc[(dataset_corpus["passage"] == "nan") | dataset_corpus["passage"].isna(), "id"])
    return dataset, dataset_corpus


def save_data(question, answer, corpus):
    """
    :param question:
    :param answer:
    :param corpus:
    :return:
    """
    data = (question, answer, corpus)
    file_path = "Data/embeddings.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)



def save_samples(corpus, name):
    file_path = "Data/embedding_sample_" + name + ".pkl"
    with open(file_path, "wb") as f:
        pickle.dump(corpus, f)



def save_graph(G, name):
    """
    :param G:
    :param name:
    :return:
    """
    file_path = "Data/" + name
    with open(file_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data():
    """
    :return:
    """
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
    """
    :param name:
    :return:
    """
    G = nx.Graph()
    file_path = "Data/Graph_Baseline.gpickle"
    with open(file_path, "rb") as f:
        G = pickle.load(f)

    return G