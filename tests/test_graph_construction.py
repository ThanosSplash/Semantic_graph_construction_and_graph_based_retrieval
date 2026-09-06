import pytest
import graph_construction as gc
import data_loading as dt
def test_empty_dataset():
    dataset = {}
    knn_grid = {
        "n_neighbors": 5,
        "metric": "cosine",
        "radius": 1.0, "algorithm": "auto", "leaf_size": 30,
        "p": 2, "metric_params": None, "n_jobs": None,
    }
    graph_params = {"Directed": False, "Weighted": True}
    preprocess = {"scaler": None, "pca": None}
    G = gc.build_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert G.number_of_nodes() == 0
    G_mutual = gc.build_mutual_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert G_mutual.number_of_nodes() == 0
    threshold_grid = {
        "threshold_distance": 0.8
    }
    G_threhsold = gc.build_threshold_graph(dataset, threshold_grid, preprocess, "", graph_params, False)
    assert G_threhsold.number_of_nodes() == 0

def test_graph_construction_params_weighted_directed():
    _, _, dataset = dt.load_data()
    knn_grid = {
        "n_neighbors": 5,
        "metric": "cosine",
        "radius": 1.0, "algorithm": "auto", "leaf_size": 30,
        "p": 2, "metric_params": None, "n_jobs": None,
    }
    graph_params = {"Directed": True, "Weighted": True}
    preprocess = {"scaler": None, "pca": None}
    G = gc.build_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert not (all(data["weight"] == 1.0 for _, _, data in G.edges(data=True))) and G.is_directed()
    G_mutual = gc.build_mutual_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert not (all(data["weight"] == 1.0 for _, _, data in G_mutual.edges(data=True))) and G_mutual.is_directed()
    threshold_grid = {
        "threshold_distance": 0.8
    }
    G_threhsold = gc.build_threshold_graph(dataset, threshold_grid, preprocess, "", graph_params, False)
    assert not (all(data["weight"] == 1.0 for _, _, data in G_threhsold.edges(data=True))) and G_threhsold.is_directed()

def test_graph_construction_params_unweighted_undirected():
    _, _, dataset = dt.load_data()
    knn_grid = {
        "n_neighbors": 5,
        "metric": "cosine",
        "radius": 1.0, "algorithm": "auto", "leaf_size": 30,
        "p": 2, "metric_params": None, "n_jobs": None,
    }
    graph_params = {"Directed": False, "Weighted": False}
    preprocess = {"scaler": None, "pca": None}
    G = gc.build_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert all(data["weight"] == 1.0 for _, _, data in G.edges(data=True)) and (G.is_directed() == False)
    G_mutual = gc.build_mutual_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert all(data["weight"] == 1.0 for _, _, data in G_mutual.edges(data=True)) and (G_mutual.is_directed() is False)
    threshold_grid = {
        "threshold_distance": 0.8
    }
    G_threhsold = gc.build_threshold_graph(dataset, threshold_grid, preprocess, "", graph_params, False)
    assert all(data["weight"] == 1.0 for _, _, data in G_threhsold.edges(data=True)) and (G_threhsold.is_directed() is False)

def test_graph_construction_params_unweighted_directed():
    _, _, dataset = dt.load_data()
    knn_grid = {
        "n_neighbors": 5,
        "metric": "cosine",
        "radius": 1.0, "algorithm": "auto", "leaf_size": 30,
        "p": 2, "metric_params": None, "n_jobs": None,
    }
    graph_params = {"Directed": True, "Weighted": False}
    preprocess = {"scaler": None, "pca": None}
    G = gc.build_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert all(data["weight"] == 1.0 for _, _, data in G.edges(data=True)) and G.is_directed()
    G_mutual = gc.build_mutual_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert all(data["weight"] == 1.0 for _, _, data in G_mutual.edges(data=True)) and G_mutual.is_directed()
    threshold_grid = {
        "threshold_distance": 0.8
    }
    G_threhsold = gc.build_threshold_graph(dataset, threshold_grid, preprocess, "", graph_params, False)
    assert all(data["weight"] == 1.0 for _, _, data in G_threhsold.edges(data=True)) and G_threhsold.is_directed()

def test_graph_construction_params_weighted_undirected():
    _, _, dataset = dt.load_data()
    knn_grid = {
        "n_neighbors": 5,
        "metric": "cosine",
        "radius": 1.0, "algorithm": "auto", "leaf_size": 30,
        "p": 2, "metric_params": None, "n_jobs": None,
    }
    graph_params = {"Directed": False, "Weighted": True}
    preprocess = {"scaler": None, "pca": None}
    G = gc.build_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert not (all(data["weight"] == 1.0 for _, _, data in G.edges(data=True))) and (G.is_directed() is False)
    G_mutual = gc.build_mutual_knn_graph(dataset, knn_grid, preprocess, "", graph_params, False)
    assert not (all(data["weight"] == 1.0 for _, _, data in G_mutual.edges(data=True))) and (G_mutual.is_directed() is False)
    threshold_grid = {
        "threshold_distance": 0.8
    }
    G_threhsold = gc.build_threshold_graph(dataset, threshold_grid, preprocess, "", graph_params, False)
    assert not (all(data["weight"] == 1.0 for _, _, data in G_threhsold.edges(data=True))) and (G_threhsold.is_directed() is False)



def test_nodes_fewer_than_neighbors():
    _, _, dataset = dt.load_data()
    knn_grid = {
        "n_neighbors": 5,
        "metric": "cosine",
        "radius": 1.0, "algorithm": "auto", "leaf_size": 30,
        "p": 2, "metric_params": None, "n_jobs": None,
    }
    preprocess = {"scaler": None, "pca": None}
    docs = len(dataset.keys())
    indices, distances, ids = gc.run_knn(dataset, preprocess, knn_grid)
    assert indices.shape == (docs, 5)



    dataset_sample = dict(list(dataset.items())[:4])
    indices, distances, ids = gc.run_knn(dataset_sample, preprocess, knn_grid)
    assert indices.shape == (4, 4)
