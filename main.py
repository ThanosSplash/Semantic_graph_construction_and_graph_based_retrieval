import experiments as ex
import data_loading as dt
import retrieval as rt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotting as pt
import graph_construction as gc
import evaluation as ev
kmeans_params = {
    "n_clusters": None,
    "init": "k-means++",
    "n_init": "auto",
    "max_iter": 300,
    "tol": 1e-4,
    "verbose": 0,
    "random_state": None,
    "copy_x": True,
    "algorithm": "lloyd",
    "pipeline_id": 0
}
agglo_params = {
    "n_clusters": None,
    "distance_threshold": None,
    "linkage": "ward",
    "metric": "euclidean",
    "compute_full_tree": "auto",
    "connectivity": None
}
dbscan_params = {
      "eps": 0.5,
      "min_samples": 5,
      "metric": "euclidean",
      "metric_params": None,
      "algorithm": "auto",
      "Leaf_size": 30,
      "p": None,
      "n_jobs": None
}
knn_params = {
    "n_neighbors": 5,
    "radius": 1.0,
    "algorithm": "auto",
    "leaf_size": 30,
    "metric": "cosine",
    "p": 2,
    "metric_params": None,
    "n_jobs": None
}
threshold_params = {
    "threshold_distance": 0.3
}
preprocess = {
    "scaler": None,
    "pca": None
}
result_file = "Outputs/evaluation_results"

#pt.print_graph_stats(dt.load_graph("Graph_Μutual_KNN.gpickle"))



# Preparing dataset Part
#ex.prepare_all()
#ex.prepare_samples()



# Retrieve Data Part
q, a, c = dt.load_sample("embeddings.pkl")
q_ids = list(q.keys())[:1]
ex.baseline_search(q_ids, 10, result_file + str(10)+".txt")
ex.personalised_pagerank_search(q_ids, dt.load_graph("Graph_KNN.gpickle"),"KNN", 10, result_file + str(10)+".txt")
ex.personalised_pagerank_search(q_ids, dt.load_graph("Graph_Μutual_KNN.gpickle"), "MKNN", 10, result_file + str(10)+".txt")
ex.personalised_pagerank_search(q_ids, dt.load_graph("Graph_KNN_Kmeans.gpickle"), "Clustering + KNN" ,10, result_file + str(10)+".txt")
ex.personalised_pagerank_search(q_ids, dt.load_graph("Graph_Mutual_KNN_Kmeans.gpickle"), "Clustering + MKNN", 10, result_file + str(10)+".txt")



# Graph Construction Part

#q, a, c = dt.load_sample("embeddings.pkl")
#ex.knn_Graph(c, knn_params, preprocess, "Graph_KNN.gpickle", True)
#ex.mutual_knn(c, knn_params, preprocess, "Graph_Μutual_KNN.gpickle", True)
#ex.cluster_graph(c, kmeans_params,threshold_params, agglo_params, preprocess, knn_params, dbscan_params,
                # "kmeans")

# Print Graph info Part
# pt.print_graph_stats(dt.load_graph("Graph_Μutual_KNN.gpickle"))
# pt.print_graph_stats(dt.load_graph("Graph_KNN.gpickle"))
# pt.print_graph_stats(dt.load_graph("Graph_Mutual_KNN_Kmeans.gpickle"))
# pt.print_graph_stats(dt.load_graph("Graph_Threshold.gpickle"))



