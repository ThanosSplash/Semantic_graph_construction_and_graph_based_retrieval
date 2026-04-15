import experiments as ex
import data_loading as dt
import graph_construction as gc

# Baseling Search
ex.baseline_search([1,2,100], 3)


# mutual KNN graph
#ex.mutual_knn(dt.load_sample("embedding_sample_100.pkl"))
#ex.mutual_knn(dt.load_sample("embedding_sample_15.pkl"))
#ex.mutual_knn(dt.load_sample("embedding_sample_1000.pkl"))

# KNN graph
#ex.knn_Graph(dt.load_sample("embedding_sample_15.pkl"))
#ex.knn_Graph(dt.load_sample("embedding_sample_100.pkl"))
#ex.knn_Graph(dt.load_sample("embedding_sample_1000.pkl"))



# CLuster graph
#ex.cluster_graph(dt.load_sample("embedding_sample_15.pkl"), 34)
#ex.cluster_graph(dt.load_sample("embedding_sample_100.pkl"), 47)
#ex.cluster_graph(dt.load_sample("embedding_sample_1000.pkl"), 105)

# Threshold graph
#ex.threshold_graph(dt.load_sample("embedding_sample_15.pkl"))
#ex.threshold_graph(dt.load_sample("embedding_sample_100.pkl"))
#ex.threshold_graph(dt.load_sample("embedding_sample_1000.pkl"))

# Preparing dataset
#ex.prepare_all()
#ex.prepare_samples()

