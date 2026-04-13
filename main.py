import experiments as ex
import data_loading as dt
import graph_construction as gc
#ex.baseline_search([1], 3)
#ex.run_all()
#ex.baseline()
#dt.load_sample("embedding_sample_15.pkl")
ex.mutual_knn(dt.load_sample("embedding_sample_100.pkl"))
ex.knn_Graph(dt.load_sample("embedding_sample_100.pkl"))
#ex.cluster_graph(dt.load_sample("embedding_sample_100.pkl"), 2.2)
#ex.threshold_graph(dt.load_sample("embedding_sample_100.pkl"))
#ex.cluster_graph()
# plt.tight_layout()
# plt.show()
#ex.prepare_all()
#ex.prepare_samples()
#ex.cluster_graph(dt.load_sample("embedding_sample_100.pkl"), 2.2)
