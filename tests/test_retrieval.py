import retrieval as rt
import data_loading as dt
from scipy.sparse import csr_matrix
def test_normalisation():
    small, _, _ = dt.load_splits_dev()
    query_id = small[0]
    query_type = "dev"
    init = 5
    alpha = 0.5
    k = 10
    norm = False
    file_name = "knn_scNone_pcaNone_Directed_False_Weighted_True_neighbors_10_metriccosine"
    graph = dt.load_graph(file_name)
    graph_cache = dt.load_graph_cache(file_name)
    adjacency = csr_matrix(graph_cache['adjacency'])
    imporant_nodes, correct, sims = rt.top_k(query_id, query_type, init)
    # Running personalised pagerank
    predictions = rt.personalised_pagerank(graph, adjacency, graph_cache['node_to_idx'],
                                           graph_cache['global_node_list'],
                                           imporant_nodes, sims, k, query_id, alpha, query_type, norm)
    #print(predictions)
    norm = True
    predictions = rt.personalised_pagerank(graph, adjacency, graph_cache['node_to_idx'],
                                           graph_cache['global_node_list'],
                                           imporant_nodes, sims, k, query_id, alpha, query_type, norm)
    #print(predictions)
