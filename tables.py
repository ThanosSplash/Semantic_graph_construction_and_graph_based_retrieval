import os
import json
import data_loading as dt
COL_WIDTHS = {
    "method": 25,
    "density": 25,
    "components": 15,
    "recall": 25,
    "mrr": 25,
    "ndcg": 25,
    "map": 25,
    "graph": 120,
    "graph_name": 90,
    "hops": 25,
    "init": 25,
    "id": 25,
    "baseline_rank": 25,
    "baseline_recall": 25,
    "rank": 25,
    "difference": 25,
    "case": 25,
    "alpha": 20,
    "k": 15,
    "reranker": 25,
    "weighted_score": 20,
    "sample_type":20

}


def leaderboard_header_(headers):
    header_row = ""
    for header in headers:
        header_row += f"{header:<{COL_WIDTHS[header]}}"

    return header_row
def leaderboard_row_(row_content, headers):
    row_ = ""
    for header in headers:
        if header in row_content:
            row_ += f"{row_content[header]:<{COL_WIDTHS[header]}}"
        else:
            blank = "-"
            row_ += f"{blank:<{COL_WIDTHS[header]}}"


    return row_


def save_queries_rank_diff_table(data, dir, name, sort_key="difference"):
    headers = ['method', 'baseline_rank', 'rank', 'difference', 'case']
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph in data.keys():
        rows = sorted(data[graph][0].items(), key=lambda kv: kv[1]["difference"], reverse=True)
        with open(f"{dir}/{name}", "a") as f:
            f.write(f"----------{graph}----------------------\n\n")
            f.write(f"{'id':<{COL_WIDTHS['id']}}")
            f.write(leaderboard_header_(headers) + "\n")
            for id_, row in rows:
                f.write(f"{id_:<{COL_WIDTHS['id']}}")
                f.write(leaderboard_row_(row, headers) + "\n")
            f.write(
                f"\n\nImproved: {data[graph][1]['Improved']}, Same: {data[graph][1]['Same']}, Worst: {data[graph][1]['Worst']}")
            f.write(f"\n\n--------------------------------\n\n")

def save_queries_recall_diff_table(data, dir, name, sort_key="difference"):
    headers = ['method', 'baseline_recall', 'recall', 'difference', 'case']
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    for graph in data.keys():
        rows = sorted(data[graph][0].items(), key=lambda kv: kv[1]["difference"], reverse=True)
        with open(f"{dir}/{name}", "a") as f:
            f.write(f"----------{graph}----------------------\n\n")
            f.write(f"{'id':<{COL_WIDTHS['id']}}")
            f.write(leaderboard_header_(headers) + "\n")
            for id_, row in rows:
                f.write(f"{id_:<{COL_WIDTHS['id']}}")
                f.write(leaderboard_row_(row, headers) + "\n")
            f.write(
                f"\n\nImproved: {data[graph][1]['Improved']}, Same: {data[graph][1]['Same']}, Worst: {data[graph][1]['Worst']}")
            f.write(f"\n\n--------------------------------\n\n")
def save_ppr_alpha_sensitivity_table(data, dir, name, sort_key="init"):
    # Clear txt file
    headers = ['graph_name', 'sample_type', 'init', 'k', 'alpha', 'recall', 'mrr', 'ndcg', 'map',"weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")

    rows = sorted(data, key=lambda kv: kv[sort_key], reverse=False)
    with open(f"{dir}/{name}", "a") as f:
            f.write(leaderboard_header_(headers) + "\n")
            for row in rows:
                f.write(leaderboard_row_(row, headers) + "\n")

def save_ppr_init_sensitivity_table(data, dir, name, sort_key="alpha"):
    headers = ['graph_name', 'sample_type', 'alpha', 'init', 'k', 'recall', 'mrr', 'ndcg', 'map', "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")

    rows = sorted(data, key=lambda kv: kv[sort_key], reverse=False)
    with open(f"{dir}/{name}", "a") as f:
        f.write(leaderboard_header_(headers) + "\n")
        for row in rows:
            f.write(leaderboard_row_(row, headers) + "\n")


def save_graph_sensitivity_table(data, dir, name):
    headers = ['graph_name','sample_type',"density", "components", "recall", "mrr", "ndcg", "map", "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    rows = sorted(data, key=lambda kv: kv['weighted_score'], reverse=True)
    with open(f"{dir}/{name}", "a") as f:
        f.write(leaderboard_header_(headers) + "\n")
        for row in rows:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"---------------------------------  \n\n")

def save_hop_sensitivity_table(data, dir, name):
  headers = ['graph_name', 'sample_type', 'hops', "init", "alpha", "reranker", "k", "recall", "mrr", "ndcg", "map"]
  with open(f"{dir}/{name}", "w") as f:
        f.write("")
  rows = sorted(data, key=lambda kv: kv['hops'], reverse=False)
  with open(f"{dir}/{name}", "a") as f:
      f.write(leaderboard_header_(headers) + "\n")
      for row in rows:
          f.write(leaderboard_row_(row, headers) + "\n")
      f.write(f"---------------------------------  \n\n")




def save_leaderboard(data, dir, name):
    headers = ['graph_name',"sample_type","alpha", "init", "hops", "k", "recall", "mrr", "ndcg", "map", "weighted_score"]
    with open(f"{dir}/{name}", "w") as f:
        f.write("")
    rows = sorted(data, key=lambda kv: kv['weighted_score'], reverse=True)
    with open(f"{dir}/{name}", "a") as f:
        f.write(leaderboard_header_(headers) + "\n")
        for row in rows:
            f.write(leaderboard_row_(row, headers) + "\n")
        f.write(f"---------------------------------  \n\n")




#------------------------------------------------------------------------------------
def make_query_table(file):
    # Load baseline and file queries
    queries = dt.load_queries_results(f"Outputs/runs/{file}")
    baseline_queries = dt.load_queries_results(f"Outputs/Baseline-RAG")
    # Finding the best performed params and method for mrr
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = []
    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            for sample_type, results in sample_types.items():
                params = results[0].pop("params", {})
                params['method'] = method
                params['graph'] = graph
                all_data.append((results[0]['graph_name'], params))
    all_queries_rank = {}
    for graph, params in all_data:
        queries_rank, diff_summary_rank = dt.extract_best_run_and_compare_to_baseline(queries, baseline_queries, params, 'rank')
        all_queries_rank[graph] = (queries_rank, diff_summary_rank)
    all_queries_recall = {}
    for graph, params in all_data:
        queries_recall, diff_summary_recall = dt.extract_best_run_and_compare_to_baseline_recall(queries, baseline_queries, params, 'recall')
        all_queries_recall[graph] = (queries_recall, diff_summary_recall)
    save_queries_rank_diff_table(all_queries_rank, f"Outputs/runs/{file}/tables", "queries_rank_diff.txt")
    save_queries_recall_diff_table(all_queries_recall, f"Outputs/runs/{file}/tables", "queries_recall_diff.txt")

def make_alpha_sensitivity_table(file):
    # Loading values for each init
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = []

    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            for sample_type, results in sample_types.items():
                for result in results:
                    params = result.pop("params", {})

                    flat_entry = {
                        "method": method,
                        "sample_type": sample_type,
                        **result,
                        **params
                    }

                    all_data.append(flat_entry)

    # For each init make a seperate table that track changes in recall , mrr as alpha changes
    save_ppr_alpha_sensitivity_table(all_data, f"Outputs/runs/{file}/tables", f"ppr_alpha_sensitivity_init.txt")


def make_init_sensitivity_table(file):
    # Loading values for each alpha
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = []

    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            for sample_type, results in sample_types.items():
                for result in results:
                    params = result.pop("params", {})

                    flat_entry = {
                        "method": method,
                        "sample_type": sample_type,
                        **result,
                        **params
                    }

                    all_data.append(flat_entry)
    # For each alpha make a seperate table that track changes in recall , mrr as init changes
    save_ppr_init_sensitivity_table(all_data, f"Outputs/runs/{file}/tables", f"ppr_init_sensitivity_alpha.txt")


def make_graph_construction_sensitivity_table(file):
      grouped = dt.load_grouped(f"Outputs/runs/{file}")
      all_data = []
      for graph, methods in grouped.items():
          for method, sample_types in methods.items():
              for sample_type, results in sample_types.items():
                      params = results[0].pop("params", {})

                      flat_entry = {
                          "method": method,
                          "sample_type": sample_type,
                          **results[0],
                          **params
                      }

                      all_data.append(flat_entry)

      save_graph_sensitivity_table(all_data, f"Outputs/runs/{file}/tables", "graph_sensitivity.txt")


def make_hop_sensitivity_table(file):
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = []
    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            if method == "k-steph":
                for sample_type, results in sample_types.items():
                    for result in results:
                        params = result.pop("params", {})

                        flat_entry = {
                            "method": method,
                            "sample_type": sample_type,
                            **results[0],
                            **params
                        }

                        all_data.append(flat_entry)

    save_hop_sensitivity_table(all_data, f"Outputs/runs/{file}/tables", f"hop_sensitivity.txt")


def make_leaderboard_table(file):
    # Loading values for each alpha
    grouped = dt.load_grouped(f"Outputs/runs/{file}")
    all_data = []
    for graph, methods in grouped.items():
        for method, sample_types in methods.items():
            for sample_type, results in sample_types.items():
                    for result in results:
                        params = result.pop("params", {})

                        flat_entry = {
                            "method": method,
                            "sample_type": sample_type,
                            **result,
                            **params
                        }

                        all_data.append(flat_entry)
    save_leaderboard(all_data, f"Outputs/runs/{file}/tables", "leaderboard.txt")
