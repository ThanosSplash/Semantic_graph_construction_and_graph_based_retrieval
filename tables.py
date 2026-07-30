import os
import json
import data_loading as dt
COL_WIDTHS_QUERIES = {
    "id": 10,
    "method": 20,
    "baseline_rank": 20,
    "baseline_recall": 20,
    "rank": 15,
    "recall": 15,
    "difference": 12,
    "case": 15,
}
COL_WIDTHS_PPR_ALPHA = {
    "alpha": 10,
    "recall": 25,
    "mrr": 25,
    "ndcg": 25,
    "map": 25
}
COL_WIDTHS_PPR_INIT = {
    "init": 10,
    "recall": 25,
    "mrr": 25,
    "ndcg": 25,
    "map": 25
}

COL_WIDTHS = {
    "method": 70,
    "density": 15,
    "components": 15,
    "recall": 15,
    "mrr": 15,
    "ndcg": 15,
    "map": 20,
    "graph": 20,
}
COL_WIDTHS_GRAPH_SENS = {
    "graph": 60,
    "density": 20,
    "components": 20,
    "recall": 15,
    "mrr": 15,
    "ndcg": 15,
    "map": 15,

}


def leaderboard_header_queries_rank_diff():
    return (
        f"{'ID':<{COL_WIDTHS_QUERIES['id']}}"
        f"{'Method':<{COL_WIDTHS_QUERIES['method']}}"
        f"{'Baseline_rank':<{COL_WIDTHS_QUERIES['baseline_rank']}}"
        f"{'Rank':<{COL_WIDTHS_QUERIES['rank']}}"
        f"{'Difference':<{COL_WIDTHS_QUERIES['difference']}}"
        f"{'Case':<{COL_WIDTHS_QUERIES['case']}}\n"
        + "-" * sum(COL_WIDTHS_QUERIES.values())
    )

def leaderboard_row_queries_rank_diff(id_, row):
    return (
        f"{id_:<{COL_WIDTHS_QUERIES['id']}}"
        f"{row['Method']:<{COL_WIDTHS_QUERIES['method']}}"
        f"{row['Baseline_rank']:<{COL_WIDTHS_QUERIES['baseline_rank']}}"
        f"{row['rank']:<{COL_WIDTHS_QUERIES['rank']}}"
        f"{row['Difference']:<{COL_WIDTHS_QUERIES['difference']}}"
        f"{row['Case']:<{COL_WIDTHS_QUERIES['case']}}"
    )


def save_queries_rank_diff_table(data, dir, diff_summary, sort_key="Difference"):
    rows = sorted(data.items(), key=lambda kv: kv[1][sort_key], reverse=True)
    with open(dir, "w") as f:
        f.write(leaderboard_header_queries_rank_diff() + "\n")
        for id_, row in rows:
            f.write(leaderboard_row_queries_rank_diff(id_, row) + "\n")
        f.write(f"\n\nImproved: {diff_summary['Improved']}, Same: {diff_summary['Same']}, Worst: {diff_summary['Worst']}")

def leaderboard_header_queries_recall_diff():
    return (
        f"{'ID':<{COL_WIDTHS_QUERIES['id']}}"
        f"{'Method':<{COL_WIDTHS_QUERIES['method']}}"
        f"{'Baseline_recall':<{COL_WIDTHS_QUERIES['baseline_recall']}}"
        f"{'recall':<{COL_WIDTHS_QUERIES['recall']}}"
        f"{'Difference':<{COL_WIDTHS_QUERIES['difference']}}"
        f"{'Case':<{COL_WIDTHS_QUERIES['case']}}\n"
        + "-" * sum(COL_WIDTHS_QUERIES.values())
    )
def leaderboard_row_queries_recall_diff(id_, row):
    return (
        f"{id_:<{COL_WIDTHS_QUERIES['id']}}"
        f"{row['Method']:<{COL_WIDTHS_QUERIES['method']}}"
        f"{row['Baseline_recall']:<{COL_WIDTHS_QUERIES['baseline_recall']}.4f}"
        f"{row['recall']:<{COL_WIDTHS_QUERIES['recall']}.4f}"
        f"{row['Difference']:<{COL_WIDTHS_QUERIES['difference']}.4f}"
        f"{row['Case']:<{COL_WIDTHS_QUERIES['case']}}"
    )
def save_queries_recall_diff_table(data, dir, diff_summary, sort_key="Difference"):
    rows = sorted(data.items(), key=lambda kv: kv[1][sort_key], reverse=True)
    with open(dir, "w") as f:
        f.write(leaderboard_header_queries_recall_diff() + "\n")
        for id_, row in rows:
            f.write(leaderboard_row_queries_recall_diff(id_, row) + "\n")
        f.write(f"\n\nImproved: {diff_summary['Improved']}, Same: {diff_summary['Same']}, Worst: {diff_summary['Worst']}")
def leaderboard_header_ppr_alpha():
    return (
        f"{'Alpha':<{COL_WIDTHS_PPR_ALPHA['alpha']}}"
        f"{'Recall':<{COL_WIDTHS_PPR_ALPHA['recall']}}"
        f"{'MRR':<{COL_WIDTHS_PPR_ALPHA['mrr']}}"
        f"{'NDCG':<{COL_WIDTHS_PPR_ALPHA['ndcg']}}"
        f"{'MAP':<{COL_WIDTHS_PPR_ALPHA['map']}}\n"
        + "-" * sum(COL_WIDTHS_PPR_ALPHA.values())
    )

def leaderboard_row_ppr_alpha(row):
    return (
        f"{row['params']['alpha']:<{COL_WIDTHS_PPR_ALPHA['alpha']}}"
        f"{row['recall']:<{COL_WIDTHS_PPR_ALPHA['recall']}}"
        f"{row['mrr']:<{COL_WIDTHS_PPR_ALPHA['mrr']}}"
        f"{row['ndcg']:<{COL_WIDTHS_PPR_ALPHA['ndcg']}}"
        f"{row['map']:<{COL_WIDTHS_PPR_ALPHA['map']}}"
    )

def save_ppr_alpha_sensitivity_table(data, dir, name, sort_key="alpha"):
    rows = sorted(data, key=lambda kv: kv['params'][sort_key], reverse=False)
    with open(f"{dir}/{name}", "w") as f:
        f.write(leaderboard_header_ppr_alpha() + "\n")
        for row in rows:
            f.write(leaderboard_row_ppr_alpha(row) + "\n")


def leaderboard_header_ppr_init():
    return (
        f"{'Init':<{COL_WIDTHS_PPR_INIT['init']}}"
        f"{'Recall':<{COL_WIDTHS_PPR_INIT['recall']}}"
        f"{'MRR':<{COL_WIDTHS_PPR_INIT['mrr']}}"
        f"{'NDCG':<{COL_WIDTHS_PPR_INIT['ndcg']}}"
        f"{'MAP':<{COL_WIDTHS_PPR_INIT['map']}}\n"
        + "-" * sum(COL_WIDTHS_PPR_INIT.values())
    )

def leaderboard_row_ppr_init(row):
    return (
        f"{row['params']['init']:<{COL_WIDTHS_PPR_INIT['init']}}"
        f"{row['recall']:<{COL_WIDTHS_PPR_INIT['recall']}}"
        f"{row['mrr']:<{COL_WIDTHS_PPR_INIT['mrr']}}"
        f"{row['ndcg']:<{COL_WIDTHS_PPR_INIT['ndcg']}}"
        f"{row['map']:<{COL_WIDTHS_PPR_INIT['map']}}"
    )
def save_ppr_init_sensitivity_table(data, dir, name, sort_key="init"):
    rows = sorted(data, key=lambda kv: kv['params'][sort_key], reverse=False)
    #print(rows[0]['params'])
    with open(f"{dir}/{name}", "w") as f:
        f.write(leaderboard_header_ppr_init() + "\n")
        for row in rows:
            f.write(leaderboard_row_ppr_init(row) + "\n")


def leaderboard_header_graph_sensitivity():
    return (
        f"{'Graph':<{COL_WIDTHS_GRAPH_SENS['graph']}}"
        f"{'Recall':<{COL_WIDTHS_GRAPH_SENS['recall']}}"
        f"{'MRR':<{COL_WIDTHS_GRAPH_SENS['mrr']}}"
        f"{'NDCG':<{COL_WIDTHS_GRAPH_SENS['ndcg']}}"
        f"{'MAP':<{COL_WIDTHS_GRAPH_SENS['map']}}\n"
        + "-" * sum(COL_WIDTHS_GRAPH_SENS.values())
    )

def leaderboard_row_graph_sensitivity(graph, row):
    return (
        f"{graph:<{COL_WIDTHS_GRAPH_SENS['graph']}}"
        f"{row['recall']:<{COL_WIDTHS_GRAPH_SENS['recall']}.4f}"
        f"{row['mrr']:<{COL_WIDTHS_GRAPH_SENS['mrr']}.4f}"
        f"{row['ndcg']:<{COL_WIDTHS_GRAPH_SENS['ndcg']}.4f}"
        f"{row['map']:<{COL_WIDTHS_GRAPH_SENS['map']}.4f}"
    )
def save_graph_sensitivity_table(data, dir):
    rows = sorted(data.items(), key=lambda kv: kv[0])
    with open(f"{dir}/graph_sensitivy.txt", "w") as f:
        f.write(leaderboard_header_graph_sensitivity() + "\n")
        for id_, row in rows:
            f.write(leaderboard_row_graph_sensitivity(id_, row) + "\n")

def leaderboard_header():
    return (
        f"{'Method':<{COL_WIDTHS['method']}}"
        f"{'Density':<{COL_WIDTHS['density']}}"
        f"{'Components':<{COL_WIDTHS['components']}}"
        f"{'Recall':<{COL_WIDTHS['recall']}}"
        f"{'Recall':<{COL_WIDTHS['recall']}}"
        f"{'MRR':<{COL_WIDTHS['mrr']}}"
        f"{'nDCG':<{COL_WIDTHS['ndcg']}}"
        f"{'MAP':<{COL_WIDTHS['map']}}\n"
        f"{'Graph':<{COL_WIDTHS['graph']}}\n"
        + "-" * sum(COL_WIDTHS.values())
    )

def leaderboard_row(row):
    return (
        f"{row['method']:<{COL_WIDTHS['method']}}"
        f"{row['recall']:<{COL_WIDTHS['recall']}.4f}"
        f"{row['mrr']:<{COL_WIDTHS['mrr']}.4f}"
        f"{row['ndcg']:<{COL_WIDTHS['ndcg']}.4f}"
        f"{row['map']:<{COL_WIDTHS['map']}.4f}"
        f"{row.get('graph', ''):<{COL_WIDTHS['graph']}}"
    )


def save_leaderboard(eval_file, output_dir, glob=False):
    grouped = {}

    if glob == False:
        eval_dir = f"{eval_file}/eval_results.json"
    else:
        eval_dir = f"{eval_file}/global_eval_records.json"


    with open(eval_dir) as f:
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

            if "hops" in r and r["hops"] != "":
                    parts.append(f"hops={r['hops']}")

            method = " | ".join(parts)

            grouped.setdefault(sample_type, []).append({
                "method": method,
                "recall": final["recallk"],
                "mrr": final["mrr"],
                "ndcg": final["ndcg"],
                "map": final["mapk"],
                "graph": r.get("graph", ""),

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

def save_on_global_record_eval(out_path, dir):
    # Gathering all the eval files from the graphs
    files = dt.get_files(dir)
    with open(out_path, "a") as f_out:  # fresh file each run
        for file in files:
            eval_path = f"{dir}/{file}/eval_results.json"
            if not os.path.exists(eval_path):
                continue
            with open(eval_path) as f_in:
                for line in f_in:
                    line = line
                    if not line:
                        continue
                    r = json.loads(line)
                    r["graph"] = file
                    f_out.write(json.dumps(r) + "\n")
def make_global_leaderboard():
    out_path = "Outputs/Global leaderboard/global_eval_records.json"
    dt.clear_eval(out_path)
    #Gathering all the eval files from the graphs
    save_on_global_record_eval(out_path, f"Outputs/graphs")
    save_on_global_record_eval(out_path, f"Outputs/Directed_unweighted_graphs")
    save_on_global_record_eval(out_path, f"Outputs/Directed_weighted_graphs")
    save_on_global_record_eval(out_path, f"Outputs/Undirected_unweighted_graphs")
    save_leaderboard("Outputs/Global leaderboard", "leaderboards", True)

#------------------------------------------------------------------------------------
def make_query_table(file):
    # Load baseline and file queries
    queries = dt.load_queries_results(f"Outputs/graphs/{file}")
    baseline_queries = dt.load_queries_results(f"Outputs/Baseline-RAG")
    # Finding the best performed params and method for mrr
    best_parmas = dt.get_best_performing_method_parms_from_a_graph("all_samples", f"Outputs/graphs/{file}", 'by_mrr')
    queries_rank, diff_summary_rank = dt.extract_best_run_and_compare_to_baseline(queries, baseline_queries, best_parmas, 'rank')
    # Finding the best performed params and method for recall
    best_parmas = dt.get_best_performing_method_parms_from_a_graph("all_samples", f"Outputs/graphs/{file}", 'by_recall')
    queries_recall, diff_summary_recall = dt.extract_best_run_and_compare_to_baseline(queries, baseline_queries, best_parmas, 'recall')

    # For each query find the record that fits the params of the best performance
    save_queries_rank_diff_table(queries_rank, f"Outputs/graphs/{file}/leaderboards/queries_rank_diff.txt", diff_summary_rank)
    save_queries_recall_diff_table(queries_recall, f"Outputs/graphs/{file}/leaderboards/queries_recall_diff.txt", diff_summary_recall)


def make_alpha_sensitivity_table(file):
    # Loading values for each init
    by_init = dt.load_grouped_results(f"Outputs/graphs/{file}", 'by_init')
    # For each init make a seperate table that track changes in recall , mrr as alpha changes
    for init in by_init['PPR']['all_samples'].keys():
        save_ppr_alpha_sensitivity_table(by_init['PPR']['all_samples'][init], f"Outputs/graphs/{file}/leaderboards", f"ppr_alpha_sensitivity_init_{init}.txt")


def make_init_sensitivity_table(file):
    # Loading values for each alpha
    by_alpha = dt.load_grouped_results(f"Outputs/graphs/{file}", 'by_alpha')
    # For each alpha make a seperate table that track changes in recall , mrr as init changes
    for alpha in by_alpha['PPR']['all_samples'].keys():
        save_ppr_init_sensitivity_table(by_alpha['PPR']['all_samples'][alpha], f"Outputs/graphs/{file}/leaderboards", f"ppr_init_sensitivity_alpha_{alpha}.txt")


def make_graph_construction_sensitivity_table():
      best_perf = dt.get_best_performing_graphs_and_params_by_method('PPR', 'all_samples', f"Outputs/graphs")
      save_graph_sensitivity_table(best_perf, f"Outputs/Global leaderboard")


