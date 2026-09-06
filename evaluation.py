import numpy as np

def dcg_score(y_true,  gains = "linear"):
    """ Function that calculates dcg and idcg score
        y_true: The prediction in binary form
        k: The k to take into account
    """
    # Calculating gains
    if gains == "exp":
        gains = 2**y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")
    # Calculating discounts
    discounts = np.log2(np.arange(len(y_true)) + 2)

    return float(np.sum(gains/discounts))


def nDCGk_score(predictions, ground_truth, k, gains="linear"):
    """Function that calculates ndcgk score given the predictions and ground truth in this
    form [id1, id2, id3,...]
    predictions: Predictions that the retriever made
    ground_truth: The correct results
    k: The k predictions to take into account
    """
    correct_set = set(ground_truth)
    # Binary scores
    y_true = [1 if id in ground_truth else 0 for id in predictions[:k]]
    if 1 not in y_true:
        return 0
    # Calculating dcg
    dcg = dcg_score(y_true, gains)
    # Ideal binary scores
    denominator = min(len(correct_set), k)
    ideal_y_true = np.ones(denominator)
    # Calculating idcg
    idcg = dcg_score(ideal_y_true, gains)
    return dcg / idcg

def recallk_score(predictions, correct_results, k):
    # Function that calculates the recallk score
    correct_set = set(correct_results)
    if len(correct_set) == 0:
        return 0.0

    correct_guess = 0.0
    for i, prediction in enumerate(predictions):
        if i >= k:
            break
        if prediction in correct_results:
            correct_guess += 1

    return correct_guess/len(correct_results)

def RR_score(predictions, correct_results, k):
    # Function that calculates the rank score
    correct_set = set(correct_results)
    for indx, prediction in enumerate(predictions):
        if prediction in correct_set:
            return 1/(indx + 1), indx

    return 0, k+1


def MRR_score(rr_scores):
    # Function that calculates the mrr score using the rank(rr_score)
    return sum(rr_scores)/len(rr_scores)



def avg_precision(predictions, correct_results, k):
    # Function that calculates the average precision score

    correct_guess = 0
    correct_set = set(correct_results)
    denominator = min(len(correct_set), k)
    sums = 0
    for i, prediction in enumerate(predictions):
        if i >= k:
            break
        if prediction in correct_set:
            correct_guess += 1
            sums += correct_guess/(i+1)

    if len(correct_results) == 0:
        return 0

    return sums/denominator


def paired_bootstrap_ci(baseline, graph, n_boot=5000, seed=42, alpha=0.05):
    """Function that calculates paired bootstrap for baseline and graph retrieval
       the pairs consists of queries
    """
    rng = np.random.default_rng(seed)

    baseline = np.asarray(baseline)
    graph = np.asarray(graph)
    if len(baseline) != len(graph):
        raise ValueError("Unmatched baseline and graph lists")
    # Calculate differences and mean difference
    differences = graph - baseline
    observed_diff = differences.mean()

    means = []
    for _ in range(n_boot):
        # Resampling data in the population and calculating mean
        sample = rng.choice(differences, size=len(differences), replace=True)
        means.append(sample.mean())
    # Calculate lower , upper
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    significant = not (lower <= 0 <= upper)

    return {
        "observed_diff": observed_diff,
        "ci": (lower, upper),
        "significant": significant,
    }