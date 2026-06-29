from sklearn.metrics import ndcg_score
import numpy as np
import math as m
def dcg_score(y_true, k, gains = "linear"):
    """ Function that calculates dcg and idcg score
        y_true: The prediction in binary form
        k: The k to take into account
    """
    rels = y_true[:k]
    # Calculating gains
    if gains == "exp":
        gains = 2**rels - 1
    elif gains == "linear":
        gains = rels
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
    # Binary scores
    y_true = [1 if id in ground_truth else 0 for id in predictions]
    if 1 not in y_true:
        return 0
    # Calculating dcg
    dcg = dcg_score(y_true, k, gains)
    # Ideal binary scores
    y_true.sort(reverse=True)
    # Calculating idcg
    idcg = dcg_score(y_true, k, gains)
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

def RR_score(predictions, correct_results):
    # Function that calculates the rank score
    correct_set = set(correct_results)
    for indx, prediction in enumerate(predictions):
        if prediction in correct_set:
            return 1/(indx + 1), indx

    return 0, -1


def MRR_score(rr_scores):
    # Function that calculates the mrr score using the rank(rr_score)
    return sum(rr_scores)/len(rr_scores)



def avg_precision(predictions, correct_results, k):
    # Function that calculates the average precision score

    correct_guess = 0
    correct_set = set(correct_results)
    sums = 0
    for i, prediction in enumerate(predictions):
        if i >= k:
            break
        if prediction in correct_set:
            correct_guess += 1
            sums += correct_guess/(i+1)

    if len(correct_results) == 0:
        return 0

    return sums/len(correct_results)


#def nDCGk_score(predictions, correct_results, k):
    #if len(predictions) == 0:
        #return 0.0
    #correct_set = set(correct_results)


    #relevance = np.array([[1 if p in correct_set else 0 for p in predictions]])


    #mock_scores = np.array([[len(predictions) - i for i in range(len(predictions))]])

    #return ndcg_score(relevance, mock_scores, k=k)

