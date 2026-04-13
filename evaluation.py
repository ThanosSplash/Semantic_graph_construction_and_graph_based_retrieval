from sklearn.metrics import ndcg_score
import numpy as np
def recallk_score(predictions, correct_results, k):
    if len(correct_results) == 0:
        return 0.0
    correct_guess = 0
    for prediction in predictions:
        if prediction in correct_results:
            correct_guess += 1

    return correct_guess/k

def RR_score(predictions, correct_results):
    for indx, answer in enumerate(correct_results):
        if answer in predictions:
            return 1/(indx + 1)

    return 0


def MRR_score(rr_scores):
    return sum(rr_scores)/len(rr_scores)


def DCGk_score(predictions, correct_results, k):
    gt_scores = {id: len(correct_results) - i for i, id in enumerate(correct_results)}
    # → {"id1": 5, "id6": 4, "id2": 3, "id10": 2, "id3": 1}

    true_relevance = np.array([[gt_scores.get(id, 0) for id in predictions]])
    # → [[1, 5, 0, 3]]

    predicted_scores = np.array([[len(predictions) - i for i in range(len(predictions))]])
    # → [[4, 3, 2, 1]]
    return ndcg_score(true_relevance, predicted_scores, k=k)

