import numpy as np
from time import time


def hit(top_k_results, pos_nums):
    result = np.cumsum(top_k_results, axis=1)
    return (result > 0).astype(int)

def recall(top_k_results, pos_nums):

    recall = np.cumsum(top_k_results, axis=1) / pos_nums.reshape(-1, 1)
    return recall

def ndcg(top_k_results, pos_nums):
    len_rank = np.full_like(pos_nums, top_k_results.shape[1])
    idcg_len = np.where(pos_nums > len_rank, len_rank, pos_nums).astype(int)

    iranks = np.zeros_like(top_k_results, dtype=np.float32)
    iranks[:, :] = np.arange(1, top_k_results.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(top_k_results, dtype=np.float32)
    ranks[:, :] = np.arange(1, top_k_results.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(top_k_results, dcg, 0.0), axis=1)

    result = dcg / idcg
    return result

def mrr(top_k_results, pos_nums):
    idxs = top_k_results.argmax(axis=1)
    result = np.zeros_like(top_k_results, dtype=np.float32)
    for row, idx in enumerate(idxs):
        if top_k_results[row, idx] > 0:
            result[row, idx:] = 1 / (idx + 1)
        else:
            result[row, idx:] = 0
    return result


metrics_to_function = {
    "hit": hit,
    "recall": recall,
    "ndcg": ndcg,
    "mrr": mrr,
}