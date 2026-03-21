import torch
from collections import OrderedDict
from recbole.evaluator.metrics import metrics_to_function


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.topk = self.config["topk"]
        self.max_topk = max(self.topk)
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

    def evaluate(self, scores, labels):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        _, topk_idx = torch.topk(
            scores, self.max_topk, dim=-1
        )  # B x k
        topk_idx = topk_idx.detach().cpu()
        labels = labels.detach().cpu()
        
        one_hot_labels = torch.zeros_like(scores).detach().cpu()
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        top_k_labels = torch.gather(one_hot_labels, dim=1, index=topk_idx).numpy()
        pos_nums = one_hot_labels.sum(dim=1).numpy()

        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = metrics_to_function[metric](top_k_labels, pos_nums)
            for k in self.topk:
                result_dict[f"{metric}@{k}"] = metric_val[:, k - 1].sum().item()
        
        return result_dict
    
