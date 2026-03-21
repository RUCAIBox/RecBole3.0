from torch.nn.utils.rnn import pad_sequence
import torch
from typing import Dict


class Collator:
    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        item_seq, item, tsp_seq, tsp = zip(*batch)
    
        item_seq = [torch.LongTensor(seq) for seq in item_seq]
        tsp_seq = [torch.LongTensor(seq) for seq in tsp_seq]

        item_seq= pad_sequence(item_seq, batch_first=True, padding_value=0)
        tsp_seq = pad_sequence(tsp_seq, batch_first=True, padding_value=0)

        item = torch.LongTensor(item).view(-1)
        tsp = torch.LongTensor(tsp).view(-1)

        batch =  dict(input_ids=item_seq, labels=item,
                      tsp_seq=tsp_seq, tsp=tsp)
        
        if self.config["model"] == "HSTU":
            batch = hstu_seq_features_process(batch)
        
        return batch

def hstu_seq_features_process(
    batch: Dict[str, torch.Tensor]):
    item_seq = batch["input_ids"]
    tsp_seq = batch["tsp_seq"]
    item = batch["labels"]
    tsp = batch["tsp"]

    tsp_seq = torch.cat([tsp_seq, tsp_seq.new_zeros((tsp_seq.size(0), 1))], dim=1)
    tsp_seq.scatter_(
        dim=1,
        index=(item_seq != 0).sum(dim=1, keepdim=True),
        src=tsp.view(-1, 1),
    )
    return dict(input_ids=item_seq, tsp_seq=tsp_seq, labels=item, tsp=tsp)