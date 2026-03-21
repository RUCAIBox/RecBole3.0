import torch
from typing import Dict
from recbole.data.collator import Collator
from torch.utils.data import DataLoader
from recbole.data.dataset import SRDataset


def get_collator(config):
    return Collator

def data_preparation(config, dataset):
    train_dataset, val_dataset, test_dataset = dataset.split()
    collator = get_collator(config)(config)
    num_workers = config["num_workers"]

    train_data = DataLoader(train_dataset, batch_size=config["train_batch_size"],
                            shuffle=True, collate_fn=collator, num_workers=num_workers)
    val_data = DataLoader(val_dataset, batch_size=config["eval_batch_size"],
                          shuffle=False, collate_fn=collator, num_workers=num_workers)
    test_data = DataLoader(test_dataset, batch_size=config["eval_batch_size"],
                           shuffle=False, collate_fn=collator, num_workers=num_workers)

    return train_data, val_data, test_data

def get_dataset(config):
    return SRDataset
