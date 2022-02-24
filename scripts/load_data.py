import os
import numpy as np 
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd 
import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import default_collate
from transformers import data


def get_senteval_data_preprocessed(fpath, args):
    if not os.path.exists(fpath):
        raise FileNotFoundError(f'{fpath} not exists.')

    data = torch.load(fpath)
    data['X'] = torch.tensor(data['X'], dtype=torch.float)
    data['y'] = torch.tensor(data['y'], dtype=torch.long)
    nclasses = torch.unique(data['y']).shape[0]


    if args.even_distribute:
        assert args.train_size_per_class * nclasses + args.val_size_per_class * nclasses < len(data['y']), "train and val sizes should add up to be no more than the total num in the data!"

        train_x, other_x, train_y, other_y = train_test_split(
            data['X'], data['y'],
            random_state=args.seed, 
            train_size=args.train_size_per_class * nclasses, 
            shuffle=True,
            stratify=data['y']
        )
        val_x, remain_x, val_y, remain_y = train_test_split(
            other_x, other_y,
            random_state=args.seed,
            train_size=args.val_size_per_class * nclasses,
            shuffle=True,
            stratify=other_y
        )
        test_x, _, test_y, _ = train_test_split(
            remain_x, remain_y,
            random_state=args.seed,
            train_size=args.val_size_per_class * nclasses,
            shuffle=True,
            stratify=remain_y
        )
    else:
        raise ValueError("Only supports even distribution.")


    # Optional: Inject Gaussian noise
    if args.representation_gaussian_noise > 0:
        train_x += torch.normal(0, args.representation_gaussian_noise, size=train_x.size())
        val_x += torch.normal(0, args.representation_gaussian_noise, size=val_x.size())
        test_x += torch.normal(0, args.representation_gaussian_noise, size=test_x.size())
    return TensorDataset(train_x, train_y), TensorDataset(val_x, val_y), TensorDataset(test_x, test_y), nclasses 


def get_collate_fn(batcher, use_cuda, task):
    """
    Required when constructing torch.utils.DataLoader.
    """
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    def _senteval_collate_fn(databatch):
        # xtensor = torch.tensor(batcher([item['X'] for item in databatch])).float()
        # ytensor = torch.tensor([item['y'] for item in databatch]).long()
        return tuple(d.to(device) for d in default_collate(databatch))

    return _senteval_collate_fn

def senteval_load_file(filepath="../../data/senteval/subj_number.txt"):
    """
    Input:
        filepath. e.g., "<repo_dir>/data/senteval/bigram_shift.txt"
    Return: 
        task_data: list of {'X': str, 'y': int}
        nclasses: int
    """

    # Just load all portions, and then do train/dev/test splitting myself
    tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
    task_data=[]
    
    for linestr in Path(filepath).open().readlines():
        line = linestr.rstrip().split("\t")
        task_data.append({
            'X': line[-1], 'y': line[1]
        })

    # Convert labels str to int
    all_labels = [item['y'] for item in task_data]
    labels = sorted(np.unique(all_labels))
    tok2label = dict(zip(labels, range(len(labels))))
    nclasses = len(tok2label) 
    for i, item in enumerate(task_data):
        item['y'] = tok2label[item['y']]
    
    return task_data, nclasses


def main():
    # For testing.
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()