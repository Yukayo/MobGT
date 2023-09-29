# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collator import collator, collator_foursquare, collator_toyota, collator_gowalla
from wrapper import (
    MyGraphPropPredDataset,
    MyPygPCQM4MDataset,
    MyFoursquareDataset,
    MyFoursquareGraphDataset,
    MyToyotaGraphDataset,
    MyGowallaGraphDataset
)

import numpy as np

from pytorch_lightning import LightningDataModule
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
import ogb
import ogb.lsc
import ogb.graphproppred
from ogb.nodeproppred import Evaluator as NodePropPredEvaluator
from functools import partial


dataset = None

class GraphDataModule_train(Dataset):

    def __init__(self,path):
        self.dataset=MyFoursquareDataset(
                subset=False, root=path, split="train"
            )
    def __len__(self):
        return(len(self.dataset))
        # assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        # return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.dataset[index])

def get_dataset(dataset_name="abaaba"):
    global dataset
    if dataset is not None:
        return dataset

    # max_node is set to max(max(num_val_graph_nodes), max(num_test_graph_nodes))
    if dataset_name == "foursquare":
        dataset = {
            # "num_class": 1,
            #loss和deepmove暂且一致
            "loss_fn": nn.NLLLoss(ignore_index=0).cuda(),
            # "loss_fn": nn.CrossEntropyLoss(ignore_index=0).cuda(),
            # "metric": "cross_entropy",
            "metric": "NLLLoss",
            "metric_mode": "min",
            "evaluator": NodePropPredEvaluator(name="ogbn-arxiv"),
            "train_dataset": MyFoursquareDataset(
                subset=False, root="../dataset/foursquare", split="train"
            ),
            "val_dataset": MyFoursquareDataset(
                subset=False, root="../dataset/foursquare", split="test"
            ),
            "test_dataset": MyFoursquareDataset(
                subset=False, root="../dataset/foursquare", split="test"
            ),

            # "max_node": 2708,
        }
    elif dataset_name == "foursquaregraph":
        dataset = {
            # "num_class": 1,
            #loss和deepmove暂且一致
            "loss_fn": nn.NLLLoss(ignore_index=0).cuda(),
            # "loss_fn": nn.CrossEntropyLoss(ignore_index=0).cuda(),
            # "metric": "cross_entropy",
            "metric": "NLLLoss",
            "metric_mode": "min",
            "evaluator": NodePropPredEvaluator(name="ogbn-arxiv"),
            "train_dataset": MyFoursquareGraphDataset(
                subset=False, root="../dataset/foursquaregraph", split="train"
            ),
            "val_dataset": MyFoursquareGraphDataset(
                subset=False, root="../dataset/foursquaregraph", split="test"
            ),
            "test_dataset": MyFoursquareGraphDataset(
                subset=False, root="../dataset/foursquaregraph", split="test"
            ),

            # "max_node": 2708,
        }
    elif dataset_name == "gowalla_nevda":
        dataset = {
            # "num_class": 1,
            #loss和deepmove暂且一致
            "loss_fn": nn.NLLLoss(ignore_index=0).cuda(),
            # "loss_fn": nn.CrossEntropyLoss(ignore_index=0).cuda(),
            # "metric": "cross_entropy",
            "metric": "NLLLoss",
            "metric_mode": "min",
            "evaluator": NodePropPredEvaluator(name="ogbn-arxiv"),
            "train_dataset": MyGowallaGraphDataset(
                subset=False, root="../dataset/gowalla_nevda", split="train"
            ),
            "val_dataset": MyGowallaGraphDataset(
                subset=False, root="../dataset/gowalla_nevda", split="test"
            ),
            "test_dataset": MyGowallaGraphDataset(
                subset=False, root="../dataset/gowalla_nevda", split="test"
            ),

            # "max_node": 2708,
        }
    elif dataset_name == "gowalla_7day":
        dataset = {
            # "num_class": 1,
            #loss和deepmove暂且一致
            "loss_fn": nn.NLLLoss(ignore_index=0).cuda(),
            # "loss_fn": nn.CrossEntropyLoss(ignore_index=0).cuda(),
            # "metric": "cross_entropy",
            "metric": "NLLLoss",
            "metric_mode": "min",
            "evaluator": NodePropPredEvaluator(name="ogbn-arxiv"),
            "train_dataset": MyGowallaGraphDataset(
                subset=False, root="../dataset/gowalla_7day", split="train"
            ),
            "val_dataset": MyGowallaGraphDataset(
                subset=False, root="../dataset/gowalla_7day", split="test"
            ),
            "test_dataset": MyGowallaGraphDataset(
                subset=False, root="../dataset/gowalla_7day", split="test"
            ),

            # "max_node": 2708,
        }
    elif dataset_name == "toyota":
        dataset = {
            # "num_class": 1,
            #loss和deepmove暂且一致
            "loss_fn": nn.NLLLoss(ignore_index=0).cuda(),
            # "metric": "cross_entropy",
            "metric": "NLLLoss",
            "metric_mode": "min",
            "evaluator": NodePropPredEvaluator(name="ogbn-arxiv"),
            "train_dataset": MyFoursquareDataset(
                subset=False, root="../dataset/toyota", split="train"
            ),
            "val_dataset": MyFoursquareDataset(
                subset=False, root="../dataset/toyota", split="test"
            ),
            "test_dataset": MyFoursquareDataset(
                subset=False, root="../dataset/toyota", split="test"
            ),

            # "max_node": 2708,
        }
    
    # ADDED toyotagraph from graphormer_new_cuda2
    elif dataset_name == "toyotagraph":
        dataset = {
            # "num_class": 1,
            #loss和deepmove暂且一致
            "loss_fn": nn.NLLLoss(ignore_index=0).cuda(),
            # "loss_fn": nn.CrossEntropyLoss(ignore_index=0).cuda(),
            # "metric": "cross_entropy",
            "metric": "NLLLoss",
            "metric_mode": "min",
            "evaluator": NodePropPredEvaluator(name="ogbn-arxiv"),
            "train_dataset": MyToyotaGraphDataset(
                subset=False, root="../dataset/toyotagraph", split="train"
            ),
            "val_dataset": MyToyotaGraphDataset(
                subset=False, root="../dataset/toyotagraph", split="test"
            ),
            "test_dataset": MyToyotaGraphDataset(
                subset=False, root="../dataset/toyotagraph", split="test"
            ),

            # "max_node": 2708,
        }


    else:
        raise NotImplementedError

    print(f" > {dataset_name} loaded!")
    print(dataset)
    print(f" > dataset info ends")
    return dataset


class GraphDataModule(LightningDataModule):
    name = "OGB-GRAPH"

    def __init__(
        self,
        dataset_name: str = "ogbg-molpcba",
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 1,
        multi_hop_max_dist: int = 5,
        rel_pos_max: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.rel_pos_max = rel_pos_max

    def setup(self, stage: str = None):
        if self.dataset_name == "foursquare":
            self.dataset_train = self.dataset["train_dataset"]
            self.dataset_val = self.dataset["val_dataset"]
            self.dataset_test = self.dataset["test_dataset"]

        elif self.dataset_name == "foursquaregraph":
            self.dataset_train = self.dataset["train_dataset"]
            self.dataset_val = self.dataset["val_dataset"]
            self.dataset_test = self.dataset["test_dataset"]
        
        elif self.dataset_name=="toyota":
            self.dataset_train = self.dataset["train_dataset"]
            self.dataset_val = self.dataset["test_dataset"]
            self.dataset_test = self.dataset["test_dataset"]
        
        # ADDED toyotagraph from graphormer_new_cuda2
        elif self.dataset_name == "toyotagraph":
            self.dataset_train = self.dataset["train_dataset"]
            self.dataset_val = self.dataset["val_dataset"]
            self.dataset_test = self.dataset["test_dataset"]
            
        elif self.dataset_name == 'gowalla_7day' or self.dataset_name == 'gowalla_nevda':
            self.dataset_train = self.dataset["train_dataset"]
            self.dataset_val = self.dataset["val_dataset"]
            self.dataset_test = self.dataset["test_dataset"]

        else:
            split_idx = self.dataset["dataset"].get_idx_split()
            self.dataset_train = self.dataset["dataset"][split_idx["train"]]
            self.dataset_val = self.dataset["dataset"][split_idx["valid"]]
            self.dataset_test = self.dataset["dataset"][split_idx["test"]]

    def train_dataloader(self):

        if self.dataset_name == "foursquare":
            loader = DataLoader(
                self.dataset["train_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_foursquare,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        elif self.dataset_name == "gowalla_7day" or self.dataset_name == "gowalla_nevda":
            loader = DataLoader(
                self.dataset["train_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_gowalla,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        elif self.dataset_name == "foursquaregraph":
            loader = DataLoader(
                self.dataset["train_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_foursquare,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        elif self.dataset_name=="toyota":
            loader = DataLoader(
                self.dataset["train_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        
        # ADDED toyotagraph from graphormer_new_cuda2
        elif self.dataset_name == "toyotagraph":
            loader = DataLoader(
                self.dataset["train_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_toyota,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        
        else:
            loader = DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        print("len(train_dataloader)", len(loader))
        return loader

    def val_dataloader(self):
        if self.dataset_name == "foursquare":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_foursquare,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
            
        elif self.dataset_name == "foursquaregraph":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_foursquare,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )

        elif self.dataset_name == "gowalla_7day" or self.dataset_name == "gowalla_nevda":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_gowalla,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )

        elif self.dataset_name=="toyota":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )

        # ADDED toyotagraph from graphormer_new_cuda2
        elif self.dataset_name == "toyotagraph":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_toyota,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )

        else:
            loader = DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=partial(
                    collator,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
            print("len(val_dataloader)", len(loader))
        return loader

    def test_dataloader(self):
        if self.dataset_name == "foursquare":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_foursquare,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
            
        elif self.dataset_name == "foursquaregraph":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_foursquare,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
            
        elif self.dataset_name == "gowalla_7day" or self.dataset_name == "gowalla_nevda":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_gowalla,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )

        elif self.dataset_name=="toyota":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        
        # ADDED toyotagraph from graphormer_new_cuda2
        elif self.dataset_name == "toyotagraph":
            loader = DataLoader(
                self.dataset["test_dataset"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(
                    collator_toyota,
                    max_node=30000,
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        
        else:
            loader = DataLoader(
                self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=partial(
                    collator,
                    max_node=get_dataset(self.dataset_name)["max_node"],
                    multi_hop_max_dist=self.multi_hop_max_dist,
                    rel_pos_max=self.rel_pos_max,
                ),
            )
        print("len(test_dataloader)", len(loader))
        return loader
