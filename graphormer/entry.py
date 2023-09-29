# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import print_function
from __future__ import division
# from model import Graphormer
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# from model_sparse_attention import Graphormer
# from model_fast_attention import Graphormer
from model_fqandtoyo import Graphormer
from data import GraphDataModule, get_dataset

from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import MLFlowLogger

import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import argparse
import numpy as np
import copy
import pandas as pd
import math
import pickle
from json import encoder

import sys
sys.path.insert(0, r'/home/kanezashi/Toyota/graphormer_new/graphormer')

# encoder.FLOAT_REPR = lambda o: format(o, '.3f')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from torch.nn.parallel import DataParallel, DistributedDataParallel

wandb_logger = WandbLogger(project="graphormer")
# torch.autograd.set_detect_anomaly(True)


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Graphormer.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)


    # ------------
    # model
    # ------------
    if args.checkpoint_path != "":
        model = Graphormer.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    else:
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=args.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    # model = torch.nn.DataParallel(model.to("cuda"), device_ids=[0, 1]) 
    if not args.test and not args.validate:
        print(model)
    print("total params:", sum(p.numel() for p in model.parameters()))
    # ------------
    # training
    # ------------
    metric = "valid_" + get_dataset(args.dataset_name)["metric"]
    dirpath = args.default_root_dir + f"/lightning_logs/checkpoints"
    print("#######")
    print(dirpath)
    print("#######")
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        filename=args.dataset_name + "-{epoch:03d}-{" + metric + ":.4f}",
        save_top_k=100,
        mode=get_dataset(args.dataset_name)["metric_mode"],
        save_last=True,
    )
    if not args.test and not args.validate and os.path.exists(dirpath + "/last.ckpt"):
        args.resume_from_checkpoint = dirpath + "/last.ckpt"
        print("args.resume_from_checkpoint", args.resume_from_checkpoint)

    # args.logger = wandb_logger
    print("##", torch.cuda.device_count(), torch.cuda.is_available())
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval="step"))
    # mlf_logger = MLFlowLogger(experiment_name="graphormer_mlf", save_dir='/fast/xuxh/mlruns')
    # trainer = pl.Trainer.from_argparse_args(args, logger=mlf_logger)
    # trainer.ddp_find_unused_parameters_only = False
    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer.callbacks.append(checkpoint_callback)
    # trainer.callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    # trainer.callbacks.append(pl.callbacks.StochasticWeightAveraging())
    # torch.backends.cudnn.enable =True
    # torch.backends.cudnn.benchmark = True

    if args.test:
        result = trainer.test(model, datamodule=dm)
        print(result)
    elif args.validate:
        result = trainer.validate(model, datamodule=dm)
        print(result)
    else:
        trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    print("####", torch.cuda.device_count(), torch.cuda.is_available())
    cli_main()
