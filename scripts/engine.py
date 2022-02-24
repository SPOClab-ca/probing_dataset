import os
from functools import partial
from collections import OrderedDict
import json 
import numpy as np
import pandas as pd
from pathlib import Path
from ray.tune.stopper import CombinedStopper, \
        MaximumIterationStopper, TrialPlateauStopper
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from sklearn.metrics import accuracy_score, f1_score
import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import wandb

from BayesianLayers import LinearGroupNJ
from load_data import get_senteval_data_preprocessed, get_ud_pos_data, get_collate_fn, get_olmpics_data
from logging_utils import get_logger
logger = get_logger('Engine')

class ProbingEngine:
    
    def __init__(self, args):
        self.args = args
    
    def setup_ray_tune(self):
        ray.init(
            object_store_memory=20 * 10**9, # 20GB
        )

        config = {
            "lr": tune.grid_search(self.args.lr_list),
            "batch_size": tune.grid_search(self.args.bs_list)
        }

        scheduler = ASHAScheduler(
            time_attr='epoch',
            metric="val_acc",
            mode="max",
            max_t=self.args.max_epochs,
            grace_period=self.args.min_epochs,
            reduction_factor=2)

        reporter = CLIReporter(
            parameter_columns=["lr", "batch_size"],
            metric_columns=["val_loss", "val_acc", "training_iteration"])

        return config, scheduler, reporter

    def _run(self, config, scheduler, reporter, resume=False):
        args = self.args
        return tune.run(
            tune.with_parameters(Trainer, args=args),
            verbose=2,
            resources_per_trial={"cpu": 4, "gpu": 1},
            stop=CombinedStopper(
                MaximumIterationStopper(max_iter=args.max_epochs),
                TrialPlateauStopper(metric="val_loss", num_results=5, grace_period=args.min_epochs)
            ),
            config=config,
            num_samples=1,
            scheduler=scheduler,
            progress_reporter=reporter,
            checkpoint_at_end=False,
            local_dir=os.path.join(args.ray_tune_result_path, args.main_task),
            name=f"{args.task}-{args.model}-{args.seed}-{args.slurm_id}",
            log_to_file=True,
            reuse_actors=True,
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_score_attr="min-val_loss",
            resume=resume)
        
    def probe(self):
        args = self.args

        config, scheduler, reporter = self.setup_ray_tune()

        try:
            result = self._run(config, scheduler, reporter, args.resume)
        except ValueError as e:
            if args.resume:
                logger.error(f"{e} Try from scratch instead.")
                result = self._run(config, scheduler, reporter, False)
            else:
                raise e
        return result

class PrequentialMDLProbingEngine(ProbingEngine):
    def __init__(self, args):
        super().__init__(args) 

    def probe(self):
        opt = self.args
        probe = MLP(
            d_in = opt.d_in,
            nclasses = self.nclasses,
            hid_sizes=opt.hid_sizes
        )
        if opt.use_cuda:
            probe.cuda()

        val_dataloader = DataLoader(self.val_data, collate_fn=get_collate_fn(self.batcher, opt.use_cuda, opt.task), batch_size=opt.batch_size)
        trainer = Trainer(
            min_epochs=1,
            max_epochs=opt.max_epochs,
            use_cuda=opt.use_cuda,
            checkpoint=None  # Handle checkpointing here
        )

        N = len(self.train_data)
        
        mdl_time_slices = (np.array([0.0010, 0.0020, 0.0040, 0.0080, 0.0160, 0.0320, 0.0625, 0.1250, 0.2500, 0.5000, 1.0000]) * N).astype(int)
        mdl_time_slices = mdl_time_slices[mdl_time_slices >= opt.batch_size]  # Can't train model for too small timeslice. Use random coding for t0... t_{i} where i is the first time slice allowing one batch to pass
        
        wandb.log({"pmdl_n_slices": len(mdl_time_slices)})
        
        if Path(opt.checkpoint).exists():
            with open(opt.checkpoint, "rb") as f:
                checkpoint = torch.load(f)
            i = checkpoint["timeslice_i"]
            codelength = checkpoint["codelength"]
        else:
            i = 0
            codelength = mdl_time_slices[0] * np.log2(self.nclasses)

        result = dict()
        while i < len(mdl_time_slices)-1:
            start, end = mdl_time_slices[i], mdl_time_slices[i+1]
            train_data = self.train_data[start: end]
            train_dataloader = DataLoader(train_data, collate_fn=get_collate_fn(self.batcher, opt.use_cuda, opt.task), batch_size=opt.batch_size)
            optim = torch.optim.Adam(probe.parameters(), opt.init_lr) 
            result = trainer.fit(probe, optim, train_dataloader, val_dataloader)
            codelength += (result["mean_train_loss"] / np.log2(np.e))
            result["timeslice_end"] = end
            result["prequential_mdl"] = codelength 
            i += 1
            result["global_epoch"] = i * result["epoch"]
            with open(opt.checkpoint, "wb") as f:
                torch.save({"timeslice_i": i, "codelength": codelength}, f)
        return result 


class Trainer(tune.Trainable):
    def _load_data(self):
        args = self.args
        train_data, val_data, test_data, nclasses, task_type = None, None, None, None, None
        if args.task.startswith("ud_pos_"):
            train_data, val_data, test_data, nclasses = get_ud_pos_data(args)
        elif args.task.startswith("olmpics_"):
            #TODO: change to preprocessed data loading
            train_data, val_data, test_data, nclasses, task_type = get_olmpics_data(args.task.replace("olmpics_", ""), args)
        elif args.task.startswith("cats_"):
            pass
        else:
            train_data, val_data, test_data, nclasses = get_senteval_data_preprocessed(
                                                os.path.join(args.project_path,
                                                            'data/senteval',
                                                            "{}.{}".format(args.task, args.model)),
                                                args)
        return {'train_data': train_data, 'val_data': val_data, 'test_data': test_data,
                'nclasses': nclasses, 'task_type': task_type}
    
    def _init_model(self, nclasses=None, task_type=None):
        args = self.args
        if args.task.startswith("olmpics") and task_type=='mlm':
            # Masking language model task
            return MLM_MLP(
                d_in = args.d_in,
                nclasses = self.nclasses,
                hid_sizes = args.hid_sizes
            ).to(self.device)
        elif args.task.startswith("olmpics") and task_type=='qa':
            # Question answering task
            return QA_MLP(
                d_in = args.d_in,
                hid_sizes = args.hid_sizes
            ).to(self.device)
        else:
            # Sentence classification task
            if args.probe_metric == "variational_mdl":
                return BayesianMLP(
                    d_in = args.d_in,
                    nclasses = nclasses,
                    hid_sizes = args.hid_sizes
                ).to(self.device)
            else:
                return MLP(
                    d_in = args.d_in,
                    nclasses = nclasses,
                    hid_sizes = args.hid_sizes
                ).to(self.device)
        
    
    def setup(self, config, args=None):
        self.args = args
        self.config = config
        self.device = torch.device("cuda") if args.use_cuda else torch.device("cpu")

        # Loading data
        data = self._load_data()
        # Dataloader 
        self.train_dl = DataLoader(data['train_data'], batch_size=config['batch_size'], shuffle=True)
        self.val_dl = DataLoader(data['val_data'], batch_size=config['batch_size'], shuffle=False)
        self.test_dl = DataLoader(data['test_data'], batch_size=config['batch_size'], shuffle=False)
        # Model
        self.nclasses = data['nclasses']
        self.task_type = data['task_type']
        self.model = self._init_model(self.nclasses, self.task_type)
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), config['lr'])

        self.min_val_loss = np.inf 

    def reset_config(self, new_config):
        self.model = self._init_model(self.nclasses, self.task_type)
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), new_config['lr'])
        self.min_val_loss = np.inf 
        self.config = new_config
        return True
    
    def step(self):
        results = None
        epoch_start_time = time.time()
        train_losses = self._train_one_epoch()
        mean_train_loss = np.mean(train_losses)
        area_under_learning_curve = np.sum(train_losses)

        # Eval
        val_loss_mean, val_acc, val_f1, val_preds, val_labels = self._eval(self.val_dl)
        test_loss_mean, test_acc, test_f1, test_preds, test_labels = self._eval(self.test_dl)

        results = {
            "mean_train_loss": mean_train_loss,
            "area_under_learning_curve": area_under_learning_curve,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_loss": val_loss_mean,
            "test_acc": test_acc, 
            "test_f1": test_f1,
            "test_loss": test_loss_mean,
            "test_preds": json.dumps(test_preds),
            "test_labels": json.dumps(test_labels),
            "epoch_time": time.time() - epoch_start_time
        }

        return results

    def save_checkpoint(self, checkpoint_dir):
        ckp_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        with open(ckp_path, "wb") as f:
            torch.save({"model": self.model.state_dict(),
                        "optim": self.optim.state_dict()}, f)
            # logger.info(f"Saved checkpoint to {ckp_path}")
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        ckp_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        checkpoint = torch.load(ckp_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optim"])
        # logger.info(f"Loaded checkpoint from {ckp_path}")

    def _train_one_epoch(self):
        self.model.train()
        train_losses = []
        for batch_x, batch_y in self.train_dl:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits, loss = self.model(batch_x, batch_y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            train_losses.append(loss.item())
        return train_losses
    
    def _eval(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits, loss = self.model(batch_x, batch_y)

                if type(batch_y) is tuple:
                    # Account for the MLM task where batch_y is a tuple of (yindex, y).
                    # yindex is the vocab index of choices and y is the correct choice.
                    batch_y = batch_y[-1] 
                
                maxval, pred = logits.max(dim=1)
                val_loss.append(loss.item())
                all_preds.extend(pred.cpu().numpy().tolist())
                all_labels.extend(batch_y.cpu().numpy().tolist())
        acc = accuracy_score(all_labels, all_preds) 
        f1 = f1_score(all_labels, all_preds, average="micro")
        loss_mean = np.mean(val_loss) 
        return loss_mean, acc, f1, all_preds, all_labels

class MLP(nn.Module):
    def __init__(self, d_in, nclasses, hid_sizes):
        super().__init__()
        d = OrderedDict()
        dims = [d_in] + hid_sizes + [nclasses]
        i = 0
        for _ in range(len(dims)-2):
            d[f'fc_{i}'] = nn.Linear(dims[i], dims[i+1])
            d[f'relu_{i}'] = nn.ReLU()
            i += 1
        d[f'fc_{i}'] = nn.Linear(dims[i], dims[i+1])
        self.network = nn.Sequential(d)

    def forward(self, x, y=None):
        logits = self.network(x)
        if y is not None:
            loss = nn.CrossEntropyLoss()(logits, y)
            return logits, loss 
        else:
            return logits, None 

class MLM_MLP(MLP):
    def __init__(self, d_in, nclasses, hid_sizes):
        super().__init__(d_in, nclasses, hid_sizes)

    def forward(self, x, y=None):
        logits = self.network(x)
        if y is not None:
            yindex, yid = y
            logits = logits.gather(dim=1, index=yindex)
            loss = nn.CrossEntropyLoss()(logits, yid)
            return logits, loss 
        else:
            return logits, None

class QA_MLP(MLP):
    def __init__(self, d_in, hid_sizes):
        super().__init__(d_in, 1, hid_sizes)

    def forward(self, x, y=None):
        logits = self.network(x) # (bsz, nclasses, 1)
        logits = logits.squeeze(-1) # (bsz, nclasses)
        if y is not None:
            loss = nn.CrossEntropyLoss()(logits, y)
            return logits, loss 
        else:
            return logits, None 

class BayesianMLP(nn.Module):
    def __init__(self, d_in, nclasses, hid_sizes):
        super().__init__()
        networks = OrderedDict()
        dims = [d_in] + hid_sizes + [nclasses]
        self.layers = []
        for i in range(len(dims)-1):
            layer = LinearGroupNJ(dims[i], dims[i+1])
            networks[f"fc_{i}"] = layer
            self.layers.append(layer)
        self.networks = nn.Sequential(networks)
    
    def forward(self, x: torch.tensor, y=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:  # No ReLU at the last layer
                x = nn.ReLU()(x)
        logits = x
        if y is not None:
            disc_loss = nn.functional.cross_entropy(logits, y)
            variational_bound = disc_loss + self.kl_divergence() / x.shape[0]
            return logits, variational_bound
        else:
            return logits, None
        
    def kl_divergence(self):
        result = 0 
        for layer in self.layers:
            result += layer.kl_divergence()
        return result