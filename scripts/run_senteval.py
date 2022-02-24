import argparse
import numpy as np 
import torch 
import wandb
from typing import List, Tuple, Any 

from logging_utils import get_logger
logger = get_logger('SentEval')

from engine import ProbingEngine, PrequentialMDLProbingEngine
from utils import init_or_resume_wandb_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project_path", type=str)
    parser.add_argument("--model", type=str, choices=["bert", "bert_corr200", "bert_corr800", "bert_corr3200", "sbert", "infersent", "glove"]) 
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_entity", type=str, default="linux")
    parser.add_argument("--wandb_id_file_path", type=str, default="wandb_id_file.txt")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.ckpt")
    parser.add_argument("--slurm_id", type=str, required=True)
    parser.add_argument("--ray_tune_result_path", type=str)
    
    # Dataset args
    parser.add_argument("--task", choices=['bigram_shift', 'coordination_inversion', 'obj_number', 'odd_man_out', 'past_present', 'sentence_length', 'subj_number', 'top_constituents', 'tree_depth', 'word_content'], default="bigram_shift")
    parser.add_argument("--even_distribute", action="store_true")
    parser.add_argument("--train_size_per_class", type=int, default=10, help="Activated only when even_distribute is activated")
    parser.add_argument("--val_size_per_class", type=int, default=-1, help="Activated only when even_distribute is activated. If set to -1, follow the 80-20 convention to set val_size_per_class to be int(0.25*train_size_per_class)")
    parser.add_argument("--representation_gaussian_noise", type=float, default=0., help="Add Gaussian noise with this scale (std) into representations")

    # Probe, training args
    parser.add_argument("--probe_metric", type=str, choices=["prequential_mdl", "variational_mdl", "others"], default="others")
    parser.add_argument("--d_in", type=int, default=768)
    parser.add_argument("--hid_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=50)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr_list", nargs="*", type=float)
    parser.add_argument("--bs_list", nargs="*", type=int)
    
    args = parser.parse_args()

    args.main_task = 'senteval'
    
    if args.val_size_per_class == -1:
        args.val_size_per_class = int(0.25 * args.train_size_per_class) 

    # Set random seeds 
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)

    # Specify model
    batcher = None
    if args.model.startswith("bert"):
        pass
    elif args.model == "sbert":
        pass
    elif args.model == "infersent":
        args.d_in = 4096
    elif args.model == "glove":
        args.d_in = 300
    else:
        raise ValueError

    print(args)
    
    # Initialize wandb
    wandb_config = init_or_resume_wandb_run(
        args.wandb_id_file_path,
        project_name="probing_dataset",
        run_name=f"senteval_{args.task}_{args.model}",
        config={
        "model": args.model,
        "seed": args.seed,
        "task": args.task,
        "even_distribute": args.even_distribute,
        "train_size_per_class": args.train_size_per_class,
        "val_size_per_class": args.val_size_per_class,
        "representation_gaussian_noise": args.representation_gaussian_noise,
        "probe_metric": args.probe_metric,
        "d_in": args.d_in,
        "hid_sizes": args.hid_sizes,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "use_cuda": args.use_cuda,
        "slurm_id": args.slurm_id
    })

    # Specify probing task
    if args.probe_metric == "prequential_mdl":
        mdl_pe = PrequentialMDLProbingEngine(batcher, args)
        results = mdl_pe.probe()
    elif args.probe_metric == "variational_mdl":
        pe = ProbingEngine(args)
        results = pe.probe()
    else:
        pe = ProbingEngine(args)
        results = pe.probe()

    best_trial = results.get_best_trial("val_acc", "max", "all")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {:.4f}, accuracy: {:.4f}".format(
            best_trial.last_result["val_loss"],
            best_trial.last_result["val_acc"]))
    logger.info("Best trial final test loss: {:.4f}, accuracy: {:.4f}".format(
            best_trial.last_result["test_loss"],
            best_trial.last_result["test_acc"]))
    
    wandb.log({
        'best_val_loss':  best_trial.last_result["val_loss"],
        'best_val_acc':  best_trial.last_result["val_acc"],
        'test_loss':  best_trial.last_result["test_loss"],
        'test_acc':  best_trial.last_result["test_acc"],
        'best_hyperparameters': best_trial.config,
        'test_predictions':  best_trial.last_result["test_preds"],
        'test_labels':  best_trial.last_result["test_labels"]
    })