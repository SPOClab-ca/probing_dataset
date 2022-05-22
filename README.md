# probing_dataset
Repository for the paper "On the data requirements of probing"


## Environment  
This repo was developed using packages of these versions:  
```
transformers==4.3.2
wandb==0.10.30
torch==1.8.1
torchtext==0.9.1
torchvision==0.9.1
spacy==3.0.6
tensorboard==2.4.1
sentence-transformers==1.1.1
```

## Steps to reproduce the findings
1. Preprocess the embeddings: `python preprocess_data.py`  
- There are currently the preprocessors for SentEval, CATS, and oLMpics. The paper only reports experiments for SentEval (fixed-class problem)  
- For the corrupted models, use `preprocess_corrupted_bert.py`  


2. Run the probing experiments (on slurm): 
```bash
size_per_class=128
python run_senteval.py  \
    --project_path <path_to_github>/probing_dataset \
    --model bert --task bigram_shift --seed 0 \
    --even_distribute --train_size_per_class ${size_per_class} --val_size_per_class ${size_per_class} \
    --lr_list 1e-4 5e-4 1e-3 5e-3 1e-2 \
    --bs_list 8 16 32 64 \
    --use_cuda --probe_metric "others" \
    --wandb_id_file_path "/checkpoint/$USER/$SLURM_JOB_ID/wandb_id.txt" \
    --checkpoint "/checkpoint/$USER/$SLURM_JOB_ID/checkpoint.ckpt" \
    --ray_tune_result_path "${project_path}/results" \
    --resume
```

3. Download the probing results on `wandb.ai`. The logged results include both the performance metrics and the test predictions.  

4. Head to the corresponding ipynb in `notebooks` directory to further analyze the results:  
- `theory_vs_experiments.ipynb`: Experiment 4.2  
- `power_curves.ipynb`: Experiment 4.3 - 4.6    


Helper files:  
- `learning_theory.py`  
- `power_analysis.py`: Based on [the repo of Card etal (2020)](https://github.com/dallascard/NLP-power-analysis/blob/master/notebooks_for_power_calculations/accuracy.ipynb)  
- `load_data.py`: Load some data.  
- `BayesianLayers.py`: Used for variational MDL probing.  
- `engine.py`: engine for probing classification.  
- `notebooks/worse_finetuning.ipynb`: Notebook for corruption-pretraining Transformer LMs.  


## Reference
```
@inproceedings{zhu_etal_data_2022,
    title = {{On the data requirements of probing}},
    author = {Zhu, Zining and Wang, Jixuan and Li, Bai and Rudzicz, Frank},
    year={2022},
    url={https://aclanthology.org/2022.findings-acl.326/},
    booktitle={{Findings of the Association of Computational Linguistics}}
}
```