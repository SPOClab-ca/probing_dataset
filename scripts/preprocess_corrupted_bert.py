import argparse 
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("../")
from scripts.load_data import senteval_load_file

parser = argparse.ArgumentParser()
parser.add_argument("corr_steps", type=int, default=200)
parser.add_argument("task", type=str)
args = parser.parse_args()

model_name = "bert-base-multilingual-cased"
model_path = f"../data/checkpoints/{model_name}-corrupt-{args.corr_steps}-steps"
device = torch.device("cuda")

model = AutoModelForMaskedLM.from_pretrained(model_path,
                                            output_hidden_states=True)
model.to(device)
model.eval()
print("Model loaded from {}, device: {}".format(model_path, model.device))

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_w_bert(task, bsize=4):
    x_list = []
    y_list = []
    data, n_class = senteval_load_file(f"../data/senteval/{task}.txt")
    print("Loaded task {} data: len={}".format(task, len(data)))
    for i in tqdm(range(0, len(data), bsize)):
        batch_text = [d['X'] for d in data[i:i+bsize]]
        batch_inputs = tokenizer(batch_text,
            truncation=True, max_length=512, padding='max_length', 
                                 return_tensors='pt')
        batch_inputs = {k:v.to(device) for k,v in batch_inputs.items()}
        with torch.no_grad():
            batch_output = model(**batch_inputs)
            hids = batch_output.hidden_states
            pooler_output = hids[-1][:, 0]  # (bsize, D)
            x_list.append(pooler_output.cpu().detach())
            y_list.extend([d['y'] for d in data[i:i+bsize]])
    x_list = torch.cat(x_list, dim=0).numpy()
    y_list = np.array(y_list, dtype=np.int16)
    
    save_path = f"../data/senteval/{task}.bert_corr{args.corr_steps}"
    torch.save({"X": x_list, 'y': y_list}, save_path)
    print("Saved preprocessed representations!")
    
preprocess_w_bert(args.task)