import os 
import glob
import json
import sys
from typing import List, OrderedDict
import numpy as np 
from pathlib import Path
import torch
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
sys.path.insert(0, "../libs/InferSent")
from models import InferSent
from load_data import senteval_load_file
from nltk import word_tokenize

OLMPICS_DATA_PATH = '../data/oLMpics/data'
MAXLEN=512

class Preprocessor(object):
    def __init__(self, skip_existing=False) -> None:
        super().__init__()
        self.skip_existing = skip_existing
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    
class SentEvalPreprocessor(Preprocessor):
    data_path = "../data/senteval/"
    file_path = data_path + "{}.txt"

    def __init__(self, skip_existing) -> None:
        super().__init__(skip_existing=skip_existing)

        tasks = []
        for dpath in glob.glob(self.file_path.format("*")):
            task_name = os.path.basename(dpath).split(".")[0]
            tasks.append(task_name)
        self.tasks = tasks
            
        self.bert, self.bert_tokenizer = self.prepare_bert()

    def preprocess_all_w_bert(self):
        for task in self.tasks:
            self.preprocess_w_bert(task)

    def preprocess_all_w_sbert(self):
        for task in self.tasks:
            self.preprocess_w_sbert(task)

    def preprocess_all_w_glove(self):
        glove_model, tokenizer = self.prepare_glove()
        for task in self.tasks:
            self.preprocess_w_glove(task, glove_model, tokenizer)
    
    def preprocess_all_w_infersent(self):
        for task in self.tasks:
            self.preprocess_w_infersent(task)

    def prepare_bert(self):
        model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") 
        model.to(self.device)
        model.eval()
        return model, tokenizer 
    
    def preprocess_w_bert(self, task, bsize=50):
        save_path = os.path.join(self.data_path, f"{task}.bert")
        if self.skip_existing and os.path.exists(save_path):
            print(f"Preprocessed file found at {save_path}. Skip it.")
            return 

        bert, tokenizer = self.prepare_bert()

        x_list = []
        y_list = []
        data, n_classes = senteval_load_file(filepath=self.file_path.format(task))
        for i in tqdm(range(0, len(data), bsize)):
            batch_text = [d['X'] for d in data[i:i+bsize]]
            batch_inputs = tokenizer(batch_text,
                                        truncation=True,
                                        max_length=MAXLEN,
                                        padding='max_length',
                                        return_tensors="pt")
            batch_inputs = {k:v.to(self.device) for k,v in batch_inputs.items()}
            with torch.no_grad():
                batch_output = bert(**batch_inputs).pooler_output
                x_list.append(batch_output.cpu())
                y_list += [d['y'] for d in data[i:i+bsize]]
        x_list = torch.cat(x_list, dim=0).numpy()
        y_list = np.array(y_list, dtype=np.int16)

        torch.save({'X': x_list, 'y': y_list}, save_path)
        print(f"Saved to {save_path}")

    def preprocess_w_sbert(self, task, bsize=50):
        save_path = os.path.join(self.data_path, f"{task}.sbert")
        if self.skip_existing and os.path.exists(save_path):
            print(f"Preprocessed file found at {save_path}. Skip it.")
            return 
        
        sbert = self.prepare_sbert()

        x_list = []
        y_list = []
        data, n_classes = senteval_load_file(filepath=self.file_path.format(task))
        for i in tqdm(range(0, len(data), bsize)):
            batch_text = [d['X'] for d in data[i:i+bsize]]
            x_list.append(sbert.encode(batch_text))
            y_list += [d['y'] for d in data[i:i+bsize]]
        x_list = np.concatenate(x_list, axis=0)
        y_list = np.array(y_list, dtype=np.int16)

        torch.save({'X': x_list, 'y': y_list}, save_path)
        print(f"Saved to {save_path}")

    def preprocess_w_infersent(self, task, bsize=50):
        save_path = os.path.join(self.data_path, f"{task}.infersent")
        if self.skip_existing and os.path.exists(save_path):
            print(f"Preprocessed file found at {save_path}. Skip it.")
            return 
        
        infersent_model = self.prepare_infersent()

        x_list = []
        y_list = []
        data, n_classes = senteval_load_file(filepath=self.file_path.format(task))
        for i in tqdm(range(0, len(data), bsize)):
            batch_text = [d['X'] for d in data[i:i+bsize]]
            x_list.append(infersent_model.encode(batch_text, bsize=len(batch_text), tokenize=False))
            y_list += [d['y'] for d in data[i:i+bsize]]
        x_list = np.concatenate(x_list, axis=0)
        y_list = np.array(y_list, dtype=np.int16)

        torch.save({'X': x_list, 'y': y_list}, save_path)
        print(f"Saved to {save_path}")

    def preprocess_w_glove(self, task, glove_model, tokenizer, bsize=50):
        save_path = os.path.join(self.data_path, f"{task}.glove")
        if self.skip_existing and os.path.exists(save_path):
            print(f"Preprocessed file found at {save_path}. Skip it.")
            return 
        
        data, n_classes = senteval_load_file(filepath=self.file_path.format(task))
        x_list = []
        y_list = [d['y'] for d in data]
        for dt in tqdm(data):
            sent_vecs = []
            for w in tokenizer(dt['X']):
                sent_vecs.append(glove_model.get(w, np.zeros(300)))
            x_list.append(np.array(sent_vecs).mean(axis=0, keepdims=True).astype(float))

        x_list = np.concatenate(x_list, axis=0)
        y_list = np.array(y_list, dtype=np.int16)

        torch.save({'X': x_list, 'y': y_list}, save_path)
        print(f"Saved to {save_path}")

    def prepare_sbert(self):
        model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        model.to(self.device)
        return model

    def prepare_infersent(self):
        MODEL_PATH = "../data/encoder/infersent1.pkl"
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        infersentmodel = InferSent(params_model)
        infersentmodel.load_state_dict(torch.load(MODEL_PATH))

        W2V_PATH = "../data/GloVe/glove.840B.300d.txt"
        infersentmodel.set_w2v_path(W2V_PATH) 
        infersentmodel.build_vocab_k_words(K=100000)

        return infersentmodel.to(self.device)

    def prepare_glove(self):
        print('Loading glove...')
        word_vec = {}
        with open("../data/GloVe/glove.840B.300d.txt", encoding="utf-8") as f:
            for line in tqdm(f):
                word, vec = line.split(" ", 1)
                word_vec[word] = np.fromstring(vec, sep=' ')
        model = word_vec 
        tokenizer = word_tokenize
        return model, tokenizer 

class CATSPreprocessor(object):
    CATS_ABILITY_PATH = '../data/CATS/commonsense_ability_test'
    CATS_ROBUST_PATH = '../data/CATS/Robust_commonsense_test'
    robust_tasks = ["add", "del", "sub", "swap"]
    ability_tasks = ["arct_1", "arct_2", "ca", "hella_swag", "sm", "smr", "swag", "wsc"]

    def __init__(self, skip_existing=False) -> None:
        super().__init__()
        self.skip_existing = skip_existing
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.bert_model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        self.bert_model.to(self.device)
        self.bert_model.eval()
    
    def _scoring_sentences(self, sent_list):
        """Scoring a list of sentences."""
        return np.array([self._sent_scoring(sent) for sent in tqdm(sent_list)], dtype=np.float64)
        
    def _sent_scoring(self, text):
        # Tokenized input
        # text = "[CLS] I got restricted because Tom reported my reply [SEP]"
        if not "[SEP]" in text:
            inputs = self.bert_tokenizer(text,
                                            truncation=True,
                                            max_length=MAXLEN,
                                            padding='max_length',
                                            return_tensors="pt")
        else:
            sent_1, sent_2 = text.split("[SEP]")
            inputs = self.bert_tokenizer(sent_1, sent_2,
                                            truncation="longest_first",
                                            max_length=MAXLEN,
                                            padding='max_length',
                                            return_tensors="pt")
        
        special_token_ids = (self.bert_tokenizer.cls_token_id,
                             self.bert_tokenizer.sep_token_id,
                             self.bert_tokenizer.pad_token_id)

        mask_token_id = self.bert_tokenizer.mask_token_id
                                    
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        masked_input_ids = []
        labels = []
        for i in range(len(input_ids)):
            if attention_mask[i] == 1 and input_ids[i] not in special_token_ids:
                # Saved the token id to be masked.
                masked_ids = input_ids.clone()
                # Change to the mask token id.
                masked_ids[i] = mask_token_id
                masked_input_ids.append(masked_ids)
                labels.append(input_ids[i])
        masked_input_ids = torch.stack(masked_input_ids)
        labels = torch.tensor(labels, dtype=torch.long)

        masked_inputs = {"input_ids": masked_input_ids}
        for k in inputs:
            if k not in masked_inputs:
                repeat_shape = [1] * len(inputs[k].shape)
                repeat_shape[0] = masked_input_ids.shape[0]
                masked_inputs[k] = inputs[k].repeat(*repeat_shape)

        with torch.no_grad():
            bsize = 20
            log_p_label_list = []
            for i in range(0, masked_inputs['input_ids'].shape[0], bsize):
                masked_inputs_batch = {k:v[i:i+bsize].to(self.device) for k,v in masked_inputs.items()}
                output_logits = self.bert_model(**masked_inputs_batch).logits # (bsize, seq_length, vocab_size)
                masked_logits = output_logits[masked_inputs_batch["input_ids"] == self.bert_tokenizer.mask_token_id] # (bsize, vocab_size)
                log_p = F.log_softmax(masked_logits, dim=-1) # (bsize, vocab_size)
                log_p_label_list.append(log_p[range(log_p.shape[0]), labels[i:i+bsize]]) # (bsize)
                del output_logits
                del masked_logits

            sum_log_p_label = torch.cat(log_p_label_list, dim=0).sum().item()
            num_reg_tokens = input_ids.apply_(lambda x: x not in special_token_ids).int().sum() 

            return (sum_log_p_label / num_reg_tokens).item()

    def process(self):
        self.process_robust_tasks()
        self.process_ability_tasks()
    
    def process_robust_tasks(self):
        """Preprocessing 'robust' tasks."""
        for name in self.robust_tasks:
            processed_file_path = os.path.join(self.CATS_ROBUST_PATH, f"{name}_scores.pt")
            if self.skip_existing and os.path.exists(processed_file_path):
                print(f"Preprocessed file found at {processed_file_path}. Skip it.")
                continue

            lines = Path(self.CATS_ROBUST_PATH, f"{name}.txt").read_text().split("\n")
            label_list = []
            sentence_list = []

            for line in lines:
                if line.strip():
                    assert len(line.split("\x01")) == 6, "Each line of any robust task should has the following format: label, sent_1, sent_2, label, sent_1, sent_2"
                    temp = line.split("\x01")
                    label_list.append(temp[0])
                    label_list.append(temp[3])
                    sentence_list += temp[1:3]
                    sentence_list += temp[4:6]
            
            labels = np.array(label_list, dtype=np.int16)
            scores = self._scoring_sentences(sentence_list)
            scores = scores.reshape(-1, 2)

            assert scores.shape[0] == labels.shape[0], "Number of sentence pairs and number of labels do not match."

            torch.save({"scores": scores, "labels": labels}, processed_file_path)    
            print(f"Saved to {processed_file_path}")

    def process_ability_tasks(self):
        """Prreprocessing 'ability' tasks."""
        for name in self.ability_tasks:
            processed_file_path = os.path.join(self.CATS_ABILITY_PATH, f"{name}_scores.pt")
            if self.skip_existing and os.path.exists(processed_file_path):
                print(f"Preprocessed file found at {processed_file_path}. Skip it.")
                continue

            lines = Path(self.CATS_ABILITY_PATH, f"{name}.txt").read_text().split("\n")
            label_list = []
            sentence_list = []

            num_labels = len(lines[0].split("\x01")) - 1
            for line in lines:
                if line.strip():
                    temp = line.split("\x01")
                    label_list.append(temp[0])
                    sentence_list += temp[1:]
            
            labels = np.array(label_list, dtype=np.int16)
            scores = self._scoring_sentences(sentence_list)
            scores = scores.reshape(-1, num_labels)

            assert scores.shape[0] == labels.shape[0], "Number of sentence pairs and number of labels do not match."

            torch.save({"scores": scores, "labels": labels},
                       os.path.join(self.CATS_ABILITY_PATH, f"{name}_scores.pt"))    
            print(f"Saved to {processed_file_path}")
    
class oLMpicsPreprocessor(object):
    task_type = OrderedDict([
        ("antonym_synonym_negation", "mlm"),
        ("coffee_cats_quantifiers", "mlm"),
        ("composition_v2", "qa"),
        ("compositional_comparison", "mlm"),
        ("conjunction_filt4", "qa"),
        ("hypernym_conjunction", "mlm"),
        ("number_comparison_age_compare_masked", "mlm"),
        ("size_comparison", "mlm"),
    ])

    def __init__(self, skip_existing=True) -> None:
        super().__init__()
        self.bert, self.bert_tokenizer = self._prepare_bert()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.bert.to(self.device)
        self.skip_existing = skip_existing

    def _prepare_bert(self):
        model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        return model, tokenizer 

    def process(self) -> None:
        for task, task_type in self.task_type.items():
            print('Preprocessing task: {}'.format(task))
            if task_type == 'mlm':
                self._process_mlm_task(task, 'train')
                self._process_mlm_task(task, 'dev')
            else: # task_type == 'qa'
                self._process_qa_task(task, 'train')
                self._process_qa_task(task, 'dev')

    def extract_emb_x_y(self, batch_x: List[str], batch_y: List[List[str]]) -> np.array:
        """Extract embeddings for sentence pairs.

        Each x in batch_x is a question and each list in batch_y is the choices.
          Concatenate each x with a list of y respectively. Then get the contextual
          embeddings of the CLS tokens for all sentence pairs. 

        Args:
            batch_x (List[str]): A list of questions.
            batch_y (List[List[str]]): Contains the list of choices for every question in batch_x.

        Returns:
            np.array: [description]
        """
        assert len(batch_x) == len(batch_y)
        bsz = len(batch_x)
        npair_per_sent = len(batch_y[0])
        inputs_list = []
        bert_batch_size = 200

        for i in tqdm(range()):
            inputs = self.bert_tokenizer(x, y,
                                        truncation='only_first',
                                        max_length=MAXLEN,
                                        padding='max_length',
                                        return_tensors="pt")
            inputs_list.append(inputs)
        
        # Merge inputs.
        inputs = {k: torch.cat([inp[k] for inp in inputs_list], dim=0) for k in inputs_list[0]} 
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
 
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, batch_tensor.shape[0], b_size), desc='Extract embeddings'):
                embs = self.bert(batch_tensor[i:i+b_size])[0]
                embeddings.append(embs.cpu()[:, 0, :])
        embeddings = torch.cat(embeddings, dim=0) # (bsz*npair_per_sent, MAXLEN, D)

        embeddings = embeddings.reshape(bsz, npair_per_sent, -1) # (bsz, npair_per_sent, D)
        return embeddings

    def extract_mask_emb_and_y_ids(self, batch_x: List[str], batch_choices: List[List[str]], batch_label: List[int]) -> np.array:
        """
        Input: 
            batch: list (len batch size) of str. Each str contains a [MASK] token.
        Output:
            embeddings: np.array of shape (batch size x D). Each row is the 
                embedding of the [MASK] token in the corresponding str.
        """
        bsz = len(batch_x)
        x_ids = []
        y_ids = []
        y = []
        for x, ylist, label in tqdm(zip(batch_x, batch_choices, batch_label), desc='Tokenization'):

            yid_list = [self.bert_tokenizer.convert_tokens_to_ids(y) for y in ylist]
            if any([yid == self.bert_tokenizer.unk_token_id for yid in yid_list]):
                # If any choice contains more than a single vocab token, skip this example.
                continue
            y_ids.append(yid_list)
            y.append(label)
                
            ids = self.bert_tokenizer.encode(x,
                                            truncation=True,
                                            max_length=MAXLEN,
                                            padding='max_length')
            if not self.bert_tokenizer.mask_token_id in ids:
                # If [MASK] token was truncated, re-truncating the sequence by
                # keep the tokens around the [MASK] token.
                x_num = MAXLEN - 2
                x_tokens = self.bert_tokenizer.tokenize(x)
                assert self.bert_tokenizer.mask_token in x_tokens
                mask_index = x_tokens.index(self.bert_tokenizer.mask_token)
                last_token_index = mask_index + min(x_num//2, len(x_tokens)-mask_index-1)
                first_token_index = last_token_index - x_num + 1
                x_tokens = x_tokens[first_token_index : last_token_index + 1]

                ids = [self.bert_tokenizer.cls_token_id] + self.bert_tokenizer.convert_tokens_to_ids(x_tokens) + [self.bert_tokenizer.seq_token_id]
                assert len(ids) == MAXLEN
            x_ids.append(ids)
            
        xids_tensor = torch.tensor(x_ids)
        yids_tensor = torch.tensor(y_ids)
        y_tensor = torch.tensor(y)

        # Extract embeddings.
        embeddings = []
        xids_tensor = xids_tensor.to(self.device)
        with torch.no_grad():
            b_size = 200
            for i in tqdm(range(0, xids_tensor.shape[0], b_size), desc='Extract embeddings'):
                subset_ids = xids_tensor[i:i+b_size]
                embs = self.bert(subset_ids)[0]
                mask_indexes = (subset_ids == self.bert_tokenizer.mask_token_id).nonzero(as_tuple=True)
                embeddings.append(embs.cpu()[mask_indexes[0], mask_indexes[1]])

        embeddings = torch.cat(embeddings, dim=0) # (bsz, D)
        # Each input sequence must contain a single [MASK] token.
        assert embeddings.shape[0] == yids_tensor.shape[0]

        return embeddings, yids_tensor, y_tensor

    def _process_item(self, item):
        X = item['question']['stem']
        y_list = [c['text'] for c in item['question']['choices']]
        y = ord(item['answerKey']) - ord('A')
        return {"X": X, "y_list": y_list, "y": y}

    def _process_mlm_task(self, task, split):
        """Preprocess for a MLM task given a data split and save to disk.
        
        For MLM tasks, feed the questions into a LM and get the contextual embeddings
          of the [MLM] tokens, which will be used to classify against the choices.
          Note that each choice must NOT be splitted into multiple word pieces by
          the self.bert_tokenizer. Otherwise we will remove them.

        Args:
            task ([type]): [description]
            split ([type]): [description]
        """

        output_path = Path(OLMPICS_DATA_PATH, f"{task}_{split}_preprocessed.pt")
        if self.skip_existing and output_path.exists():
            print(f'Found preprocessed file at {output_path.stem}. Skipping it.')
            return 

        with open(Path(OLMPICS_DATA_PATH, f"{task}_{split}.jsonl"), "r") as f:
            L = f.readlines()
            items = [json.loads(line) for line in L]
            task_data = [self._process_item(item) for item in items]

            unique_num_labels = set([len(d['y_list']) for d in task_data])
            assert len(unique_num_labels) == 1, "Every item in an oLMpics task should have the same number of classes."

            x_data = [d['X'] for d in task_data]
            ylist_data = [d['y_list'] for d in task_data]
            y_data = [d['y'] for d in task_data]
            embs, yids, y = self.extract_mask_emb_and_y_ids(x_data, ylist_data, y_data)

        torch.save({
            'embs': embs,
            'yids': yids,
            'y': y,
            'task_type': 'mlm',
            'vocab_size': self.bert_tokenizer.vocab_size
            }, output_path)
        print(f'Saved to {output_path.stem}.')
        
    def _process_qa_task(self, task, split):
        """Preprocess for a QA task given a data split and save to disk.

        For QA tasks, concatenate each question and its choices, and use the 
          contextual embedding of the [CLS] token as the sentence pair embedding.
          The sentence pair embeddings will be used to calculate the logit of each
          choice within each question.

        Input:
            task: task name.
            split: data split (train or dev).
        Return:
            task_data: list of {'X': str, 'y': int}
            nclasses: int
        """

        output_path = Path(OLMPICS_DATA_PATH, f"{task}_{split}_preprocessed.pt")
        if self.skip_existing and output_path.exists():
            print(f'Found preprocessed file at {output_path.stem}. Skipping it.')
            return 

        with open(Path(OLMPICS_DATA_PATH, f"{task}_{split}.jsonl"), "r") as f:
            L = f.readlines()
            items = [json.loads(line) for line in L]
            task_data = [self._process_item(item) for item in items]

            unique_num_labels = set([len(d['y_list']) for d in task_data])
            assert len(unique_num_labels) == 1, "Every item in an oLMpics task should have the same number of classes."

            x_data = [d['X'] for d in task_data]
            ylist_data = [d['y_list'] for d in task_data]
            embs = self.extract_emb_x_y(x_data, ylist_data) # (nexamples, nclasses, D)
            y = torch.tensor([d['y'] for d in task_data])

        torch.save({
            'embs': embs,
            'y': y,
            'task_type': 'qa'
            }, output_path)
        print(f'Saved to {output_path.stem}.')

def main():
    '''
    processor = oLMpicsPreprocessor(skip_existing=False)
    processor.process()

    processor = CATSPreprocessor(skip_existing=True)
    processor.process()
    '''

    processor = SentEvalPreprocessor(skip_existing=True)
    print("Preprocessing with BERT")
    processor.preprocess_all_w_bert()
    print("Preprocessing with InferSent")
    processor.preprocess_all_w_infersent()
    print("Preprocessing with Sentence Transformer")
    processor.preprocess_all_w_sbert()
    print("Preprocessing with Glove")
    processor.preprocess_all_w_glove()

if __name__ == '__main__':
    main()