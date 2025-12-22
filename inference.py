import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import Dataset
import sklearn
import numpy as np
import pandas as pd
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel


class Config:
    gemma_dir = 'unsloth/gemma-2-9b-it-bnb-4bit'
    lora_dir = './checkpoint-pretrain'
    max_length = 3000
    batch_size = 4
    device = torch.device("cuda")

cfg = Config()
test = pd.read_csv("./data/test.csv")

def process_text(text: str) -> str:
    text = text.replace("\\/", "/")
    return " ".join(eval(text, {"null": ""}))

test.loc[:, 'prompt'] = test['prompt'].apply(process_text)
test.loc[:, 'response_a'] = test['response_a'].apply(process_text)
test.loc[:, 'response_b'] = test['response_b'].apply(process_text)

def tokenize(
    tokenizer, prompt, response_a, response_b
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]

    prompt = tokenizer(prompt, max_length=800, truncation=True, padding=False).input_ids
    response_a = tokenizer(response_a, max_length=1000, truncation=True, padding=False).input_ids
    response_b = tokenizer(response_b, max_length=1000, truncation=True, padding=False).input_ids
    input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
    attention_mask = [[1]* len(i) for i in input_ids]

    return input_ids, attention_mask


tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

data = pd.DataFrame()
data["id"] = test["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test["prompt"], test["response_a"], test["response_b"])
data["length"] = data["input_ids"].apply(len)

aug_data = pd.DataFrame()
aug_data["id"] = test["id"]
# swap response_a & response_b
aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test["prompt"], test["response_b"], test["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)

device = torch.device('cuda:0')
model = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device,
    use_cache=False,
    num_labels=3,
)
model = PeftModel.from_pretrained(model, cfg.lora_dir)

with torch.no_grad() and torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
        a_win, b_win, tie = [], [], []
        
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            tmp = df.iloc[start_idx:end_idx]
            input_ids = tmp["input_ids"].to_list()
            attention_mask = tmp["attention_mask"].to_list()
            inputs = pad_without_fast_tokenizer_warning(
                tokenizer,
                {"input_ids": input_ids, "attention_mask": attention_mask},
                padding="longest",
                pad_to_multiple_of=None,
                return_tensors="pt",
            )
            outputs = model(**inputs.to(device))
            proba = outputs.logits.softmax(-1).cpu()
            
            a_win.extend(proba[:, 0].tolist())
            b_win.extend(proba[:, 1].tolist())
            tie.extend(proba[:, 2].tolist())
        
        df["winner_model_a"] = a_win
        df["winner_model_b"] = b_win
        df["winner_tie"] = tie
        
        return df
    
    results = inference(aug_data, model, device)
    print(results)
    proba = results[["winner_model_a", "winner_model_b", "winner_tie"]].values
    
    submission_df = results[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
    submission_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")