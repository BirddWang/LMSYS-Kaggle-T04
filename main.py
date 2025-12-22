import os
import copy
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from sklearn.metrics import log_loss, accuracy_score

@dataclass
class Config:
    lora_dir: str = "output-pretrain/checkpoint-5239"
    output_dir: str = "output-has-pretrain"
    checkpoint: str = "unsloth/gemma-2-9b-it-bnb-4bit"  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 3000
    n_splits: int = 100
    fold_idx: int = 0
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2  # global batch size is 8 
    per_device_eval_batch_size: int = 8
    n_epochs: int = 1
    freeze_layers: int = 16  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 2e-4
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"

config = Config()

training_args = TrainingArguments(
    output_dir=config.output_dir,
    logging_dir=os.path.join(config.output_dir, "logs"),
    overwrite_output_dir=True,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="steps",
    metric_for_best_model="log_loss",
    save_steps=500,
    optim=config.optim_type,
    fp16=False,
    bf16=True,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
)

lora_config = LoraConfig(
    use_dora=True,
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
)

tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True  # We'll add <eos> at the end
tokenizer.padding_side = "right"

# model = Gemma2ForSequenceClassification.from_pretrained(
#     config.checkpoint,
#     num_labels=3,
#     dtype=torch.float16,
#     device_map="auto",
# )
# model.config.use_cache = False
# model = prepare_model_for_kbit_training(model)
# # model = get_peft_model(model, lora_config)
# model = PeftModel(model, config.lora_dir)
# model.print_trainable_parameters()

# 2 options: load lora weights into base model, or create peft model and load weights
base_model = Gemma2ForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=3,
    dtype=torch.bfloat16,
    device_map="auto",
)
if os.path.exists(config.lora_dir):
    print("Loading LoRA weights from:", config.lora_dir)
    model = PeftModel.from_pretrained(
        base_model,
        config.lora_dir,
        device_map="auto",
        is_trainable=True,
    )
else:
    print("Preparing base model for k-bit training and adding LoRA adapters...")
    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

ds = Dataset.from_csv("./data/cleaned_pre_train.csv")

class CustomTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        labels = [label for label in batch["labels"]]
        # for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
        #     if a_win:
        #         label = 0
        #     elif b_win:
        #         label = 1
        #     else:
        #         label = 2
        #     labels.append(label)
        return {**tokenized, "labels": labels}
        
    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(text.split()) if text else ""


encode = CustomTokenizer(tokenizer, max_length=config.max_length)
ds = ds.map(encode, batched=True)

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}

folds = [
    (
        [i for i in range(len(ds)) if i % config.n_splits != fold_idx],
        [i for i in range(len(ds)) if i % config.n_splits == fold_idx]
    ) 
    for fold_idx in range(config.n_splits)
]


train_idx, eval_idx = folds[config.fold_idx]

trainer = Trainer(
    args=training_args, 
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds.select(train_idx),
    eval_dataset=ds.select(eval_idx),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()