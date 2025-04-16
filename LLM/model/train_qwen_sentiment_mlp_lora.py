# 文件路径：LLM/model/train_qwen_sentiment_lora_mlp.py

import os
import sys
import torch
import wandb
import evaluate
import numpy as np
from transformers import AutoConfig
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from qwen_with_mlp import QwenWithMLPForSequenceClassification
# ========== 加载你的数据 ========== #
rough_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rough_data"))
sys.path.append(rough_data_dir)

from data_preparation_sen import tokenizer, tokenized_dataset, label2id, id2label

# 如果还没划分 train/test
if "train" in tokenized_dataset:
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]
else:
    dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
# Add this after loading your dataset
print("Dataset features:", train_dataset.features)
print("Sample entry:", train_dataset[0])
# ========== 导入自定义模型 ========== #

from qwen_with_mlp import QwenWithMLPForSequenceClassification

BASE_MODEL = "Qwen/Qwen1.5-0.5B"
LORA_PATH = "/media/mldadmin/home/s124mdg32_08/MotionBERT-main/LLM/model/qwen_sentiment_lora_finetuned"

model = QwenWithMLPForSequenceClassification(
    base_model_name=BASE_MODEL,
    lora_model_path=LORA_PATH,
    hidden_dim=512,
    dropout=0.1
)

tokenizer.pad_token = tokenizer.eos_token
model.model.config.pad_token_id = tokenizer.pad_token_id

# ========== 设置训练参数 ========== #
training_args = TrainingArguments(
    output_dir="./qwen_sentiment_lora_mlp",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
)

# ========== 定义评估指标 ========== #
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="macro")
    acc = accuracy_score(p.label_ids, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ========== 初始化 Trainer ========== #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ========== 启动训练 ========== #
trainer.train()
