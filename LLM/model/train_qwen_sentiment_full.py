import os
import sys
import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ========== 1. 修正路径并导入数据 ==========
rough_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rough_data"))
sys.path.append(rough_data_dir)

from data_preparation_sen import label2id, id2label, tokenizer, tokenized_dataset

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# ========== 2. 加载模型 ==========
MODEL_NAME = "Qwen/Qwen1.5-0.5B"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ========== 3. 定义评估指标 ==========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # wandb logging (optional)
    wandb.log({
        "eval/accuracy": acc,
        "eval/precision": precision,
        "eval/recall": recall,
        "eval/f1": f1
    })

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ========== 4. 训练参数 ==========
training_args = TrainingArguments(
    output_dir="./qwen_sentiment_full",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./qwen_sentiment_full/logs",
    logging_strategy="steps",
    logging_steps=20,
    save_total_limit=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="wandb",
    fp16=True,
)

# ========== 5. Trainer 初始化 ==========
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ========== 6. 启动训练 ==========
if __name__ == "__main__":
    wandb.init(project="qwen-sentiment-full", name="full-finetuning")
    trainer.train()
    trainer.save_model("./qwen_sentiment_full/final")
    tokenizer.save_pretrained("./qwen_sentiment_full/final")
    wandb.finish()