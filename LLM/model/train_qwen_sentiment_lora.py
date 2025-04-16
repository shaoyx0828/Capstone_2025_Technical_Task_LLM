import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 可选：指定使用哪张 GPU

from wandb_settings import init_wandb_config

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import EarlyStoppingCallback

def main():
    torch.cuda.empty_cache()

    # ✅ 初始化 wandb 并设置最佳超参数
    wandb.init(
        config={
            "batch_size": 16,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.2,
            "lr": 5e-5,
            "num_epochs": 30,
            "metric_for_best_model": "accuracy"
        }
    )
    config = wandb.config
    init_wandb_config()

    # ========= 1. 加载数据 =========
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "rough_data", "sentiment_classification_dataset.csv")
    dataset_path = os.path.abspath(dataset_path)
    dataset = load_dataset("csv", data_files=dataset_path)
    label2id = {"positive": 0, "negative": 1, "neutral": 2}
    id2label = {v: k for k, v in label2id.items()}

    def map_labels(example):
        example["label"] = label2id[example["sentiment"]]
        return example

    dataset = dataset.map(map_labels).remove_columns(["sentiment"])
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    # ========= 2. 分词 =========
    model_name = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        return tokenizer(example["Content"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    # ========= 3. 模型 + LoRA =========
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        trust_remote_code=True,
        return_dict=True,
        id2label=id2label,
        label2id=label2id,
        pad_token_id=tokenizer.pad_token_id
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(base_model, lora_config)

    # ========= 4. 评估函数 =========
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "precision": precision.compute(predictions=preds, references=labels, average="macro")["precision"],
            "recall": recall.compute(predictions=preds, references=labels, average="macro")["recall"],
            "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # ========= 5. Trainer 设置 =========
    training_args = TrainingArguments(
        output_dir="./qwen_sentiment_lora",
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.lr,
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        report_to="wandb",
        fp16=True,
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # ========= 6. 启动训练 =========
    trainer.train()

    # ========= 7. 保存模型 =========
    trainer.save_model("qwen_sentiment_lora_finetuned")

    # ✅ log 最终评估结果
    wandb.log({
        "final_eval_accuracy": trainer.evaluate()["eval_accuracy"],
        "final_eval_f1": trainer.evaluate()["eval_f1"]
    })

    print("✅ 模型微调完成并已保存：qwen_sentiment_lora_finetuned")






# ========= ✅ 设置入口 =========
if __name__ == "__main__":
    main()
