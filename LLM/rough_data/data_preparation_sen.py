from datasets import load_dataset
from transformers import AutoTokenizer
import os

# ========== 1. 加载数据 ========== #
csv_path = os.path.join(os.path.dirname(__file__), "sentiment_classification_dataset.csv")
dataset = load_dataset("csv", data_files=csv_path)["train"]  # 注意加 ["train"]

# ========== 2. prompt 包装 + 标签映射 ========== #
label2id = {"positive": 0, "negative": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}

def format_prompt(example):
    content = example["Content"]
    example["Content"] = f"""Classify the sentiment of the following financial news article into one of the categories: positive, negative, or neutral.\n\nNews: "{content}" """
    return example

def map_labels(example):
    example["label"] = label2id[example["sentiment"]]
    return example

dataset = dataset.map(format_prompt)
dataset = dataset.map(map_labels)
dataset = dataset.remove_columns(["sentiment"])

# ========== 3. 分词器处理 ========== #
model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 768

def tokenize(example):
    return tokenizer(
        example["Content"],
        padding="max_length",
        truncation=True,
        max_length=768
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# ========== 4. 划分 train / test ========== #
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

# ========== 5. 转为 torch 格式（配合 Trainer） ========== #
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ========== 6. 输出样本检查 ========== #
print(tokenized_dataset)
print(tokenized_dataset["train"][0])
