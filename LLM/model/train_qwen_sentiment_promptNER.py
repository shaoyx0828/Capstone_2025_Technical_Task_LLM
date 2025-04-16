# -*- coding: utf-8 -*-
"""
高效批量情感分类 + 评估脚本（优化版，支持 wandb）：
- 使用 tokenizer + model 推理替代 pipeline（更快）
- tqdm 显示实时进度条
- 支持 max_samples 限制样本数
- 收集并保存错误样本
- 输出 wandb 记录（准确率 / F1 / per-label 分数）
"""

import json
import wandb
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# === 初始化 wandb 项目 ===
wandb.init(project="promptner-sentiment-eval", name="inference-eval-run")

# === 加载模型 ===
model_path = "/media/mldadmin/home/s124mdg32_08/MotionBERT-main/LLM/model/qwen_sentiment_lora_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=3,
    trust_remote_code=True,
    id2label={0: "positive", 1: "negative", 2: "neutral"},
    label2id={"positive": 0, "negative": 1, "neutral": 2}
).cuda().eval()

# === 构造输入文本 ===
def build_sentiment_query(entity: str, context: str) -> str:
    return f"What is the sentiment toward \"{entity}\" in the following article?\n\n{context.strip()}"

# === 主分类流程（支持 max_samples） ===
def batch_entity_sentiment_classify(input_path: str, output_path: str, error_path: str, max_samples: int = None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    output = []
    error_samples = []
    y_true, y_pred = [], []

    for entry in tqdm(data, desc="🔍 Predicting..."):
        text = entry.get("context", "")
        entity = entry.get("entity", "")
        label = entry.get("label", "")
        query = build_sentiment_query(entity, text)

        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_id = logits.argmax(dim=-1).item()

        pred_label = model.config.id2label[pred_id]
        y_true.append(label)
        y_pred.append(pred_label)

        if pred_label != label:
            error_samples.append({"entity": entity, "context": text, "true": label, "pred": pred_label})

        output.append({
            "entity": entity,
            "context": text,
            "true_label": label,
            "predicted": pred_label,
            "score": round(torch.nn.functional.softmax(logits, dim=-1)[0][pred_id].item(), 4)
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    with open(error_path, "w", encoding="utf-8") as f:
        for item in error_samples:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Saved {len(output)} predictions → {output_path}")
    print(f"❌ Misclassified samples: {len(error_samples)} → {error_path}")

    # === 输出评估 ===
    print("\n📊 Evaluation Report:")
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    print(classification_report(y_true, y_pred, digits=4))
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # === wandb 记录 ===
    wandb.log({"accuracy": acc})
    for label in ["positive", "negative", "neutral"]:
        if label in report:
            wandb.log({
                f"precision_{label}": report[label]["precision"],
                f"recall_{label}": report[label]["recall"],
                f"f1_{label}": report[label]["f1-score"]
            })
    wandb.finish()

# === 执行入口 ===
if __name__ == "__main__":
    input_json = "/media/mldadmin/home/s124mdg32_08/MotionBERT-main/LLM/rough_data/entity_sentiment_dataset.json"
    output_jsonl = "entity_sentiment_predictions.jsonl"
    error_output = "misclassified_samples.jsonl"

    # 可选：限制处理数量（加速测试）
    max_eval_samples = None  # 改为 100 试运行

    batch_entity_sentiment_classify(input_json, output_jsonl, error_output, max_samples=max_eval_samples)
