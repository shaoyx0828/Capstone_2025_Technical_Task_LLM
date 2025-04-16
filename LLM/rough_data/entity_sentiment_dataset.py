# -*- coding: utf-8 -*-
"""
生成实体级情感分类数据集：
- 输入：原始金融事件标注数据（如 Financial News Labels.json）
- 输出：结构化数据 entity_sentiment_dataset.json，用于微调或预测
"""

import json
from collections import Counter
from pathlib import Path

# === 实体级数据提取函数 ===
def prepare_entity_sentiment_dataset(data: dict, max_samples: int = None):
    valid_sentiments = {"positive", "negative", "neutral"}
    dataset = []

    for idx, (doc_id, content) in enumerate(data.items()):
        if max_samples and idx >= max_samples:
            break

        event_summaries = [e["summary"] for e in content.get("events", {}).get("details", [])]
        base_context = " ".join(event_summaries).strip()

        for entity in content.get("entities", {}).get("details", []):
            sentiment = entity.get("Sentiment", "").lower()
            if sentiment not in valid_sentiments:
                continue

            entity_name = entity.get("Entity_name", "").strip()
            entity_summary = entity.get("Summary", "").strip()
            if not entity_name or not entity_summary:
                continue

            full_context = f"{base_context} {entity_summary}".strip()
            dataset.append({
                "entity": entity_name,
                "context": full_context,
                "label": sentiment
            })

    return dataset

# === 主程序入口 ===
if __name__ == "__main__":
    input_path = "Financial News Labels.json"  # ← 原始标注数据（json 格式）
    output_path = "entity_sentiment_dataset.json"

    print("🔄 Loading raw annotated data...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("⚙️  Extracting structured entity-level sentiment samples...")
    dataset = prepare_entity_sentiment_dataset(raw_data)

    print(f"✅ Total samples: {len(dataset)}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"📦 Saved to {output_path}")
