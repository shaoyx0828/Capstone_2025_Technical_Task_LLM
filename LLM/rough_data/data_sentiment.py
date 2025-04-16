import pandas as pd
import ast

# ========= 1. 加载合并后的新闻数据 =========
df = pd.read_csv("merged_financial_news.csv")

# ========= 2. 提取调试用情感标签（便于分析分布） =========
def extract_sentiment_debug(entity_str):
    try:
        entity_obj = ast.literal_eval(entity_str)
        details = entity_obj.get("details", [])
        if details and "Sentiment" in details[0]:
            return details[0]["Sentiment"].lower()
        else:
            return "NO_SENTIMENT"
    except Exception:
        return "PARSE_ERROR"

df["sentiment_debug"] = df["entities"].apply(extract_sentiment_debug)

# ========= 3. 打印分布统计 =========
print("🔍 各种情感提取结果统计：")
print(df["sentiment_debug"].value_counts())

# ========= 4. 提取主情感标签（正式用于训练） =========
def extract_main_sentiment(entity_str):
    try:
        entity_obj = ast.literal_eval(entity_str)
        details = entity_obj.get("details", [])
        if details and "Sentiment" in details[0]:
            return details[0]["Sentiment"].lower()
    except Exception:
        return None

df["sentiment"] = df["entities"].apply(extract_main_sentiment)

# ========= 5. 构造有效情感训练集 =========
valid_labels = ["positive", "negative", "neutral"]
df_sentiment = df[["Content", "sentiment"]].dropna()
df_sentiment = df_sentiment[df_sentiment["sentiment"].isin(valid_labels)]

# 保存为主训练数据集
df_sentiment.to_csv("sentiment_classification_dataset.csv", index=False, encoding='utf-8-sig')
print(f"✅ 已生成情感分类数据集，共 {len(df_sentiment)} 条样本，保存为 sentiment_classification_dataset.csv")

# ========= 6. 保存无效样本（mixed, parse_error 等）供检查 =========
df_invalid = df[~df["sentiment_debug"].isin(valid_labels)]
df_invalid[["article_id", "sentiment_debug"]].to_csv("invalid_sentiment_articles.csv", index=False)
print(f"❗ 共发现 {len(df_invalid)} 条无效样本，已保存为 invalid_sentiment_articles.csv")
