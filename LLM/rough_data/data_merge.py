import json
import pandas as pd

# ========= 1. 加载原始新闻数据 =========
with open("Financial News.json", "r", encoding="utf-8") as f:
    news_data = json.load(f)

news_df = pd.DataFrame(news_data)
assert "article_id" in news_df.columns, "🚨 缺少 article_id 字段"

# ========= 2. 加载标签数据 =========
with open("Financial News Labels.json", "r", encoding="utf-8") as f:
    label_data = json.load(f)

# 转换为 DataFrame
label_list = []
for article_id, info in label_data.items():
    label_list.append({
        "article_id": article_id,
        "events": info.get("events", {}),
        "entities": info.get("entities", {})
    })
label_df = pd.DataFrame(label_list)

# ========= 3. 合并：只保留有标签的（inner join） =========
merged_df = pd.merge(news_df, label_df, on="article_id", how="inner")

# ========= 4. 保存合并成功的数据 =========
merged_df.to_csv("merged_financial_news.csv", index=False, encoding="utf-8-sig")
print(f"✅ 合并完成，共 {len(merged_df)} 条样本，已保存为 merged_financial_news.csv")

# ========= 5. 输出未被合并的原始新闻 ID =========
merged_ids = set(merged_df["article_id"])
all_ids = set(news_df["article_id"])
unmatched_ids = all_ids - merged_ids

unmatched_df = news_df[news_df["article_id"].isin(unmatched_ids)]
unmatched_df[["article_id", "Title", "type_news"]].to_csv("unmatched_articles.csv", index=False)

print(f"❗ 共发现 {len(unmatched_df)} 条新闻没有对应标签，已保存为 unmatched_articles.csv")
