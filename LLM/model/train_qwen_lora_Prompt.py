# -*- coding: utf-8 -*-
"""
双阶段信息抽取流程：
1. 使用 Qwen1.5-0.5B 执行 PromptNER 实体识别
2. 使用已微调的 qwen_sentiment_lora_finetuned 模型进行情感分类
输入：新闻文本 + prompt 模板
输出：结构化实体 + 情感 + 解释
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
import re
import json

# ========== 模型加载 ==========
# 1. 实体识别模型（原始 Qwen）
promptner_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
promptner_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
promptner_pipe = pipeline("text-generation", model=promptner_model, tokenizer=promptner_tokenizer)

# 2. 情感分类模型（用户微调后的LoRA模型）
sentiment_model_path = "qwen_sentiment_lora_finetuned"

sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path, trust_remote_code=True)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    sentiment_model_path,
    num_labels=3,
    trust_remote_code=True,
    id2label={0: "positive", 1: "negative", 2: "neutral"},
    label2id={"positive": 0, "negative": 1, "neutral": 2}
)
sentiment_pipe = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer)

# ========== PromptNER 模板构造 ==========
def build_prompt(text):
    return f"""
Definition:
- Company: An organization that appears in a financial or political context.
- People: Individuals involved in an event.
- Region: A country or location mentioned in the context.

Example:
Text: "Google announced a billion-dollar investment in India."
Entity: "Google"
IsEntity: Yes
Type: Company
Reason: A tech company directly involved in the action.

Entity: "India"
IsEntity: Yes
Type: Region
Reason: Location of the investment.

Now analyze:
Text: "{text}"
List all potential entities in the same format.
""".strip()

# ========== 解析 PromptNER 输出 ==========
def parse_promptner_output(output_text):
    pattern = r'Entity:\s*\"(.*?)\"\nIsEntity:\s*(Yes|No)\nType:\s*(.*?)\nReason:\s*(.*?)\n'
    matches = re.findall(pattern, output_text, re.DOTALL)
    results = []
    for m in matches:
        if m[1].strip().lower() == 'yes':
            results.append({
                "entity": m[0].strip(),
                "type": m[2].strip(),
                "reason": m[3].strip()
            })
    return results

# ========== 情感分类输入构造 ==========
def build_sentiment_query(entity, text):
    return f"What is the sentiment toward '{entity}' in the following text?\n\n{text}"

# ========== 主流程函数 ==========
def process_article(text):
    # Step 1: 实体识别
    prompt = build_prompt(text)
    response = promptner_pipe(prompt, max_new_tokens=300, do_sample=False)[0]['generated_text']
    entities = parse_promptner_output(response)

    # Step 2: 情感分类
    for item in entities:
        query = build_sentiment_query(item['entity'], text)
        sentiment = sentiment_pipe(query)[0]
        item['sentiment'] = sentiment['label']
        item['score'] = round(sentiment['score'], 4)

    return {
        "text": text,
        "entities": entities
    }

# ========== 示例使用 ==========
if __name__ == "__main__":
    article = "Google invests billions in South Africa's cloud infrastructure."
    result = process_article(article)
    print(json.dumps(result, indent=2, ensure_ascii=False))
