import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# === 模型路径和配置 ===
model_name_or_path = "qwen_sentiment_lora_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=3,
    trust_remote_code=True,
    id2label={0: "positive", 1: "negative", 2: "neutral"},
    label2id={"positive": 0, "negative": 1, "neutral": 2}
)
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# === CLI 模式：如果用户从命令行传入文本 ===
if len(sys.argv) > 1:
    input_text = " ".join(sys.argv[1:])
    result = sentiment_pipeline(input_text)[0]
    print(f"📝 Text: {input_text}")
    print(f"➡️ Sentiment: {result['label']} (Confidence: {result['score']:.2f})")

# === 默认样例模式：如果用户没传参数 ===
else:
    print("⚠️ No input provided. Running default sample examples...\n")
    sample_texts = [
        "Apple's stock surged after strong earnings report.",
        "这家公司因为丑闻导致股价下跌。",
        "目前大多数分析师对该公司持中立态度。"
    ]
    for text in sample_texts:
        result = sentiment_pipeline(text)[0]
        print(f"📝 Text: {text}")
        print(f"➡️ Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
        print("---")
