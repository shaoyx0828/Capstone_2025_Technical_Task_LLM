# Financial Sentiment Classification using Lightweight LLMs

This repository contains a complete fine-tuning and evaluation pipeline for financial sentiment classification using open-source lightweight language models, such as Qwen1.5-0.5B. The project implements multiple fine-tuning strategies including:

- Full-parameter finetuning
- LoRA (Low-Rank Adaptation)
- Prefix Tuning
- LoRA + MLP classifier
- PromptNER-style entity-level sentiment classification

## Dataset

The dataset consists of curated financial news articles labeled with sentiment (positive, negative, neutral). Raw data is provided in:
- `financial_news.json`
- `financial_news_label.json`

After preprocessing and filtering, the dataset is saved as:
- `merged_financial_news.csv`
- `sentiment_classification_dataset.csv`

## Features

- Modular training pipeline using HuggingFace Transformers and PEFT
- Support for full finetuning, LoRA, Prefix tuning, and hybrid heads
- Prompt-based inference for entity sentiment using Qwen-style prompts
- Real-time CLI sentiment classification
- Weights & Biases logging integration
- Misclassification logging and error analysis

## Environment Setup

```bash
conda create -n sentiment_env python=3.10
conda activate sentiment_env
pip install -r requirements.txt
```

## Usage

### 1. Preprocess the data

```bash
python llm/rough_data/data_merge.py
python llm/rough_data/data_pre_sen_prompt.py
```

### 2. Finetune the model (LoRA example)

```bash
python train_qwen_sentiment_lora.py
```

### 3. Run inference (CLI)

```bash
python predict_qwen_sentiment_lora.py "未来市场走势仍存在巨大不确定性"
```

### 4. Entity-level classification with PromptNER

```bash
python eval_entity_prompt_sentiment.py
```

### 5. Evaluate LoRA + MLP

```bash
python train_qwen_sentiment_lora_mlp.py
```

## Results Summary

| Method                 | Accuracy | F1 Score | Notes                          |
|------------------------|----------|----------|--------------------------------|
| Full Finetuning        | 79.49%   | 75.86%   | Best overall performance       |
| LoRA                   | 78.34%   | 73.30%   | Efficient and robust           |
| Prefix Tuning          | 78.34%   | 72.67%   | Lower recall on neutral class  |
| LoRA + MLP             | 74.11%   | 68.38%   | Slight gain, less stable       |
| PromptNER (entity-level)| 62.55%  | 58.45% (macro) | Entity-level classification |

## Project Structure

```
.
├── LLM/
│   ├── model/                         
│   ├── rough_data/                   
│   ├── train_qwen_sentiment_lora.py  
│   ├── train_qwen_sentiment_full.py  
│   ├── train_qwen_sentiment_lora_mlp.py
│   ├── eval_entity_prompt_sentiment.py 
│   └── predict_qwen_sentiment_lora.py 
├── sentiment_classification_dataset.csv
├── merged_financial_news.csv
├── README.md
```


