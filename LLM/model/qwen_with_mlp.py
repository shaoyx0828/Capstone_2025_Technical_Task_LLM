import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

class QwenWithMLPForSequenceClassification(nn.Module):
    def __init__(self, base_model_name, lora_model_path, hidden_dim=512, dropout=0.1):
        super().__init__()

        # 加载 LoRA adapter 配置
        peft_config = PeftConfig.from_pretrained(lora_model_path)

        # 配置设置
        config = AutoConfig.from_pretrained(
            base_model_name,
            num_labels=3,  # 🔥 一定要和训练时 LoRA 保持一致
            trust_remote_code=True
        )

        # 加载带分类头的基础模型（支持 labels）
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            config=config,
            trust_remote_code=True
        )

        # 应用 LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, lora_model_path)

        # MLP 分类头
        self.dropout = nn.Dropout(dropout)
        self.mlp_head = nn.Sequential(
            nn.Linear(config.num_labels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, config.num_labels)
        )

        # 设置 pad_token_id，避免 Qwen 报错
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )

    # 取分类 logits（[batch_size, num_labels]）
        x = outputs.logits  # [B, num_labels]，比如 [16, 3]
        x = self.dropout(x)  # 可加可不加
        logits = self.mlp_head(x)  # 进入你自定义的 MLP 分类头

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }