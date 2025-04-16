import wandb
import torch
torch.cuda.empty_cache()

def init_wandb_config(notes=None):
    config = wandb.config  # 从 sweep 中自动获取参数

    model_name = config.model_name if hasattr(config, "model_name") else "Qwen/Qwen1.5-0.5B"
    task_name = config.task_name if hasattr(config, "task_name") else "SentimentClassification"
    run_version = config.run_version if hasattr(config, "run_version") else "v1"

    run_name = f"{task_name}-{model_name.split('/')[-1]}-{run_version}"
    
    wandb.init(
        project="qwen-sentiment",
        name=run_name,
        group=task_name,
        config={ 
            "batch_size": 16,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.2,
            "lr": 5e-5,
            "num_epochs": 10,
            "metric_for_best_model": "accuracy"

        },
        notes=notes,
    )
