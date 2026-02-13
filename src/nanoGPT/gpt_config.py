import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from omegaconf import OmegaConf


@dataclass
class WandbConfig:
    project_name: str = "nanoGPT"
    run_name: str = "gpt2"

@dataclass
class LoggingConfig:
    wandb: bool = True
    interval: int = 1000

@dataclass
class EvalConfig:
    only: bool = False
    interval: int = 1000
    iters: int = 200

@dataclass
class MetricConfig:
    head_dropout: bool = False
    attention_head_norms: bool = False
    projection_head_norms: bool = False
    attention_head_distribution: bool = False

@dataclass
class CheckpointingConfig:
    out_dir: str = "out"
    interval: int = 10_000

@dataclass
class DataConfig:
    path: str = ""
    dataset: str = "openwebtext"
    gradient_accumulation_steps: int = 5
    batch_size: int = 12
    block_size: int = 1024

@dataclass
class ModelConfig:
    seed: int = 1337
    checkpoint: str = ""
    layer: int = 12
    heads: int = 12
    embedding_size: int = 768
    vocab_size: int = 50304
    dropout_rate: float = 0.0
    bias: bool = False
    compile: bool = True
    dtype: str = "bfloat16"

@dataclass
class AdamWConfig:
    beta1: float = 0.9
    beta2: float = 0.95

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 6e-4
    max_iters: int = 600_000
    weight_decay: float = 1e-1
    grad_clip: float = 1.0

@dataclass
class LRSchedulerConfig:
    decay: bool = True
    warmup_iters: int = 2000
    decay_iters: int = 600_000
    min_lr: float = 6e-5

@dataclass
class DDPConfig:
    backend: str = "nccl"

@dataclass
class GPTConfig:
    wandb: WandbConfig
    logging: LoggingConfig
    eval: EvalConfig
    metrics: MetricConfig
    checkpointing: CheckpointingConfig
    data: DataConfig
    model: ModelConfig
    adamw: AdamWConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    head_activation_schedule: dict[int, int]
    ddp: DDPConfig
    flash: bool | None = None
    checkpoint_folder: Path | None = None

    @classmethod
    def load(cls, yaml_file: str) -> Self:
        config: GPTConfig = OmegaConf.load(yaml_file)

        assert config.model.embedding_size % config.model.heads == 0

        assert all(0 <= i <= config.optimizer.max_iters for i in config.head_activation_schedule.keys()), \
            "Iteration must be between 0 and max_iters"

        assert all(0 < i <= config.model.heads for i in config.head_activation_schedule.values()), \
            "Active heads must be between 1 and the total number of heads"

        config.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        config.checkpoint_folder = Path(config.checkpointing.out_dir) / config.wandb.run_name
        config.checkpoint_folder.mkdir(exist_ok=True, parents=True)
        shutil.copy(yaml_file, config.checkpoint_folder)

        return config

def __test():
    cfg: GPTConfig = GPTConfig.load("../../config/config.yaml")
    print()

if __name__ == "__main__":
    __test()