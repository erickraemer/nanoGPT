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
class CheckpointConfig:
    out_dir: str = "out"
    interval: int = 10_000

@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    gradient_accumulation_steps: int = 5
    batch_size: int = 12
    block_size: int = 1024

@dataclass
class ModelConfig:
    checkpoint: str = ""
    layer: int = 12
    heads: int = 12
    active_heads: int = 8
    head_activation_step: int = 600_000
    embedding_size: int = 768
    vocab_size: int = 50304
    dropout_rate: float = 0.0
    bias: bool = False
    compile: bool = True
    dtype: str = "bfloat16"

    def __setattr__(self, key, value):
        assert key != self.head_activation_step.__name__ or value <= self.heads
        super().__setattr__(key, value)

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
    checkpoint: CheckpointConfig
    data: DataConfig
    model: ModelConfig
    adamw: AdamWConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    ddp: DDPConfig
    flash: bool | None = None
    checkpoint_folder: Path | None = None

    def __post_init__(self):
        assert self.model.embedding_size % self.model.heads == 0
        assert self.model.active_heads <= self.model.heads

    @classmethod
    def load(cls, yaml_file: str) -> Self:
        config = OmegaConf.load(yaml_file)
        config.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        config.checkpoint_folder = Path(config.checkpoint.out_dir) / config.wandb.run_name
        config.checkpoint_folder.mkdir(exist_ok=True, parents=True)
        shutil.copy(yaml_file, config.checkpoint_folder)

        return config

def __test():
    cfg: GPTConfig = GPTConfig.load("../../config/config.yaml")
    print()

if __name__ == "__main__":
    __test()