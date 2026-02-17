from pathlib import Path

import numpy as np
import torch

from nanoGPT.gpt_config import GPTConfig


class DataLoader:
    def __init__(self, cfg: GPTConfig, device: str):
        data_path: Path = Path(cfg.data.path) / cfg.data.dataset
        self._train_dir: Path = data_path / 'train.bin'
        self._val_dir: Path = data_path / 'val.bin'
        self._batch_size: int = cfg.data.batch_size
        self._block_size: int = cfg.data.block_size
        self._device: str = device
        self._cuda: bool = device.startswith('cuda')

    def get_batch(self, split: str):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(self._train_dir, dtype=np.uint16, mode='r')
        else:
            data = np.memmap(self._val_dir, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self._block_size, (self._batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self._block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self._block_size]).astype(np.int64)) for i in ix])
        if self._cuda:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self._device, non_blocking=True), y.pin_memory().to(self._device, non_blocking=True)
        else:
            x, y = x.to(self._device), y.to(self._device)
        return x, y
