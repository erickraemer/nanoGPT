from pathlib import Path

import numpy as np
import torch

from nanoGPT.gpt_config import GPTConfig


class DataLoader:
    def __init__(self, cfg: GPTConfig, device: str, train: bool):
        self._batch_size: int = cfg.data.batch_size
        self._block_size: int = cfg.data.block_size
        self._device: str = device
        self._data_path: Path = Path(cfg.data.path) / cfg.data.dataset / ('train.bin' if train else 'val.bin')

    def __iter__(self):
        return DataLoaderIterator(self)

    def __len__(self):
        data_length: int = len(np.memmap(self._data_path, dtype=np.uint16, mode='r'))
        return (data_length - self._block_size) // self._batch_size

    # def get_batch(self, split: str):
    #     # We recreate np.memmap every batch to avoid a memory leak, as per
    #     # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    #     if split == 'train':
    #         data = np.memmap(self._train_dir, dtype=np.uint16, mode='r')
    #     else:
    #         data = np.memmap(self._val_dir, dtype=np.uint16, mode='r')
    #     ix = torch.randint(len(data) - self._block_size, (self._batch_size,))
    #     x = torch.stack([torch.from_numpy((data[i:i+self._block_size]).astype(np.int64)) for i in ix])
    #     y = torch.stack([torch.from_numpy((data[i+1:i+1+self._block_size]).astype(np.int64)) for i in ix])
    #     if self._cuda:
    #         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #         x, y = x.pin_memory().to(self._device, non_blocking=True), y.pin_memory().to(self._device, non_blocking=True)
    #     else:
    #         x, y = x.to(self._device), y.to(self._device)
    #     return x, y

class DataLoaderIterator:
    def __init__(self, data_loader: DataLoader):
        self._block_size: int = data_loader._block_size
        self._batch_size: int = data_loader._batch_size
        self._device: str = data_loader._device
        self._data_path: Path = data_loader._data_path
        self._idx: int = 0

    def __iter__(self):
        return self

    def __next__(self):

        data = np.memmap(self._data_path, dtype=np.uint16, mode='r')

        start: int = self._idx * self._batch_size
        data_length: int = self._batch_size + self._block_size

        # skip last batch if it is too small
        if start + data_length > len(data):
            self._idx = 0
            start = 0

        self._idx += 1

        array = torch.from_numpy(data[start:start + data_length].copy())
        array = array.to(torch.int64)
        array = array.unfold(0, self._block_size, 1)

        x = array[:-1].contiguous().pin_memory().to(self._device, non_blocking=True)
        y = array[1:].contiguous().pin_memory().to(self._device, non_blocking=True)

        return x, y

# def __debug():
#     from nanoGPT.gpt_config import DataConfig
#     torch.cuda.set_device("cuda:0")
#     cfg = GPTConfig(
#         data=DataConfig(
#             path='/home/user/Documents/nanoGPT/data',
#             dataset='openwebtext',
#             batch_size=12,
#             block_size=512,
#         )
#     )
#     dl = DataLoader(cfg, device='cuda', True)
#     for i, (x, y) in enumerate(dl):
#         print(x.shape, y.shape)
#         if i > 5:
#             break
#
# if __name__ == '__main__':
#     __debug()