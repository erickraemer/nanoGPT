import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Final

import numpy as np
import tabulate

NVIDIA_SMI_VARS_CMD: Final[list[str]] = [
    "nvidia-smi",
    "--query-gpu=index,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits"
]

NVIDIA_SMI_CONSTANTS_CMD: Final[list[str]] = [
    "nvidia-smi",
    "--query-gpu=index,gpu_name,memory.total",
    "--format=csv,noheader,nounits"
]

@dataclass(frozen=True)
class NvidiaGPU:
    id: int
    name: str
    total_memory: int
    free_memory: int
    util: float

    def rank(self) -> float:
        return ((100.0 / self.total_memory * (self.total_memory - self.free_memory)) + self.util) // 2

    def __repr__(self):
        used_memory: float = self.total_memory - self.free_memory
        return (
            f"{self.id}, {self.name},"
            f" Memory: {used_memory}/{self.total_memory} MiB ({round(100.0 / self.total_memory * used_memory, 1)}%),"
            f" Utilization: {self.util}%,"
            f" Total Utilization: {int(self.rank())}%"
        )

def _get_nvidia_smi_cmd_output(cmd: list[str]) -> list[list[str]]:
    l = [
        line.split(", ")
        for line
        in subprocess.check_output(cmd, text=True).split('\n')
        if line
    ]

    return l

def get_gpus(n_samples: int = 5) -> list[NvidiaGPU]:
    constants: dict[int, tuple[str, int]] = {
        int(index): (name, int(mem))
        for index, name, mem in _get_nvidia_smi_cmd_output(NVIDIA_SMI_CONSTANTS_CMD)
    }

    # take n_samples samples of free memory and utilization
    print("Collecting GPU memory and utilization data...")
    vars_: defaultdict[int, list] = defaultdict(list)
    for i in range(n_samples):
        for index, fmem, util in _get_nvidia_smi_cmd_output(NVIDIA_SMI_VARS_CMD):
            vars_[int(index)].append([float(fmem), float(util)])
        time.sleep(0.7)

    assert constants.keys() == vars_.keys()

    gpus: list[NvidiaGPU] = []
    for k in constants.keys():
        mean = np.mean(np.array(vars_[k]), axis=0)
        gpus.append(
            NvidiaGPU(
                id=k,
                name=constants[k][0],
                total_memory=constants[k][1],
                free_memory=int(mean[0]),
                util=round(mean[1],1)
            )
        )

    return gpus

def print_gpus(gpus: list[NvidiaGPU]) -> None:
    data = [
        [
            gpu.id,
            gpu.name,
            gpu.total_memory,
            round(100.0/gpu.total_memory *(gpu.total_memory- gpu.free_memory), 1),
            gpu.util,
            gpu.rank()
        ]
        for gpu in sorted(gpus, key=lambda x: x.id)
    ]

    headers = ["ID", "Name", "Total Memory (MiB)", "Used Memory (%)", "GPU Util (%)", "Total Util (%)"]

    print(tabulate.tabulate(data, headers=headers, tablefmt="pretty"))


if __name__ == "__main__":
    #print("\n".join(str(gpu) for gpu in get_gpus()))
    gpus = get_gpus()
    print_gpus(gpus)