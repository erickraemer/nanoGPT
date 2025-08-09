import subprocess
from dataclasses import dataclass
from typing import Self, Final

NVIDIA_SMI_CMD: Final[list[str]] = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.free,utilization.gpu,gpu_name",
    "--format=csv,noheader,nounits"
]

@dataclass(frozen=True)
class NvidiaGPU:
    id: int
    name: str
    total_memory: int
    free_memory: int
    util: int

    @classmethod
    def from_csv(cls, line: str) -> Self:
        data = line.split(", ")
        
        gpu = cls(
            int(data[0]),
            str(data[4]),
            int(data[1]),
            int(data[2]),
            int(data[3])
        )

        return gpu

    def rank(self) -> float:
        return ((100.0 / self.total_memory * (self.total_memory - self.free_memory)) + self.util) // 2

    def __lt__(self, other: Self):
        return self.rank() < other.rank()

    def __repr__(self):
        return (
            f"{self.id} {self.name},"
            f" total_memory={self.total_memory} MiB,"
            f" free_memory={self.free_memory} MiB,"
            f" util={self.util}%"
        )

def get_gpus() -> list[NvidiaGPU]:
    lines: list[str] = subprocess.check_output(NVIDIA_SMI_CMD, text=True).split('\n')

    gpus: list[NvidiaGPU] = [
        NvidiaGPU.from_csv(line)
        for line
        in lines
        if line
    ]

    return gpus


if __name__ == "__main__":
    print("\n".join(str(gpu) for gpu in get_gpus()))