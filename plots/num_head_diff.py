from collections import defaultdict
from functools import reduce
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

import wandb

api = wandb.Api()

cache_dir = Path("wandb_run_cache")
cache_dir.mkdir(exist_ok=True)

run_names = [
    "adaptive-multi-head-attention/num_head_difference/qplhlwj0",
    "adaptive-multi-head-attention/num_head_difference/1p7h7l5r",
    "adaptive-multi-head-attention/num_head_difference/9vwfid8c",
    "adaptive-multi-head-attention/num_head_difference/kf7ea326",
    "adaptive-multi-head-attention/num_head_difference/tbkg01qy",
    "adaptive-multi-head-attention/num_head_difference/0ambnqgz",
    "adaptive-multi-head-attention/num_head_difference/itk5vc6t",
    "adaptive-multi-head-attention/num_head_difference/cs693mdt",
    "adaptive-multi-head-attention/num_head_difference/s4bs67ul",
    "adaptive-multi-head-attention/num_head_difference/zuctixqe",
    "adaptive-multi-head-attention/num_head_difference/htvbdfjb",
    "adaptive-multi-head-attention/num_head_difference/shw008bc",
    "adaptive-multi-head-attention/num_head_difference/ka8w65ev",
    "adaptive-multi-head-attention/num_head_difference/346v1sin",
    "adaptive-multi-head-attention/num_head_difference/y2f7fog3"
]

assert len(run_names) == 15, f"Expected 15 runs, got {len(run_names)}"

runs = [api.run(run_name) for run_name in run_names]

grouped = defaultdict(list)

for run in runs:
    grouped[int(run.config["head_activation_schedule"]["0"])].append(run)

for v in grouped.values():
    assert len(v) == 3, f"Expected 3 runs per group, got {len(v)}"


means = {}
for k, v in grouped.items():

    losses = []
    for i, run in enumerate(v):
        history = run.history(x_axis="iter")[["iter", "loss/val"]].dropna()
        history = history.rename(columns={"loss/val": f"loss_{i}"})
        losses.append(history)

    merged = reduce(lambda left, right: pd.merge(left, right, on="iter", how="outer"), losses)
    merged["mean_loss"] = merged[["loss_0", "loss_1", "loss_2"]].mean(axis=1)
    means[k] = merged[["iter", "mean_loss"]]

total_mean = {k: means[k].mean() for k in means}

plt.rcParams.update({'font.size': 20})

# create plot with x axis as iteration and y axis as mean validation loss, with one line per group
plt.figure(figsize=(16, 6))

for k, v in means.items():
    ema = pd.Series(v["mean_loss"].values)
    ema = ema.ewm(alpha=0.1, adjust=False).mean()
    plt.plot(v["iter"], ema, label=f"{k} Head" + ("s" if k > 1 else ""))

plt.xscale('log')
# Format x-axis with k notation
ax = plt.gca()
#ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

plt.xlabel("Iteration")
plt.ylabel("Mean Validation Loss")
# plt.title("Mean Validation Loss for Different Number of Active Heads")
plt.legend()
# plt.grid()
plt.tight_layout()
plt.savefig(f"out/{Path(__file__).stem}.png")
plt.show()

# # create plot with total mean validation loss for each group as a bar plot
# plt.figure(figsize=(10, 6))
# plt.plot(total_mean.keys(), total_mean.values())
# plt.xlabel("Number of Active Heads")
# plt.ylabel("Total Mean Validation Loss")
# plt.title("Total Mean Validation Loss for Different Number of Active Heads")
# #plt.grid()
# # plt.savefig("head_diff_bar_plot.png")
# plt.show()