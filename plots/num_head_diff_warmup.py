from collections import defaultdict
from functools import reduce
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

import wandb
from matplotlib.ticker import LogLocator, NullLocator

api = wandb.Api()

cache_dir = Path("wandb_run_cache")
cache_dir.mkdir(exist_ok=True)

project = api.runs("adaptive-multi-head-attention/num_head_difference_pretrain")

runs = [run for run in project]

grouped = defaultdict(list)

for run in runs:
    grouped[int(run.config["head_activation_schedule"].get("1000", 16))].append(run)

#for v in grouped.values():
#    assert len(v) == 3, f"Expected 3 runs per group, got {len(v)}"

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

part_start = 10000
part_stop = 100000
part_means = {}
for k, v in grouped.items():

    losses = []
    for i, run in enumerate(v):
        history = run.history(x_axis="iter")[["iter", "loss/val"]].dropna()
        history = history[(history["iter"] > part_start) & (history["iter"] <= part_stop)]
        history = history.rename(columns={"loss/val": f"loss_{i}"})
        losses.append(history)

    merged = reduce(lambda left, right: pd.merge(left, right, on="iter", how="outer"), losses)
    merged["mean_loss"] = merged[["loss_0", "loss_1", "loss_2"]].mean(axis=1)
    part_means[k] = merged[["iter", "mean_loss"]]

total_mean = {k: means[k]["mean_loss"].mean() for k in means}

plt.rcParams.update({'font.size': 20})

# create plot with x axis as iteration and y axis as mean validation loss, with one line per group
# plt.figure(figsize=(16, 6))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')

for k, v in means.items():
    ema = pd.Series(v["mean_loss"].values)
    ema = ema.ewm(alpha=0.1, adjust=False).mean()
    ax1.plot(v["iter"], ema, label=f"{k} Head" + ("s" if k > 1 else ""))

ax1.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Mean Validation Loss")
ax1.legend()

for k, v in part_means.items():
    ema = pd.Series(v["mean_loss"].values)
    ema = ema.ewm(alpha=0.3, adjust=False).mean()
    ax2.plot(v["iter"], ema, label=f"{k} Head" + ("s" if k > 1 else ""))

#ax2.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
#ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Mean Validation Loss")
# ax2.xaxis.set_major_locator(LogLocator(base=10, numticks=2))
# ax2.xaxis.set_minor_locator(NullLocator())
ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=[0.5, 0.75], numticks=4))

ax3.plot(total_mean.keys(), total_mean.values())
ax3.set_xlabel("Number of Active Heads")
ax3.set_ylabel("Total Mean Validation Loss")
ax3.set_xticks(list(total_mean.keys()))
ax3.set_xticklabels([str(int(x)) for x in total_mean.keys()])

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