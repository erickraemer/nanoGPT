from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

import wandb
from matplotlib.ticker import MaxNLocator

api = wandb.Api()

cache_dir = Path("wandb_run_cache")
cache_dir.mkdir(exist_ok=True)

run = api.run("adaptive-multi-head-attention/num_head_difference_pretrain/ov4nk2db")

start = 0
stop = 100000

plt.rcParams.update({'font.size': 20})

# create plot with x axis as iteration and y axis as mean validation loss, with one line per group
# plt.figure(figsize=(16, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

ax1.set_xscale('log')
ax2.set_xscale('log')

history = run.history(x_axis="iter")
loss = history[["loss/val", "iter"]].dropna()
num_heads = history[["n_active_heads", "iter"]].dropna()

ema = pd.Series(loss["loss/val"])
ema = ema.ewm(alpha=0.1, adjust=False).mean()
ax1.plot(loss["iter"], ema, label="1 Head")

ax1.set_ylabel("Validation Loss")
ax1.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax1.axvline(x=1000, color='red', linestyle='--', linewidth=1.5)

full_iters = pd.DataFrame({"iter": range(int(num_heads["iter"].min()), int(num_heads["iter"].max()) + 1, 1)})

# Merge and forward-fill missing values
num_heads_padded = full_iters.merge(num_heads, on="iter", how="left")
num_heads_padded["n_active_heads"] = num_heads_padded["n_active_heads"].ffill()
ax2.plot(num_heads_padded["iter"], num_heads_padded["n_active_heads"], label="1 Head")

ax2.set_ylabel("Active Attention Heads")
ax2.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.axvline(x=1000, color='red', linestyle='--', linewidth=1.5)
#ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

fig.supxlabel("Iteration")

# plt.grid()
plt.tight_layout()
plt.savefig(f"out/{Path(__file__).stem}.png")
plt.show()
