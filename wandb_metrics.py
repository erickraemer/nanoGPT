import numpy as np
import tabulate
from pandas import DataFrame
import pandas as pd

import wandb

api = wandb.Api()

# Get specific run
run = api.run("adaptive-multi-head-attention/adaptive-multi-head-attention/khwzav1o")
# print(run.summary)   Final metrics
# print(run.config)    Configuration

layer: int = run.config["model"]["layer"]

# Get run history
history = run.history(x_axis="iter")  # Returns pandas DataFrame

num_values: int = 0
first_num_values: int = 0
true_positives: int = 0
k_positives: int = 0
first_true_positives: int = 0
total_loss = np.zeros(3)
policy_loss = np.zeros(3)

for l in range(layer):

    column_names = [f"gradient_norm/layer{l:02}/c_proj/head{i:02}" for i in range(4, 8)]
    column_names.append("iter")
    c_proj: DataFrame = history[column_names]
    c_proj = c_proj[c_proj["iter"] <= 8000]
    c_proj = c_proj.dropna()

    c_proj_8k = c_proj[c_proj["iter"] == 8000]
    c_proj_8k = c_proj_8k[[col for col in c_proj_8k.columns if col != 'iter']]
    c_proj_8k = c_proj_8k.T
    c_proj_8k.sort_values(by=c_proj_8k.columns[0], ascending=False, inplace=True)

    head_columns = [col for col in c_proj.columns if col != 'iter']
    means = c_proj[head_columns].mean()
    means.sort_values(ascending=False, inplace=True)

    column_names = [f"head_importance/layer{l:02}/head{i:02}" for i in range(4, 8)]
    column_names.append("iter")
    head_imp: DataFrame = history[column_names]
    head_imp = head_imp[(head_imp["iter"] > 8000) & (head_imp["iter"] < 100000)]
    head_imp = head_imp.dropna()

    head_columns = [col for col in head_imp.columns if col != 'iter']
    means_imp = head_imp[head_columns].mean()
    means_imp.sort_values(ascending=False, inplace=True)

    loss = means_imp.iloc[0] - means_imp
    loss = loss.rename(lambda x: x.split("/")[-1])
    total_loss += loss.iloc[1:].to_numpy()
    loss_r = loss.round(4)

    avg_imp = means_imp.mean()

    better_than_avg = means_imp >= avg_imp
    better_than_avg = better_than_avg.rename(lambda x: x.split("/")[-1])

    means_axes = [s.split("/")[-1] for s in means.axes[0]]
    means_imp_axes = [s.split("/")[-1] for s in means_imp.axes[0]]
    c_proj_8k_axes = [s.split("/")[-1] for s in c_proj_8k.axes[0]]

    data = [t for t in zip(means_axes, means_imp_axes, c_proj_8k_axes)]

    for (a,b,c) in data:
        if a == b:
            true_positives += 1
        if b == c:
            k_positives += 1
        num_values += 1

    a,b,_ = data[0]
    if a == b:
        first_true_positives += 1
    first_num_values += 1

    headers = ["t. proj norm mean (0-8k)", "head importance mean (8k-100k)", "t. proj norm (8k)"]
    print(f"Layer {l:02}:")
    print(tabulate.tabulate(data, headers=headers, tablefmt="pretty"))
    print()
    loss_data = [t for t in zip(means_imp_axes, loss_r)]
    headers = ["head", "loss when picked"]
    print(tabulate.tabulate(loss_data, headers=headers, tablefmt="pretty"))
    print()
    print(f"Policy better than avg ({means_axes[0]}): {better_than_avg[means_axes[0]]}\n")
    print()

    policy_loss += loss[means_axes[0:3]].to_numpy()

print("Total results:\n---------------------")
print(f"Accuracy: {true_positives/num_values*100:.2f}%")
print(f"First Row Accuracy: {first_true_positives/first_num_values*100:.2f}%")
print(f"8k Accuracy: {k_positives/num_values*100:.2f}%")

total_loss = total_loss.cumsum()
policy_loss = policy_loss.cumsum()
policy_loss = policy_loss/total_loss*100
data = [(i+1, f"{policy_loss[i]:.2f}%") for i in range(0, 3)]
headers = ["picks", "loss to optimal policy"]
print(tabulate.tabulate(data, headers=headers, tablefmt="pretty"))