from matplotlib import pyplot as plt
import pandas as pd
import ast
import numpy as np


def create_diagram(x, labels, xlabel, ylabel, name, flexibility):
    fig, ax = plt.subplots()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    bplot = ax.boxplot(x,  
                    labels=labels) 

    ax.set_title(name)
    if ylabel == "Utilization":
        ymax = 1
    else:
        ymax = np.nanmax([np.nanmax(np.asarray(group, dtype=float)) for group in x])
        ymax += 2
    ax.set_ylim(0, ymax)

    medians = []
    for group in x:
        arr = np.asarray(group, dtype=float)
        arr = arr[~np.isnan(arr)]
        med = np.median(arr) if arr.size else np.nan
        medians.append(med)

    print(ylabel)
    print(medians)

    fig.savefig(f"diagrams/diagram_boxplot_{flexibility}_{ylabel}.png")
    plt.close(fig)


def get_usage(series):
    avg_utils = []
    for run_usage in series:           
        rels = [u[2] for u in run_usage]
        if len(rels) > 2:
            rels = rels[1:-1]          
        avg_utils.append(sum(rels) / len(rels) if rels else None)
    return avg_utils


def diagram_generation(data, labels, flexibility):
    if len(labels)==3:
        label1, label2, label3 = labels
    else:
        label1, label2, label3, label4 = labels

    cols_to_parse = ["usage", "active_usage", "downtime_per_machine", "downtime_over_makespan"]

    for col in cols_to_parse:
        data[col] = data[col].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)

    values_1 = data[data["experiment_version"] == label1]["makespan"]
    values_2 = data[data["experiment_version"] == label2]["makespan"]
    values_3 = data[data["experiment_version"] == label3]["makespan"]
    if len(labels)>3:
        values_4 = data[data["experiment_version"] == label4]["makespan"]
        x = [values_1, values_2, values_3, values_4]
    else:
        x = [values_1, values_2, values_3]

    create_diagram(x, labels, f"{flexibility} Flexibility", "Makespan", f"Makespan comparison for {flexibility} Flexibility levels", flexibility)

    values_1 = data[data["experiment_version"] == label1]["usage"]
    values_2 = data[data["experiment_version"] == label2]["usage"]
    values_3 = data[data["experiment_version"] == label3]["usage"]

    values_1 = get_usage(values_1)
    values_2 = get_usage(values_2)
    values_3 = get_usage(values_3)
    if len(labels)>3:
        values_4 = data[data["experiment_version"] == label4]["usage"]
        values_4 = get_usage(values_4)
        x = [values_1, values_2, values_3, values_4]
    else:
        x = [values_1, values_2, values_3]
    create_diagram(x, labels, f"{flexibility} Flexibility", "Utilization", f"Utilization comparison for {flexibility} Flexibility levels", flexibility)


def run_diagram_creation(data):
    experiments = pd.unique(data["experiment"])
    for experiment in experiments:
        subset = data[data["experiment"] == experiment].copy()
        if "Experiment 1" in experiment:
            flexibility = "Resource"
            labels = ["No Flexibility", "Low Flexibility", "High Flexibility"]
        elif "Experiment 2" in experiment:
            flexibility = "Layout"
            labels = ["4 Layers", "5 Layers", "6 Layers"]
        elif "Experiment 3" in experiment:
            flexibility = "Line"
            labels = ["3 Layers", "4 Layers", "5 Layers", "6 Layers"]
        diagram_generation(subset, labels, flexibility)


if __name__ == "__main__":
    data = pd.read_csv("data/experiment_results_backup_6.csv")
    run_diagram_creation(data)