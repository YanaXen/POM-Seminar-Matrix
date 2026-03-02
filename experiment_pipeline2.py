import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import pandas as pd
import numpy as np
import os

from experiment_config import experiment_configuration
from utils import extract_schedule_cars, usage, active_interval_utilization, build_greedy_warmstart, apply_mip_start
from diagram_generation import diagram_generation
from experiment_pipeline import run_model

def main():
    results = pd.DataFrame(columns=["experiment", "experiment_version", "makespan", "downtime_over_makespan", "downtime_per_machine", "usage", "active_usage", "total_work_content", "status", "runtime", "gap"])
    for experiment_index in range(0, 30):
        operations = [1, 2, 3, 4]
        ressources = [1,2,3,4,5,6,7,8]
        predecessors = [[], [0], [1], [2]]
        q = {0: 5, 1: 5, 2: 5}


        rng = np.random.default_rng(42 + experiment_index) 
        n_variants = 3
        low, high = 1, 6 

        mid = rng.integers(low, high, size=(n_variants, 2))
        duration = np.column_stack([np.zeros(n_variants, dtype=int), mid, np.zeros(n_variants, dtype=int)]).tolist()

        print(duration)

        for variant in ["No Flexibility", "Low Flexibility", "High Flexibility"]:
            if variant == "No Flexibility":
                variants = {0: [[0], [1], [4], [7]], 1: [[0], [2], [5], [7]], 2: [[0], [3], [6], [7]]}
            elif variant == "Low Flexibility":
                variants = {0: [[0], [1,2], [4,5], [7]], 1: [[0], [2,3], [5,6], [7]], 2: [[0], [1,3], [4,6], [7]]}
            elif variant == "High Flexibility":
                variants = {0: [[0], [1,2,3], [4,5,6], [7]], 1: [[0], [1,2,3], [4,5,6], [7]], 2: [[0], [1,2,3], [4,5,6], [7]]}

            makespan, downtime_per_machine, downtime_over_makespan, usage, active_usage, total_work_content, status, runtime, gap = run_model(operations, ressources, predecessors, duration, q, variants)

            row = {
                "experiment": f"Experiment {experiment_index}",
                "experiment_version": variant,
                "makespan": makespan,
                "downtime_over_makespan": downtime_over_makespan,
                "downtime_per_machine": downtime_per_machine,
                "usage": usage,
                "active_usage": active_usage,
                "total_work_content": total_work_content,
                "status": status,
                "runtime": runtime,
                "gap": gap
            }

            results.loc[len(results)] = row

            results.to_csv("data/experiment_results_backup_5.csv", index=False)

    results.to_csv("data/experiment_results_5.csv")

if __name__ == "__main__":
    main()