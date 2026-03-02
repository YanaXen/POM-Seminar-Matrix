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

def split_int(x, rng, min_part=1):
    """
    Split integer x into (a,b) with a+b=x and a,b >= min_part.
    Works for x >= 2*min_part.
    """
    if x < 2 * min_part:
        # fallback: allow zero parts if needed, but better avoid by choosing bounds
        a = x // 2
        b = x - a
        return a, b
    a = rng.integers(min_part, x - min_part + 1)  # inclusive upper via +1
    b = x - a
    return int(a), int(b)

def build_durations_4_5_6(n_variants, rng, low=2, high=7):
    """
    Generates durations for 4/5/6 layers while keeping work content constant.
    - 4 layers: [0, d1, d2, 0]
    - 5 layers: split d2 -> [0, d1, a, b, 0]
    - 6 layers: split d1 -> [0, c, d, a, b, 0]
    """
    # Generate 4-layer base (ensure we can split later => choose low>=2)
    # high is exclusive like numpy
    d1 = rng.integers(low, high, size=n_variants)
    d2 = rng.integers(low, high, size=n_variants)

    dur4 = []
    dur5 = []
    dur6 = []

    for v in range(n_variants):
        D1 = int(d1[v])
        D2 = int(d2[v])

        # 4 layers
        dur4_v = [0, D1, D2, 0]

        # 5 layers: split layer 3 (D2)
        a, b = split_int(D2, rng, min_part=1)
        dur5_v = [0, D1, a, b, 0]

        # 6 layers: split layer 2 (D1) into c,d
        c, d = split_int(D1, rng, min_part=1)
        dur6_v = [0, c, d, a, b, 0]

        dur4.append(dur4_v)
        dur5.append(dur5_v)
        dur6.append(dur6_v)

    return dur4, dur5, dur6


def main():
    results = pd.DataFrame(columns=["experiment", "experiment_version", "makespan", "downtime_over_makespan", "downtime_per_machine", "usage", "active_usage", "total_work_content", "status", "runtime", "gap"])
    for experiment_index in range(0, 30):
        
        ressources = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        q = {0: 5, 1: 5, 2: 5}

        rng = np.random.default_rng(42 + experiment_index)
        dur4, dur5, dur6 = build_durations_4_5_6(n_variants=3, rng=rng, low=3, high=8)

        for variant in ["4 Layers", "5 Layers", "6 Layers"]:

            if variant == "4 Layers":
                operations = [1, 2, 3, 4]
                predecessors = [[], [0], [1], [2]]
                variants = {0: [[0], [1,2,3,4,5,6], [7,8,9,10,11,12], [13]], 1: [[0], [1,2,3,4,5,6], [7,8,9,10,11,12], [13]], 2: [[0], [1,2,3,4,5,6], [7,8,9,10,11,12], [13]]}
                duration = dur4

            elif variant == "5 Layers":
                operations = [1, 2, 3, 4, 5]
                predecessors = [[], [0], [1], [2], [3]]
                variants = {0: [[0], [1,2,3,4], [5,6,7,8], [9,10,11,12], [13]], 1: [[0], [1,2,3,4], [5,6,7,8], [9,10,11,12], [13]], 2: [[0], [1,2,3,4], [5,6,7,8], [9,10,11,12], [13]]}
                duration = dur5

            elif variant == "6 Layers":
                operations = [1, 2, 3, 4, 5, 6]
                predecessors = [[], [0], [1], [2], [3], [4]]
                variants = {0: [[0], [1,2,3], [4,5,6], [7,8,9], [10,11,12], [13]], 1: [[0], [1,2,3], [4,5,6], [7,8,9], [10,11,12], [13]], 2: [[0], [1,2,3], [4,5,6], [7,8,9], [10,11,12], [13]]}
                duration = dur6

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

            results.to_csv("data/experiment_results_backup_6.csv", index=False)

    results.to_csv("data/experiment_results_6.csv")

if __name__ == "__main__":
    main()