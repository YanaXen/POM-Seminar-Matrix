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

def build_durations(n_variants, rng, low=4, high=7):
    """
    Generates durations for 3/4/5/6 layers while keeping work content constant.
    First and last duration are always 0.

    - 3 layers: [0, D, 0]
    - 4 layers: [0, D1, D2, 0]
    - 5 layers: split D2 -> [0, D1, a, b, 0]
    - 6 layers: split D1 -> [0, c, d, a, b, 0]
    """
    # D must be splittable into D1 + D2, and later each of those may be split again
    if low < 4:
        raise ValueError("Use low >= 4 so durations can be split into positive parts.")

    base_total = rng.integers(low, high, size=n_variants)

    dur3 = []
    dur4 = []
    dur5 = []
    dur6 = []

    for v in range(n_variants):
        TOTAL = int(base_total[v])

        # split total into two positive parts for 4-layer version
        D1, D2 = split_int(TOTAL, rng, min_part=2)

        # 3 layers
        dur3_v = [0, TOTAL, 0]

        # 4 layers
        dur4_v = [0, D1, D2, 0]

        # 5 layers: split D2
        a, b = split_int(D2, rng, min_part=1)
        dur5_v = [0, D1, a, b, 0]

        # 6 layers: split D1
        c, d = split_int(D1, rng, min_part=1)
        dur6_v = [0, c, d, a, b, 0]

        dur3.append(dur3_v)
        dur4.append(dur4_v)
        dur5.append(dur5_v)
        dur6.append(dur6_v)

    return dur3, dur4, dur5, dur6

# def build_durations(n_variants, rng, low=2, high=7):
#     """
#     Generates durations for 4/5/6 layers while keeping work content constant.
#     - 4 layers: [0, d1, d2, 0]
#     - 5 layers: split d2 -> [0, d1, a, b, 0]
#     - 6 layers: split d1 -> [0, c, d, a, b, 0]
#     """
#     # Generate 4-layer base (ensure we can split later => choose low>=2)
#     # high is exclusive like numpy
#     d1 = rng.integers(low, high, size=n_variants)
#     d2 = rng.integers(low, high, size=n_variants)

#     dur4 = []
#     dur5 = []
#     dur6 = []

#     for v in range(n_variants):
#         D1 = int(d1[v])
#         D2 = int(d2[v])

#         # 4 layers
#         dur4_v = [0, D1, D2, 0]

#         # 5 layers: split layer 3 (D2)
#         a, b = split_int(D2, rng, min_part=1)
#         dur5_v = [0, D1, a, b, 0]

#         # 6 layers: split layer 2 (D1) into c,d
#         c, d = split_int(D1, rng, min_part=1)
#         dur6_v = [0, c, d, a, b, 0]

#         dur4.append(dur4_v)
#         dur5.append(dur5_v)
#         dur6.append(dur6_v)

#     return dur4, dur5, dur6

def build_q_random(total_cars, n_variants, rng, min_per_variant=1):
    """
    Randomly distributes total_cars across n_variants.
    Ensures each variant gets at least min_per_variant cars.
    """
    if total_cars < n_variants * min_per_variant:
        raise ValueError("total_cars is too small for the chosen min_per_variant")

    remaining = total_cars - n_variants * min_per_variant

    # random split of remaining cars
    cuts = sorted(rng.integers(0, remaining + 1, size=n_variants - 1))
    parts = np.diff([0] + cuts + [remaining])

    q = {v: int(parts[v] + min_per_variant) for v in range(n_variants)}
    return q

# Experiment 1
def get_experiment_one_config(rng, total_cars, n_variants, variant_index, dur4):
    operations = [1, 2, 3, 4]
    ressources = [1,2,3,4,5,6,7,8]
    predecessors = [[], [0], [1], [2]]

    duration = dur4

    variants_list = ["No Flexibility", "Low Flexibility", "High Flexibility"]
    variant = variants_list[variant_index]

    if variant == "No Flexibility":
        variants = {0: [[0], [1], [4], [7]], 1: [[0], [2], [5], [7]], 2: [[0], [3], [6], [7]]}
    elif variant == "Low Flexibility":
        variants = {0: [[0], [1,2], [4,5], [7]], 1: [[0], [2,3], [5,6], [7]], 2: [[0], [1,3], [4,6], [7]]}
    elif variant == "High Flexibility":
        variants = {0: [[0], [1,2,3], [4,5,6], [7]], 1: [[0], [1,2,3], [4,5,6], [7]], 2: [[0], [1,2,3], [4,5,6], [7]]}
    return variant, operations, ressources, predecessors, duration, variants

# Experiment 2
def get_experiment_two_config(rng, total_cars, n_variants, variant_index, dur4, dur5, dur6):
    ressources = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    variants_list = ["4 Layers", "5 Layers", "6 Layers"]
    variant = variants_list[variant_index]

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
    return variant, operations, ressources, predecessors, duration, variants


# Experiment 3
def get_experiment_three_config(rng, total_cars, n_variants, variant_index, dur3, dur4, dur5, dur6):

    variants_list = ["3 Layers", "4 Layers", "5 Layers", "6 Layers"]
    variant = variants_list[variant_index]

    if variant == "3 Layers":
        operations = [1, 2, 3]
        ressources = [1,2,3,4,5]
        predecessors = [[], [0], [1]]
        variants = {0: [[0], [1,2,3], [4]], 1: [[0], [1,2,3], [4]], 2: [[0], [1,2,3], [4]]}
        duration = dur3

    elif variant == "4 Layers":
        operations = [1, 2, 3, 4]
        ressources = [1,2,3,4,5,6,7,8]
        predecessors = [[], [0], [1], [2]]
        variants = {0: [[0], [1,2], [4,5], [7]], 1: [[0], [2,3], [5,6], [7]], 2: [[0], [1,3], [4,6], [7]]}
        duration = dur4

    elif variant == "5 Layers":
        operations = [1, 2, 3, 4, 5]
        ressources = [1,2,3,4,5,6,7,8,9,10,11]
        predecessors = [[], [0], [1], [2], [3]]
        variants = {0: [[0], [1,2], [4,5], [7,8], [10]], 1: [[0], [2,3], [5,6], [8,9], [10]], 2: [[0], [1,3], [4,6], [7,9], [10]]}
        duration = dur5

    elif variant == "6 Layers":
        operations = [1, 2, 3, 4, 5, 6]
        ressources = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        predecessors = [[], [0], [1], [2], [3], [4]]
        variants = {0: [[0], [1,2], [4,5], [7,8], [10,11], [13]], 1: [[0], [2,3], [5,6], [8,9], [11,12], [13]], 2: [[0], [1,3], [4,6], [7,9], [10,12], [13]]}
        duration = dur6

    return variant, operations, ressources, predecessors, duration, variants

def main():
    results = pd.DataFrame(columns=["experiment", "experiment_version", "makespan", "downtime_over_makespan", "downtime_per_machine", "usage", "active_usage", "total_work_content", "status", "runtime", "gap"])
    total_cars = 15
    n_variants = 3

    for experiment in ["Experiment 1", "Experiment 2", "Experiment 3"]:

        for experiment_index in range(0, 30):
            rng = np.random.default_rng(42 + experiment_index)

            if (experiment == "Experiment 1" or experiment == "Experiment 2"):
                variant_indexes = range(3)
            else:
                variant_indexes = range(4)

            q = build_q_random(total_cars, n_variants, rng)

            dur3, dur4, dur5, dur6 = build_durations(n_variants=n_variants, rng=rng, low=4, high=8)

            for variant_index in variant_indexes:

                if experiment == "Experiment 1":
                    variant, operations, ressources, predecessors, duration, variants = get_experiment_one_config(rng, total_cars, n_variants, variant_index, dur4)
                elif experiment == "Experiment 2":
                    variant, operations, ressources, predecessors, duration, variants = get_experiment_two_config(rng, total_cars, n_variants, variant_index, dur4, dur5, dur6)
                elif experiment == "Experiment 3":
                    variant, operations, ressources, predecessors, duration, variants = get_experiment_three_config(rng, total_cars, n_variants, variant_index, dur3, dur4, dur5, dur6)

                makespan, downtime_per_machine, downtime_over_makespan, usage, active_usage, total_work_content, status, runtime, gap = run_model(operations, ressources, predecessors, duration, q, variants)

                row = {
                    "experiment": experiment,
                    "experiment_version": variant,
                    "experiment_index": experiment_index,
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

                results.to_csv("data/results_backup.csv", index=False)

    results.to_csv("data/results.csv")

if __name__ == "__main__":
    main()