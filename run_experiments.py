import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import pandas as pd
import numpy as np
import os

from utils import extract_schedule, usage, active_interval_utilization, build_greedy_warmstart, apply_mip_start
from experiment_pipeline import run_model


def split_int(x, rng, min_part=1):
    """
    Splits an integer into two parts whose sum equals the original value.

    Args:
        x (int): Integer to split.
        rng (numpy.random.Generator): Random number generator used for sampling.
        min_part (int, optional): Minimum value for each part. Defaults to 1.

    Returns:
        a (int): First part of the split.
        b (int): Second part of the split.
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
    Generates duration profiles for 3-, 4-, 5-, and 6-layer variants while
    keeping the total work content constant for each variant. The first and 
    last duration are always 0. Intermediate durations are split so that all
    layer configurations of the same variant have the same total work content.

    Args:
        n_variants (int): Number of variants to generate.
        rng (numpy.random.Generator): Random number generator used for sampling.
        low (int, optional): Lower bound for the total duration per variant.
            Defaults to 4.
        high (int, optional): Upper bound for the total duration per variant
            (exclusive). Defaults to 7.

    Returns:
        dur3 (list): Duration profiles for 3-layer variants.
        dur4 (list): Duration profiles for 4-layer variants.
        dur5 (list): Duration profiles for 5-layer variants.
        dur6 (list): Duration profiles for 6-layer variants.

    Raises:
        ValueError: If low is smaller than 4.

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

def build_q_random(total_products, n_variants, rng, min_per_variant=1):
    """
    Randomly distributes a total number of products across variants.

    Args:
        total_products (int): Total number of products to distribute.
        n_variants (int): Number of variants.
        rng (numpy.random.Generator): Random number generator used for sampling.
        min_per_variant (int, optional): Minimum number of products assigned to
            each variant. Defaults to 1.

    Returns:
        q (dict): Mapping from variant index to assigned product quantity.

    Raises:
        ValueError: If total_products is smaller than
            n_variants * min_per_variant.
    """
    if total_products < n_variants * min_per_variant:
        raise ValueError("total_products is too small for the chosen min_per_variant")

    remaining = total_products - n_variants * min_per_variant

    # random split of remaining products
    cuts = sorted(rng.integers(0, remaining + 1, size=n_variants - 1))
    parts = np.diff([0] + cuts + [remaining])

    q = {v: int(parts[v] + min_per_variant) for v in range(n_variants)}

    print(f"q: {q}")
    return q

# Experiment 1
def get_experiment_one_config(rng, total_products, n_variants, variant_index, dur4):
    """
    Returns the configuration for experiment 1 with different flexibility levels.

    Args:
        rng (numpy.random.Generator): Random number generator.
        total_products (int): Total number of products in the experiment.
        n_variants (int): Number of product variants.
        variant_index (int): Index of the flexibility setting to use.
        dur4 (list): Duration data for the 4-operation setting.

    Returns:
        variant (str): Name of the selected flexibility setting.
        operations (list): List of operations.
        ressources (list): List of available resources.
        predecessors (list): Precedence relations between operations.
        duration (list): Duration data used in the experiment.
        variants (dict): Allowed resources for each variant and operation.
    """
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
def get_experiment_two_config(rng, total_products, n_variants, variant_index, dur4, dur5, dur6):
    """
    Returns the configuration for experiment 2 with different layer counts.

    Args:
        rng (numpy.random.Generator): Random number generator.
        total_products (int): Total number of products in the experiment.
        n_variants (int): Number of product variants.
        variant_index (int): Index of the layer setting to use.
        dur4 (list): Duration data for the 4-layer setting.
        dur5 (list): Duration data for the 5-layer setting.
        dur6 (list): Duration data for the 6-layer setting.

    Returns:
        variant (str): Name of the selected layer setting.
        operations (list): List of operations.
        ressources (list): List of available resources.
        predecessors (list): Precedence relations between operations.
        duration (list): Duration data used in the experiment.
        variants (dict): Allowed resources for each variant and operation.
    """
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
def get_experiment_three_config(rng, total_products, n_variants, variant_index, dur3, dur4, dur5, dur6):
    """
    Returns the configuration for experiment 3 with different layer counts.

    Args:
        rng (numpy.random.Generator): Random number generator.
        total_products (int): Total number of products in the experiment.
        n_variants (int): Number of product variants.
        variant_index (int): Index of the layer setting to use.
        dur3 (list): Duration data for the 3-layer setting.
        dur4 (list): Duration data for the 4-layer setting.
        dur5 (list): Duration data for the 5-layer setting.
        dur6 (list): Duration data for the 6-layer setting.

    Returns:
        variant (str): Name of the selected layer setting.
        operations (list): List of operations.
        ressources (list): List of available resources.
        predecessors (list): Precedence relations between operations.
        duration (list): Duration data used in the experiment.
        variants (dict): Allowed resources for each variant and operation.
    """
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

# Experiment 4
def get_experiment_four_config(rng, n_variants, variant_index):
    """
    Returns the configuration for experiment 4 with different batch sizes.

    Args:
        rng (numpy.random.Generator): Random number generator.
        n_variants (int): Number of product variants.
        variant_index (int): Index of the batch size setting to use.

    Returns:
        q (dict): Mapping from variant index to assigned product quantity.
        variant (str): Name of the selected batch size setting.
    """
    variants_list = ["Batch Size = 15", "Batch Size = 30", "Batch Size = 45"]
    variant = variants_list[variant_index]

    if variant == "Batch Size = 15":
        total_products = 15
        q = build_q_random(total_products, n_variants, rng)

    elif variant == "Batch Size = 30":
        total_products = 30
        q = build_q_random(total_products, n_variants, rng)

    elif variant == "Batch Size = 45":
        total_products = 45
        q = build_q_random(total_products, n_variants, rng)


    return q, variant

def main(continue_run=False):
    """
    Runs the selected experiments, solves the corresponding scheduling models,
    and stores the results in CSV files.

    Args:
        continue_run (bool, optional): If True, resumes from an existing backup
            file and skips configurations that have already been processed.
            Defaults to False.

    Returns:
        None: The function writes intermediate and final results to CSV files.
    """
    # Create an empty results table with all output columns.
    results = pd.DataFrame(columns=["experiment", "experiment_version", "experiment_index", "makespan", "downtime_over_makespan", "downtime_per_machine", "usage", "active_usage", "total_work_content", "status", "runtime", "gap"])
    
    # Define the base problem size
    total_products = 15
    n_variants = 3

    # Resume from backup if requested and the backup file exists.
    if continue_run and os.path.exists("data/results_backup.csv"):
        results = pd.read_csv("data/results_backup.csv")

    # Build a set of already completed experiment combinations for exact resume
    if len(results) > 0:
        done = set(zip(
            results["experiment"].astype(str),
            results["experiment_version"].astype(str),
            results["experiment_index"].astype(int),
        ))
    else:
        done = set()
    
    # Select which experiments to run.
    for experiment in ["Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"]:
        
        # Repeat each experiment with different random seeds.
        for experiment_index in range(0, 30):
            rng = np.random.default_rng(42 + experiment_index)

            # Determine how many variant settings belong to the current experiment
            if (experiment == "Experiment 1" or experiment == "Experiment 2"):
                variant_indexes = range(3)
            elif (experiment == "Experiment 4"):
                variant_indexes = range(12)
            else:
                variant_indexes = range(4)

            # Generate random product quantities and duration profiles for this run
            q = build_q_random(total_products, n_variants, rng)
            dur3, dur4, dur5, dur6 = build_durations(n_variants=n_variants, rng=rng, low=4, high=8)

            # Iterate over all configurations of the selected experiment
            for variant_index in variant_indexes:
                
                # Load the configuration for the current experiment variant
                if experiment == "Experiment 1":
                    variant, operations, ressources, predecessors, duration, variants = get_experiment_one_config(rng, total_products, n_variants, variant_index, dur4)
                elif experiment == "Experiment 2":
                    variant, operations, ressources, predecessors, duration, variants = get_experiment_two_config(rng, total_products, n_variants, variant_index, dur4, dur5, dur6)
                elif experiment == "Experiment 3":
                    variant, operations, ressources, predecessors, duration, variants = get_experiment_three_config(rng, total_products, n_variants, variant_index, dur3, dur4, dur5, dur6)
                elif experiment == "Experiment 4":
                    # Experiment 4 combines layer configuration and batch size
                    variant_index_temp = variant_index//3
                    variant, operations, ressources, predecessors, duration, variants = get_experiment_three_config(rng, total_products, n_variants, variant_index_temp, dur3, dur4, dur5, dur6)
                    q, batch_variant = get_experiment_four_config(rng, n_variants, variant_index%3)
                    variant = f"{variant}-{batch_variant}"
                
                # Skip configurations that were already processed in a previous run
                key = (str(experiment), str(variant), int(experiment_index))
                if key in done:
                    continue
                
                # Solve the model for the current configuration
                makespan, downtime_per_machine, downtime_over_makespan, usage, active_usage, total_work_content, status, runtime, gap = run_model(operations, ressources, predecessors, duration, q, variants)

                # Store the result row for later analysis
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

                # Append the result, mark it as done, and write a backup file
                results.loc[len(results)] = row
                done.add(key)
                results.to_csv("data/results_backup.csv", index=False)
    
    # Write the final results file after all runs are complete
    results.to_csv("data/results2.csv")

if __name__ == "__main__":
    main(False)