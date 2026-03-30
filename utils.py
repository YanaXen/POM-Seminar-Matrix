from collections import defaultdict
def merge_intervals(intervals):
    """
    Merges overlapping or adjacent half-open intervals.

    Args:
        intervals (list): List of intervals in the form (start, end).

    Returns:
        merged_intervals (list): List of merged intervals in the form (start, end).
    """

    # Return an empty list if no intervals are given
    if not intervals:
        return []

    # Sort intervals by their start value
    intervals = sorted(intervals, key=lambda x: x[0])

    # Initialize the merged list with the first interval
    merged = [list(intervals[0])]

    # Process the remaining intervals one by one
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:  # overlap/adjacent
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    
    # Convert merged intervals back to tuples before returning
    return [(s, e) for s, e in merged]

def usage(schedule, makespan, R=None):
    """
    Computes machine utilization and downtime information from a schedule.

    Args:
        schedule (list): Scheduled operations in the form
            (machine, start, duration, ...).
        makespan (float): Total schedule length.
        R (int, optional): Total number of machines to report. If None, only
            machines that appear in the schedule are considered.

    Returns:
        machine_usage (list): Per-machine usage as tuples
            (machine, busy_time, relative_load).
        downtime_per_machine (list): Idle period lengths for each machine.
        downtime_over_makespan (list): Stepwise downtime series for each machine.
    """
    # Collect all non-zero processing intervals for each machine
    machine_times = defaultdict(list)

    # schedule entries have the form: (machine, start, dur, c, j)
    for r, start, dur, *_ in schedule:
        end = start + dur
        # ignore zero-length intervals for stability
        if end > start:
            machine_times[r].append((start, end))

    # Decide which machines should be included in the result
    machines = list(range(R)) if R is not None else sorted(machine_times.keys())

    machine_usage = []
    downtime_per_machine = []
    downtime_over_makespan = []

    # Evaluate usage and downtime for each machine separately
    for r in machines:
        intervals = machine_times.get(r, [])
        merged = merge_intervals(intervals)

        # busy time = union length
        busy_total = sum(e - s for s, e in merged)
        busy_total = max(0.0, min(busy_total, makespan))  # clamp safety
        idle_total = makespan - busy_total if makespan > 0 else 0.0
        rel_load = (busy_total / makespan) if makespan > 0 else 0.0
        machine_usage.append((r, busy_total, rel_load))

        # Build idle intervals as gaps between merged busy intervals
        gaps = []
        prev = 0.0
        for s, e in merged:
            if s > prev:
                gaps.append((prev, s))
            prev = max(prev, e)
        # Add trailing idle time after the last busy interval    
        if prev < makespan:
            gaps.append((prev, makespan))

        # Store downtime lengths for this machine
        downtimes = [b - a for a, b in gaps]
        downtime_per_machine.append(downtimes)

        # Create a stepwise downtime series
        series = [(a, dt) for (a, _), dt in zip(gaps, downtimes)]
        # If there is no downtime series, provide a fallback value
        if not series:
            series = [(0.0, makespan)] if makespan > 0 else [(0.0, 0.0)]
        downtime_over_makespan.append(series)

    return machine_usage, downtime_per_machine, downtime_over_makespan


def extract_schedule(S, Y, duration, type_of, PRODUCTS, J, R, T, eps=0.5):
    """
    Extracts a concrete schedule from the solved model variables.

    Args:
        S (gurobipy.tupledict): Binary start variables for product, operation,
            and time.
        Y (gurobipy.tupledict): Binary machine assignment variables for product,
            operation, and resource.
        duration (dict): Processing times per variant and operation.
        type_of (dict): Mapping from product index to variant index.
        PRODUCTS (int): Number of products.
        J (int): Number of operations.
        R (int): Number of resources.
        T (int): Time horizon.
        eps (float, optional): Threshold for interpreting binary decision
            variables as active. Defaults to 0.5.

    Returns:
        sched (list): Schedule entries in the form
            (machine, start, duration, product, operation).
    """
    # Determine the selected start time for each product-operation pair
    start = {}
    for c in range(PRODUCTS):
        for j in range(J):
            for t in range(T):
                if S[c, j, t].X > eps:
                    start[(c, j)] = t
                    break
            # Every product-operation pair must have exactly one start time
            if (c, j) not in start:
                raise RuntimeError(f"No start found for c={c}, j={j}")

     # Determine the assigned machine for each product-operation pair
    mach = {}
    for c in range(PRODUCTS):
        for j in range(J):
            for r in range(R):
                if Y[c, j, r].X > eps:
                    mach[(c, j)] = r
                    break
            # Every product-operation pair must be assigned to one machine
            if (c, j) not in mach:
                raise RuntimeError(f"No machine found for c={c}, j={j}")

    # Build the final schedule as a list of machine, start time, duration,
    # product index, and operation index
    sched = []
    for c in range(PRODUCTS):
        v = type_of[c]
        for j in range(J):
            dur = duration[v][j]
            sched.append((mach[(c, j)], start[(c, j)], dur, c, j))
    return sched


def active_interval_utilization(schedule, R=None, skip_first_last=True):
    """
    Computes utilization per machine over its active time window.

    Args:
        schedule (list): Scheduled operations in the form
            (machine, start, duration, ...).
        R (int, optional): Total number of machines to report. If None, the
            machine set is inferred from the schedule.
        skip_first_last (bool, optional): If True, skips the first and last
            machine index. Defaults to True.

    Returns:
        active_usage (list): Per-machine active utilization as tuples
            (machine, busy_time, active_window, utilization, first_start, last_end).
    """
    # Store busy time as well as the first start and last end time per machine
    machine_stats = defaultdict(lambda: {"busy": 0.0, "first": None, "last": None})

    # Collect statistics for all positive-duration operations
    for r, start, dur, *_ in schedule:
        if dur is None or dur <= 0:
            continue
        end = start + dur
        st = machine_stats[r]
        # Sum busy time and update the active time window
        st["busy"] += float(dur)
        st["first"] = start if st["first"] is None else min(st["first"], start)
        st["last"]  = end   if st["last"]  is None else max(st["last"],  end)

    # Return an empty list if no machine has positive-duration work
    if not machine_stats:
        return []

    # Determine the machine set to evaluate
    if R is not None:
        all_machines = list(range(R))
        first_machine, last_machine = 0, R - 1
    else:
        all_machines = sorted(machine_stats.keys())
        first_machine, last_machine = min(all_machines), max(all_machines)

    # Optionally exclude the first and last machine
    skip_set = set()
    if skip_first_last:
        skip_set.update([first_machine, last_machine])

    active_usage = []

    # Compute active-window utilization for each relevant machine
    for r in all_machines:
        if r in skip_set:
            continue
        # Skip machines without positive-duration jobs
        if r not in machine_stats:
            continue

        busy = machine_stats[r]["busy"]
        first = machine_stats[r]["first"]
        last = machine_stats[r]["last"]

        # The active window spans from the first start to the last end
        active_window = float(last - first) if (first is not None and last is not None) else 0.0
        
        # Compute utilization within the active window
        util_active = (busy / active_window) if active_window > 0 else 0.0
        active_usage.append((r, busy, active_window, util_active, first, last))

    # Return results sorted by machine index
    return sorted(active_usage, key=lambda x: x[0])


def build_greedy_warmstart(PRODUCTS, J, R, T, duration, type_of, predecessors, allowed_machines):
    """
    Builds a greedy warm-start solution for start times and machine assignments.

    Args:
        PRODUCTS (int): Number of products.
        J (int): Number of operations.
        R (int): Number of resources.
        T (int): Time horizon.
        duration (dict): Processing times per variant and operation.
        type_of (dict): Mapping from product index to variant index.
        predecessors (list): Precedence relations for each operation.
        allowed_machines (dict): Allowed machines for each variant and operation.

    Returns:
        start_time (dict): Start time for each product-operation pair.
        machine (dict): Assigned machine for each product-operation pair.

    Raises:
        ValueError: If the generated warm start exceeds the time horizon.
    """
    # Track when each machine becomes available again
    mach_ready = [0] * R

    # Track when each product becomes available for its next operation 
    product_ready = [0] * PRODUCTS

    # Store end times of already scheduled operations for precedence checks
    end_time = {} 

    start_time = {}
    machine = {}

    # Schedule operations product by product and operation by operation
    for c in range(PRODUCTS):
        v = type_of[c]
        for j in range(J):
            # Determine when all predecessors of the current operation are finished
            preds = predecessors[j]
            pred_ready = 0
            if preds:
                pred_ready = max(end_time[(c, h)] for h in preds)
            
            # The operation can only start when both the product and its
            # predecessors are ready.
            base_ready = max(product_ready[c], pred_ready)

            best_r = None
            best_start = None

            # Select the allowed machine that yields the earliest start time
            for r in allowed_machines[(v, j)]:
                s = max(base_ready, mach_ready[r])
                if (best_start is None) or (s < best_start) or (s == best_start and mach_ready[r] < mach_ready[best_r]):
                    best_start = s
                    best_r = r

            # Compute start and end time of the selected assignment
            dur = duration[v][j]
            s = int(best_start)
            e = s + int(dur)

            # Ensure the warm start remains within the model horizon
            if e > T:
                raise ValueError(
                    f"Warmstart exceeds horizon: c={c}, j={j}, start={s}, end={e}, T={T}. "
                    "Increase T or use a safer upper bound."
                )

            # Store the selected start time and machine
            start_time[(c, j)] = s
            machine[(c, j)] = best_r
            end_time[(c, j)] = e

            # Update availability of the product and the chosen machine
            product_ready[c] = e
            mach_ready[best_r] = e

    return start_time, machine


def apply_mip_start(S, Y, start_time, machine, PRODUCTS, J, R, T):
    """
    Writes a warm-start solution into the Gurobi variables.

    Args:
        S (gurobipy.tupledict): Binary start variables for product, operation,
            and time.
        Y (gurobipy.tupledict): Binary machine assignment variables for product,
            operation, and resource.
        start_time (dict): Start time for each product-operation pair.
        machine (dict): Assigned machine for each product-operation pair.
        PRODUCTS (int): Number of products.
        J (int): Number of operations.
        R (int): Number of resources.
        T (int): Time horizon.

    Returns:
        None: The warm-start values are written directly into the Gurobi variables.
    """
    # Initialize all start values with 0
    for c in range(PRODUCTS):
        for j in range(J):
            for t in range(T):
                S[c, j, t].Start = 0
            for r in range(R):
                Y[c, j, r].Start = 0

    # Set the selected start time and machine assignment for each operation
    for c in range(PRODUCTS):
        for j in range(J):
            t0 = start_time[(c, j)]
            r0 = machine[(c, j)]
            S[c, j, t0].Start = 1
            Y[c, j, r0].Start = 1