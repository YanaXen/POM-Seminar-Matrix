from collections import defaultdict
def merge_intervals(intervals):
    """Merge overlapping/adjacent [start,end) intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:  # overlap/adjacent
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def usage(schedule, makespan, R=None):
    machine_times = defaultdict(list)

    # schedule: (machine, start, dur, c, j)
    for r, start, dur, *_ in schedule:
        end = start + dur
        # ignore zero-length intervals for stability (optional)
        if end > start:
            machine_times[r].append((start, end))

    # decide which machines to report
    machines = list(range(R)) if R is not None else sorted(machine_times.keys())

    machine_usage = []
    downtime_per_machine = []
    downtime_over_makespan = []

    for r in machines:
        intervals = machine_times.get(r, [])
        merged = merge_intervals(intervals)

        # busy time = union length
        busy_total = sum(e - s for s, e in merged)
        busy_total = max(0.0, min(busy_total, makespan))  # clamp safety
        idle_total = makespan - busy_total if makespan > 0 else 0.0
        rel_load = (busy_total / makespan) if makespan > 0 else 0.0
        machine_usage.append((r, busy_total, rel_load))

        # gaps (idle segments) for “downtime per machine”
        gaps = []
        prev = 0.0
        for s, e in merged:
            if s > prev:
                gaps.append((prev, s))
            prev = max(prev, e)
        if prev < makespan:
            gaps.append((prev, makespan))

        downtimes = [b - a for a, b in gaps]
        downtime_per_machine.append(downtimes)

        # step series: (time, downtime_length_of_current_gap)
        # plot as plt.step(x, y, where="post")
        series = [(a, dt) for (a, _), dt in zip(gaps, downtimes)]
        if not series:
            series = [(0.0, makespan)] if makespan > 0 else [(0.0, 0.0)]
        downtime_over_makespan.append(series)

    return machine_usage, downtime_per_machine, downtime_over_makespan

# def usage(schedule, makespan):
#     from collections import defaultdict

#     machine_times = defaultdict(list)

#     # schedule: (machine, start, dur, c, j)
#     for r, start, dur, *_ in schedule:
#         machine_times[r].append((start, start + dur))

#     machine_usage = []
#     downtime_per_machine = []
#     downtime_over_makespan = []

#     machines = sorted(machine_times.keys())

#     for r in machines:
#         sorted_list = sorted(machine_times[r], key=lambda x: x[0])
#         # gaps: (prev_end, next_start)
#         gaps = [(0, sorted_list[0][0])]  # from 0 to first start
#         gaps += [(a[1], b[0]) for a, b in zip(sorted_list, sorted_list[1:])]
#         gaps.append((sorted_list[-1][1], makespan))  # last end to makespan

#         downtimes = [max(0, b - a) for a, b in gaps]  # idle lengths
#         downtime_per_machine.append(downtimes)

#         idle_total = sum(downtimes)
#         busy_total = makespan - idle_total
#         rel_load = busy_total / makespan if makespan > 0 else 0.0
#         machine_usage.append((r, busy_total, rel_load))

#         # build step-series (time, downtime) for plotting
#         # use end times as x positions: at each end time, downtime changes to next gap
#         series = []
#         for (a, b), dt in zip(gaps, downtimes):
#             series.append((a, dt))  # at time a the downtime "segment" starts
#         downtime_over_makespan.append(series)

#     return machine_usage, downtime_per_machine, downtime_over_makespan

def extract_schedule_cars(S, Y, duration, type_of, CARS, J, R, T, eps=0.5):
    # Startzeit pro (c,j)
    start = {}
    for c in range(CARS):
        for j in range(J):
            for t in range(T):
                if S[c, j, t].X > eps:
                    start[(c, j)] = t
                    break
            if (c, j) not in start:
                raise RuntimeError(f"No start found for c={c}, j={j}")

    # Maschine pro (c,j)
    mach = {}
    for c in range(CARS):
        for j in range(J):
            for r in range(R):
                if Y[c, j, r].X > eps:
                    mach[(c, j)] = r
                    break
            if (c, j) not in mach:
                raise RuntimeError(f"No machine found for c={c}, j={j}")

    # Schedule-Liste
    sched = []
    for c in range(CARS):
        v = type_of[c]
        for j in range(J):
            dur = duration[v][j]
            sched.append((mach[(c, j)], start[(c, j)], dur, c, j))
    return sched


def active_interval_utilization(schedule, R=None, skip_first_last=True):
    """
    Active-window utilization per machine.

    Active window for machine r:
        [first_start_r, last_end_r] over all jobs with dur > 0 on r
    Busy time:
        sum of (dur) of those jobs on r  (assumes no overlap due to capacity constraint)

    Utilization_active = busy_time / (last_end - first_start)

    Skips machines with no positive-duration jobs.
    Optionally skips the first and last machine index (often source/sink with dur=0).
    
    Args:
        schedule: list of tuples (machine, start, dur, c, j) or (machine, start, dur, ...)
        R: total number of machines (optional). If None, inferred from schedule.
        skip_first_last: if True, skips machine 0 and machine R-1 (or min/max inferred).

    Returns:
        active_usage: list of (r, busy_time, active_window, util_active, first_start, last_end)
                      sorted by r
    """
    machine_stats = defaultdict(lambda: {"busy": 0.0, "first": None, "last": None})

    # collect busy time + first/last times (ignore dur==0)
    for r, start, dur, *_ in schedule:
        if dur is None or dur <= 0:
            continue
        end = start + dur

        st = machine_stats[r]
        st["busy"] += float(dur)
        st["first"] = start if st["first"] is None else min(st["first"], start)
        st["last"]  = end   if st["last"]  is None else max(st["last"],  end)

    if not machine_stats:
        return []

    # determine machine universe + which to skip
    if R is not None:
        all_machines = list(range(R))
        first_machine, last_machine = 0, R - 1
    else:
        all_machines = sorted(machine_stats.keys())
        first_machine, last_machine = min(all_machines), max(all_machines)

    skip_set = set()
    if skip_first_last:
        skip_set.update([first_machine, last_machine])

    active_usage = []
    for r in all_machines:
        if r in skip_set:
            continue
        if r not in machine_stats:
            continue  # no positive-duration jobs -> no active window

        busy = machine_stats[r]["busy"]
        first = machine_stats[r]["first"]
        last = machine_stats[r]["last"]
        active_window = float(last - first) if (first is not None and last is not None) else 0.0

        util_active = (busy / active_window) if active_window > 0 else 0.0
        active_usage.append((r, busy, active_window, util_active, first, last))

    return sorted(active_usage, key=lambda x: x[0])


def build_greedy_warmstart(CARS, J, R, T, duration, type_of, predecessors, allowed_machines):
    """
    Returns:
      start_time[(c,j)] = integer start time
      machine[(c,j)]    = chosen machine r
    """
    # when each machine becomes available again
    mach_ready = [0] * R

    # when each car becomes available again (since your model forbids overlapping jobs per car)
    car_ready = [0] * CARS

    # end times for precedence lookup
    end_time = {}  # (c,j) -> end

    start_time = {}
    machine = {}

    for c in range(CARS):
        v = type_of[c]
        for j in range(J):
            # precedence finish time for this car/job
            preds = predecessors[j]
            pred_ready = 0
            if preds:
                pred_ready = max(end_time[(c, h)] for h in preds)

            base_ready = max(car_ready[c], pred_ready)

            best_r = None
            best_start = None

            for r in allowed_machines[(v, j)]:
                s = max(base_ready, mach_ready[r])
                if (best_start is None) or (s < best_start) or (s == best_start and mach_ready[r] < mach_ready[best_r]):
                    best_start = s
                    best_r = r

            dur = duration[v][j]
            s = int(best_start)
            e = s + int(dur)

            if e > T:
                raise ValueError(
                    f"Warmstart exceeds horizon: c={c}, j={j}, start={s}, end={e}, T={T}. "
                    "Increase T or use a safer upper bound."
                )

            start_time[(c, j)] = s
            machine[(c, j)] = best_r
            end_time[(c, j)] = e

            # update availabilities
            car_ready[c] = e
            mach_ready[best_r] = e

    return start_time, machine


def apply_mip_start(S, Y, start_time, machine, CARS, J, R, T):
    """
    Write warmstart into gurobi vars via .Start
    """
    # set all starts to 0 (optional, but makes it clean)
    for c in range(CARS):
        for j in range(J):
            for t in range(T):
                S[c, j, t].Start = 0
            for r in range(R):
                Y[c, j, r].Start = 0

    # set chosen ones
    for c in range(CARS):
        for j in range(J):
            t0 = start_time[(c, j)]
            r0 = machine[(c, j)]
            S[c, j, t0].Start = 1
            Y[c, j, r0].Start = 1