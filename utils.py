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