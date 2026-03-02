import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import pandas as pd
import os

from experiment_config import experiment_configuration
from utils import extract_schedule_cars, usage, active_interval_utilization, build_greedy_warmstart, apply_mip_start
from diagram_generation import diagram_generation

def run_model(Operations, Ressources, predecessors, duration, q, variants):

    J= len(Operations)
    R= len(Ressources)
    V = len(variants)

    allowed_machines= {}

    for v in variants:
        for j in range(len(variants[v])):
            allowed_machines.update({(v,j): variants[v][j]})

    cars = [] # each car produced of each variant becomes it's own entity
    for v in variants.keys():
        for idx in range(q[v]):
            cars.append((v, idx))

    CARS = len(cars)
    type_of = {c: cars[c][0] for c in range(CARS)}
    sum_d = sum(duration[0])
    T = CARS * sum_d 
    

    FEZ = [0]*J
    SEZ = [T-1]*J

    m = gp.Model("RCPSP")

    # decision variables
    S = m.addVars(CARS, J, T, vtype= GRB.BINARY) # ==1 if job j starts for car c in time t
    Y = m.addVars(CARS, J, R, vtype=GRB.BINARY) # ==1 if job j is for car c on maschine r
    C = m.addVar(lb= 0, vtype= GRB.CONTINUOUS) # makespan

    # Objective
    m.setObjective(C, GRB.MINIMIZE)

    # Constraints

    # symbreak (cars are sorted depending on their starting timepoint)
    cars_by_v = {v: [c for c in range(CARS) if type_of[c] == v] for v in range(V)}

    for v in cars_by_v:
        cars_by_v[v].sort()

    j0 = 0
    for v, car_list in cars_by_v.items():
        for a, b in zip(car_list[:-1], car_list[1:]):
            m.addConstr(
                quicksum(t * S[a, j0, t] for t in range(FEZ[j0], SEZ[j0] + 1))
                <=
                quicksum(t * S[b, j0, t] for t in range(FEZ[j0], SEZ[j0] + 1)),
                name=f"sym_start_v{v}_c{a}_c{b}"
            )

    # time constraint
    for c in range(CARS):
        for j in range(J):
            m.addConstr(C >= sum((t + duration[type_of[c]][j]) * S[c,j,t] for t in range(FEZ[j], SEZ[j]+1)))

    # timeslot constraint
    # the job needs to be finished in a certain timeframe
    for c in range(CARS):
        for j in range(J):
            m.addConstr((quicksum(S[c, j, t] for t in range(FEZ[j], SEZ[j]+1)) == 1))

    # variant_based useable machines
    # variants are allowed to be produced on a predefined set of machines
    for c in range(CARS):
        v = type_of[c]
        for j in range(J):
            m.addConstr(quicksum(Y[c,j,r] for r in allowed_machines[(v,j)]) == 1)
            for r in range(R):
                if r not in allowed_machines[(v,j)]:
                    m.addConstr(Y[c,j,r] == 0)

    # precendence constraint
    # realise the order of the correct process
    for c in range(CARS):
        for j in range(J):
                for h in predecessors[j]:
                        m.addConstr(quicksum(t * S[c,h, t] for t in range(FEZ[h], SEZ[h]+1)) + duration[type_of[c]][h] <= quicksum(t * S[c,j, t] for t in range(FEZ[j], SEZ[j]+1)))

    # capacity constraint
    # the defined capacity needs to be kept
    for r in range(R):
        for t in range(T):
            m.addConstr((quicksum(Y[c,j,r] * quicksum(S[c,j,q] for q in range(max(FEZ[j], t - duration[type_of[c]][j] + 1), min(SEZ[j], t) + 1)) for c in range(CARS) for j in range(J)) <= 1))

    latest = {(c,j): min(SEZ[j], T - duration[type_of[c]][j]) for c in range(CARS) for j in range(J)}

    # ensuring that no job ende after the defined periods
    for c in range(CARS):
        for j in range(J):
            for t in range(T):
                if t < FEZ[j] or t > latest[(c,j)]:
                    m.addConstr(S[c,j,t] == 0)

    # each variant can only be at a machine at the same time
    for c in range(CARS):
        for t in range(T):
            m.addConstr(gp.quicksum(gp.quicksum(S[c,j,q] for q in range(max(FEZ[j], t - duration[type_of[c]][j] + 1), min(latest[(c,j)], t) + 1)) for j in range(J)) <= 1)

    m.setParam("TimeLimit", 1800)  # 1800 30 minutes
    #m.setParam("MIPGap", 0.02) 
    #m.setParam("Heuristics", 0.2)
    # start_time, machine = build_greedy_warmstart(
    #     CARS=CARS,
    #     J=J,
    #     R=R,
    #     T=T,
    #     duration=duration,
    #     type_of=type_of,
    #     predecessors=predecessors,
    #     allowed_machines=allowed_machines
    # )

    # apply_mip_start(S, Y, start_time, machine, CARS, J, R, T)
    
    m.optimize()

    if m.SolCount == 0:
        return None, None, None, None, None, None, m.Status, m.Runtime, None

    makespan = C.X

    total_work_content = 0
    for variant in q.keys():
        for job in range(len(Operations)):
            total_work_content += duration[variant][job] * q[variant]

    schedule = extract_schedule_cars(S, Y, duration, type_of, CARS, J, R, T, eps=0.5)

    machine_usage, downtime_per_machine, downtime_over_makespan = usage(schedule, makespan, R=R)

    active_usage = active_interval_utilization(schedule, R=R, skip_first_last=True)

    gap = m.MIPGap if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    return makespan, downtime_per_machine, downtime_over_makespan, machine_usage, active_usage, total_work_content, m.Status, m.Runtime, gap

def main(continue_run = False):
    results = pd.DataFrame(columns=["experiment", "experiment_version", "makespan", "downtime_over_makespan", "downtime_per_machine", "usage", "active_usage", "total_work_content", "status", "runtime", "gap"])

    if continue_run and os.path.exists("data/experiment_results_backup_9.csv"):
        results = pd.read_csv("data/experiment_results_backup_9.csv")
    
    done = set(zip( results["experiment"].astype(str), results["experiment_version"].astype(str)))

    for experiment in experiment_configuration.keys():
        if experiment not in  ["Experiment 1", "Experiment 2", "Experiment 6", "Experiment 7"]:
            continue
        for variation in experiment_configuration[experiment].keys():
            key = (str(experiment), str(variation))
            if key in done:
                continue    

            operations = experiment_configuration[experiment][variation]["Operations"]
            ressources = experiment_configuration[experiment][variation]["Ressources"]
            predecessors = experiment_configuration[experiment][variation]["predecessors"]
            duration = experiment_configuration[experiment][variation]["duration"]
            q = experiment_configuration[experiment][variation]["q"]
            variants = experiment_configuration[experiment][variation]["variants"]
            

            print(f"starting experiment {experiment} and variant {variation}")
            makespan, downtime_per_machine, downtime_over_makespan, usage, active_usage, total_work_content, status, runtime, gap = run_model(operations, ressources, predecessors, duration, q, variants)
            print(downtime_per_machine)
            row = {
                "experiment": experiment,
                "experiment_version": variation,
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

            done.add(key)

            results.to_csv("data/experiment_results_backup_9.csv", index=False)


    # create diagrams 
    diagram_generation(results)

    # save data
    results.to_csv("data/experiment_results_9.csv")

if __name__ == "__main__":
    main(continue_run=False)