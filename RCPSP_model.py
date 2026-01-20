import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import math


Operations = [1, 2, 3, 4]
J= len(Operations)
Ressources = [1,2,3,4,5,6,7,8]
R= len(Ressources)


predecessors = [
    [],     # j0
    [0],    # j1 nach j0
    [1],    # j2 nach j1
    [2]     # j3 nach j2
]

duration = [[1, 3, 2, 1], [1, 2, 2, 3], [1, 1, 2, 3]]
D= len(duration)

adaption = 1.1

q = {0: 20, 1: 20, 2: 20} # number of cars produced per Variant


# allowed machines
variants= {0: [[0], [1,2,3], [5], [7]], 1: [[0], [1,2,3], [4,6], [7]], 2: [[0], [1], [6], [7]]}
allowed_machines= {}

V = len(variants)

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
T= 1500
#T = CARS * sum_d   # safe upper bound
#T = int((CARS * sum_d) / max(1, R//2)) + 50 #heuristic
FEZ = [0]*J
SEZ = [int((T))-1]*J

def improvement (duration, adaption):
    if adaption != 0:
        for i in range(D):
            duration[i]= [math.ceil(duration[i][j]*adaption) for j in range(0, len(duration[i]))]
    return duration

def RCPSP(J, R, predecessors, duration, CARS, type_of, t, FEZ, SEZ):
    m = gp.Model("RCPSP")

    #set params
    #m.setParam('Heuristics', 0.1)
    #m.setParam('MIPgap', )

    # decision variables
    S = m.addVars(CARS, J, T, vtype= GRB.BINARY) # ==1 if job j starts for car c in time t
    Y = m.addVars(CARS, J, R, vtype=GRB.BINARY) # ==1 if job j is for car c on maschine r
    C = m.addVar(lb= 0, vtype= GRB.CONTINUOUS) # makespan

    # Objective
    m.setObjective(C, GRB.MINIMIZE)

    # Constraints

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
    # for c in range(CARS):
    #     for j in range(J):
    #         for t in range(T):
    #             if t < FEZ[j] or t > latest[(c,j)]:
    #                 m.addConstr(S[c,j,t] == 0)

    # each variant can only be at a machine at the same time
    for c in range(CARS):
        for t in range(T):
            m.addConstr(gp.quicksum(gp.quicksum(S[c,j,q] for q in range(max(FEZ[j], t - duration[type_of[c]][j] + 1), min(latest[(c,j)], t) + 1)) for j in range(J)) <= 1)

    # Solve
    m.optimize()
    return S, Y, C



def extract_schedule_cars(S, Y, duration, CARS, J,T, eps=0.5):
    """
    Returns list of (machine, start, dur, car, job).
    """
    # start time per (c,j)
    start = {}
    for (c, j, t), var in S.items():
        if var.X > eps:
            start[(c, j)] = t

    # machine per (c,j)
    mach = {}
    for (c, j, r), var in Y.items():
        if var.X > eps:
            mach[(c, j)] = r

    sched = []
    for c in range(CARS):
        for j in range(J):
            sched.append((mach[(c, j)], start[(c, j)], duration[type_of[c]][j]))
    return sched

def usage (schedule):
    import numpy as np
    from collections import defaultdict

    machine_times= defaultdict(list)
    machine_usage= []

    for first, second, *_ in schedule:
        machine_times[first].append(second)

    for i in range(len(machine_times)):
        machine_times[i].sort()
        machine_usage.append((sum(np.diff(machine_times[i])), np.diff(machine_times[i]).mean()))

    return machine_usage


if __name__ == "__main__":
    solution= [RCPSP(J, R, predecessors, duration, CARS, type_of, T, FEZ, SEZ)]
    schedule= extract_schedule_cars(S, Y, duration, CARS, J,T, eps=0.5)
    usage(schedule)
