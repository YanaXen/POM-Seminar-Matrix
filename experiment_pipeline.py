import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import pandas as pd
import os

from utils import extract_schedule, usage, active_interval_utilization, build_greedy_warmstart, apply_mip_start

def run_model(Operations, Resources , predecessors, duration, q, variants):
    """
    Builds and solves an RCPSP-like scheduling model with Gurobi for multiple
    product variants and resource-dependent operations.

    Args:
        Operations (list): List of operations.
        Resources (list): List of available machines/resources.
        predecessors (dict): Precedence relations for each operation.
        duration (dict): Processing times per variant and operation.
        q (dict): Number of products to be produced per variant.
        variants (dict): Allowed machines per variant and operation.

    Returns:
        makespan (float): Objective value of the solved schedule.
        downtime_per_machine (dict): Downtime for each machine.
        downtime_over_makespan (dict): Downtime per machine relative to the makespan.
        machine_usage (dict): Overall utilization per machine.
        active_usage (dict): Utilization over active intervals.
        total_work_content (float): Total processing time over all products.
        status (int): Gurobi solver status code.
        runtime (float): Solver runtime in seconds.
        gap (float | None): MIP gap if available, otherwise None.
    """

    J= len(Operations)
    R= len(Resources )
    V = len(variants)

    allowed_machines= {}

    for v in variants:
        for j in range(len(variants[v])):
            allowed_machines.update({(v,j): variants[v][j]})

    products = [] # each product produced of each variant becomes it's own entity
    for v in variants.keys():
        for idx in range(q[v]):
            products.append((v, idx))

    PRODUCTS = len(products)
    type_of = {c: products[c][0] for c in range(PRODUCTS)}
    sum_d = sum(duration[0])
    T = PRODUCTS * sum_d 
    

    FEZ = [0]*J
    SEZ = [T-1]*J

    m = gp.Model("RCPSP")

    # decision variables
    S = m.addVars(PRODUCTS, J, T, vtype= GRB.BINARY) # ==1 if job j starts for product c in time t
    Y = m.addVars(PRODUCTS, J, R, vtype=GRB.BINARY) # ==1 if job j is for product c on maschine r
    C = m.addVar(lb= 0, vtype= GRB.CONTINUOUS) # makespan

    # Objective
    m.setObjective(C, GRB.MINIMIZE)

    # Constraints

    # symbreak (products are sorted depending on their starting timepoint)
    products_by_v = {v: [c for c in range(PRODUCTS) if type_of[c] == v] for v in range(V)}

    for v in products_by_v:
        products_by_v[v].sort()

    j0 = 0
    for v, product_list in products_by_v.items():
        for a, b in zip(product_list[:-1], product_list[1:]):
            m.addConstr(
                quicksum(t * S[a, j0, t] for t in range(FEZ[j0], SEZ[j0] + 1))
                <=
                quicksum(t * S[b, j0, t] for t in range(FEZ[j0], SEZ[j0] + 1)),
                name=f"sym_start_v{v}_c{a}_c{b}"
            )

    # time constraint
    for c in range(PRODUCTS):
        for j in range(J):
            m.addConstr(C >= sum((t + duration[type_of[c]][j]) * S[c,j,t] for t in range(FEZ[j], SEZ[j]+1)))

    # timeslot constraint
    # the job needs to be finished in a certain timeframe
    for c in range(PRODUCTS):
        for j in range(J):
            m.addConstr((quicksum(S[c, j, t] for t in range(FEZ[j], SEZ[j]+1)) == 1))

    # variant_based useable machines
    # variants are allowed to be produced on a predefined set of machines
    for c in range(PRODUCTS):
        v = type_of[c]
        for j in range(J):
            m.addConstr(quicksum(Y[c,j,r] for r in allowed_machines[(v,j)]) == 1)
            for r in range(R):
                if r not in allowed_machines[(v,j)]:
                    m.addConstr(Y[c,j,r] == 0)

    # precendence constraint
    # realise the order of the correct process
    for c in range(PRODUCTS):
        for j in range(J):
                for h in predecessors[j]:
                        m.addConstr(quicksum(t * S[c,h, t] for t in range(FEZ[h], SEZ[h]+1)) + duration[type_of[c]][h] <= quicksum(t * S[c,j, t] for t in range(FEZ[j], SEZ[j]+1)))

    # capacity constraint
    # the defined capacity needs to be kept
    for r in range(R):
        for t in range(T):
            m.addConstr((quicksum(Y[c,j,r] * quicksum(S[c,j,q] for q in range(max(FEZ[j], t - duration[type_of[c]][j] + 1), min(SEZ[j], t) + 1)) for c in range(PRODUCTS) for j in range(J)) <= 1))

    latest = {(c,j): min(SEZ[j], T - duration[type_of[c]][j]) for c in range(PRODUCTS) for j in range(J)}

    # ensuring that no job ende after the defined periods
    for c in range(PRODUCTS):
        for j in range(J):
            for t in range(T):
                if t < FEZ[j] or t > latest[(c,j)]:
                    m.addConstr(S[c,j,t] == 0)

    # each product can only be at a machine at the same time
    for c in range(PRODUCTS):
        for t in range(T):
            m.addConstr(gp.quicksum(gp.quicksum(S[c,j,q] for q in range(max(FEZ[j], t - duration[type_of[c]][j] + 1), min(latest[(c,j)], t) + 1)) for j in range(J)) <= 1)

    m.setParam("TimeLimit", 1800)  # 1800 30 minutes
    #m.setParam("MIPGap", 0.02) 
    #m.setParam("Heuristics", 0.2)
    # start_time, machine = build_greedy_warmstart(
    #     PRODUCTS=PRODUCTS,
    #     J=J,
    #     R=R,
    #     T=T,
    #     duration=duration,
    #     type_of=type_of,
    #     predecessors=predecessors,
    #     allowed_machines=allowed_machines
    # )

    # apply_mip_start(S, Y, start_time, machine, PRODUCTS, J, R, T)
    
    m.optimize()

    if m.SolCount == 0:
        return None, None, None, None, None, None, m.Status, m.Runtime, None

    makespan = C.X

    total_work_content = 0
    for variant in q.keys():
        for job in range(len(Operations)):
            total_work_content += duration[variant][job] * q[variant]

    schedule = extract_schedule(S, Y, duration, type_of, PRODUCTS, J, R, T, eps=0.5)

    machine_usage, downtime_per_machine, downtime_over_makespan = usage(schedule, makespan, R=R)

    active_usage = active_interval_utilization(schedule, R=R, skip_first_last=True)

    gap = m.MIPGap if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    return makespan, downtime_per_machine, downtime_over_makespan, machine_usage, active_usage, total_work_content, m.Status, m.Runtime, gap
