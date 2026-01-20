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

adaption = 1

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
T= 150
#T = CARS * sum_d   # safe upper bound
#T = int((CARS * sum_d) / max(1, R//2)) + 50 #heuristic
FEZ = [0]*J
SEZ = [int((T))-1]*J

def improvement (duration, adaption):
    if adaption != 0:
        for i in range(D):
            duration[i]= [math.ceil(duration[i][j]*adaption) for j in range(0, len(duration[i]))]
    return duration


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

improved_duration= improvement(duration, adaption)
# Constraints

# time constraint
for c in range(CARS):
    for j in range(J):
        m.addConstr(C >= sum((t + improved_duration[type_of[c]][j]) * S[c,j,t] for t in range(FEZ[j], SEZ[j]+1)))

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
                    m.addConstr(quicksum(t * S[c,h, t] for t in range(FEZ[h], SEZ[h]+1)) + improved_duration[type_of[c]][h] <= quicksum(t * S[c,j, t] for t in range(FEZ[j], SEZ[j]+1)))

# capacity constraint
# the defined capacity needs to be kept
for r in range(R):
    for t in range(T):
        m.addConstr((quicksum(Y[c,j,r] * quicksum(S[c,j,q] for q in range(max(FEZ[j], t - improved_duration[type_of[c]][j] + 1), min(SEZ[j], t) + 1)) for c in range(CARS) for j in range(J)) <= 1))

latest = {(c,j): min(SEZ[j], T - improved_duration[type_of[c]][j]) for c in range(CARS) for j in range(J)}
# ensuring that no job ende after the defined periods
# for c in range(CARS):
#     for j in range(J):
#         for t in range(T):
#             if t < FEZ[j] or t > latest[(c,j)]:
#                 m.addConstr(S[c,j,t] == 0)

# each variant can only be at a machine at the same time
for c in range(CARS):
    for t in range(T):
        m.addConstr(gp.quicksum(gp.quicksum(S[c,j,q] for q in range(max(FEZ[j], t - improved_duration[type_of[c]][j] + 1), min(latest[(c,j)], t) + 1)) for j in range(J)) <= 1)

# Solve
m.optimize()



def extract_schedule_cars(S, Y, duration, CARS, J, eps):
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
    from collections import defaultdict
    
    machine_times= defaultdict(list)
    machine_usage= []
    downtime_per_machine= []
    sorted_times= []

    for first, second, third, *_ in schedule:
        end= second+third
        machine_times[first].append((second, end)) # build up a list for the start and durations per machine
        
    for i in range(len(machine_times)):
        sorted_list= sorted(machine_times[i], key= lambda x: x[0]) #sort the machine_times by their starting times
        sorted_times.append(sorted_list)

    
        times= [(a[1], b[0]) for a, b in zip(sorted_list, sorted_list[1:])] #start+duration ist the first entry and then the distance to the next starting point
        times.insert(0, (0, sorted_list[0][0])) # add the start
        times.append((sorted_list[-1][1], int(C.X))) # add the makespan

            
        downtime= [k[0]-k[1] for k in times] #calculate the downtimes
        absolute_downtime= abs(sum(d for d in downtime)) # sum up the overall downtime
        downtime_per_machine.append([abs(k[0]-k[1]) for k in times])

    

        machine_usage.append((i, C.X-absolute_downtime, 1-absolute_downtime/C.X))

    for i in sorted_times:
        i.insert(0, (0,0)) # add the starting point

    downtime_over_makespan= []
    h= 0
    for i in sorted_times:
        downtime_over_makespan.append([(sorted_times[h][j][1], downtime_per_machine[h][j]) for j in range(len(i))]) # combine times with the downtimes (special case time=0)
        h= h+1

    return machine_usage, downtime_per_machine, downtime_over_makespan

def prepare_step(series, end_time):
    """
    series: Liste von (t, y)
    end_time: vorher berechnete Endzeit (z. B. Makespan)
    """
    x = [t for t, _ in series]
    y = [v for _, v in series]

    # letzte Stufe bis end_time verlängern
    if x[-1] < end_time:
        x.append(end_time)
        y.append(y[-1])

    return x, y
def individual_downtime_per_makespan (downtime_over_makespan):
    end_time = C.X  # oder ein anderer vorher berechneter Wert

    plt.figure(figsize=(10, 6))

    for i, series in enumerate(downtime_over_makespan):
        x, y = prepare_step(series, end_time)
        plt.step(x, y, where="post", label=f"m{i}")

    plt.xlabel("time")
    plt.ylabel("downtime")
    plt.title("downtime over makespan")
    plt.legend()
    plt.grid(True)
    plt.show()

def machine_usage_over_makespan (machine_usage):
    maschinen = [f"m{d[0]}" for d in machine_usage]
    relative = [d[2] for d in machine_usage]
    absolute = [d[1] for d in machine_usage]

    fig, ax1 = plt.subplots(figsize=(10,5))

    # Balken: relative Nutzung
    bars = ax1.bar(maschinen, relative, color="steelblue", alpha=0.7)
    ax1.set_ylabel("relative load", color="steelblue")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("machine")

    # Absolute Werte als Text auf den Balken
    for bar, abs_val in zip(bars, absolute):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{abs_val}", 
                ha="center", color="black", fontsize=9)

    plt.title("machine usage: relativ (columns) + absolute (numbers)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

machine_usage, downtime_per_machine, downtime_over_makespan= usage(schedule)
individual_downtime_per_makespan(downtime_over_makespan)
machine_usage_over_makespan(machine_usage)