import gurobipy as gp
from gurobipy import GRB
from config import alpha

def set_objective_functions(model, config, y, phi_results, p):
    Vset = config.Vset
    Lset = config.Lset
    Tset = config.Tset
    Jset = config.Jset
    Wset = config.Wset
    B = config.B
    Bplus = config.Bplus


# JUMPS PENALTY
    total_penalty = gp.quicksum(p[v, t] for v in Vset for t in Tset[:-1])   
    model.setObjective(total_penalty, GRB.MINIMIZE)


# FLEET SIZE
    psi = {v: model.addVar(vtype=GRB.BINARY, name=f"psi_{v}") for v in Vset}
    vessel_utilization = gp.quicksum(psi[v] for v in Vset) 

    M = Tset[-1]
    for v in Vset:
        model.addConstr(psi[v] >= (1 / M) * gp.quicksum(y[v, l, t] for l in Lset for t in Tset), name=f"utilize_vessel_{v}")

# REBALANCING TIME
    rebalancing_time = gp.quicksum(1 - gp.quicksum(y[v, j, t_prime] for j in Jset for t_prime in phi_results[(j, t)]) for v in Vset for t in Tset )
    rebalancing_time += total_penalty

# WEIGHTED OBJECTIVES
    # weighted_objective = (alpha * ( 26 - vessel_utilization) / 1 +  # Fleet size
    #                         (1 - alpha) * rebalancing_time / 256) # rebalancing time: 168
    # model.setObjective(weighted_objective, GRB.MINIMIZE)



    # store
    model._vessel_utilization = vessel_utilization
    model._rebalancing_time = rebalancing_time
    model._total_panelise = total_penalty


    return psi