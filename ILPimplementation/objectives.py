import gurobipy as gp
from gurobipy import GRB

def set_objective_functions(model, config, y, phi_results):
    Vset = config.Vset
    Lset = config.Lset
    Tset = config.Tset
    Jset = config.Jset

    # Variable psi[v]
    psi = {v: model.addVar(vtype=GRB.BINARY, name=f"psi_{v}") for v in Vset}

    # Objective Function 7
    model.setObjective(gp.quicksum(psi[v] for v in Vset), GRB.MINIMIZE)

    # Objective Function 8
    M = Tset[-1]
    for v in Vset:
        model.addConstr(psi[v] >= (1 / M) * gp.quicksum(y[v, l, t] for l in Lset for t in Tset), name=f"utilize_vessel_{v}")

    # # Objective Function 9: Minimizing Rebalancing Time
    # rebalancing_time = gp.quicksum(1 - gp.quicksum(y[v, j, t_prime] for j in Jset for t_prime in phi_results[(j, t)]) for v in Vset for t in Tset)
    # model.setObjective(rebalancing_time, GRB.MINIMIZE)

    return psi
