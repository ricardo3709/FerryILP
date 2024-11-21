import gurobipy as gp
from gurobipy import GRB


def set_objective_functions(model, config, y, phi_results):
    Vset = config.Vset
    Lset = config.Lset
    Tset = config.Tset
    Jset = config.Jset

    # Objective Function 7: Minimize number of vessels utilized (psi[v])
    psi = {v: model.addVar(vtype=GRB.BINARY, name=f"psi_{v}") for v in Vset}
    vessel_utilization = gp.quicksum(psi[v] for v in Vset) ## 

    # Objective Function 8: Constraint on vessel utilization
    M = Tset[-1]
    for v in Vset:
        model.addConstr(psi[v] >= (1 / M) * gp.quicksum(y[v, l, t] for l in Lset for t in Tset), name=f"utilize_vessel_{v}")

    # Objective Function 9: Minimizing Rebalancing Time
    rebalancing_time = gp.quicksum(1 - gp.quicksum(y[v, j, t_prime] for j in Jset for t_prime in phi_results[(j, t)]) for v in Vset for t in Tset)
    
    weighted_objective = 0.9*vessel_utilization/19 + 0.1*rebalancing_time/168

    # # Set multi-objective optimization
    # model.setObjectiveN(vessel_utilization, index=0, priority=1, name="Minimize Vessel Utilization")
    # model.setObjectiveN(rebalancing_time, index=1, priority=0, name="Minimize Rebalancing Time")

    model.setObjective(weighted_objective, GRB.MINIMIZE)

    model._vessel_utilization = vessel_utilization
    model._rebalancing_time = rebalancing_time

    
    return psi