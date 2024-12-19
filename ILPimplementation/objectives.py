import gurobipy as gp
from gurobipy import GRB

def set_objective_functions(model, config, y, phi_results):
    Vset = config.Vset
    Lset = config.Lset
    Tset = config.Tset
    Jset = config.Jset
    B = config.B
    Bplus = config.Bplus

    # Objective Function 7: Minimize number of vessels utilized (psi[v])
    psi = {v: model.addVar(vtype=GRB.BINARY, name=f"psi_{v}") for v in Vset}
    vessel_utilization = gp.quicksum(psi[v] for v in Vset) ## 

    # ----------------------------------------Objective Function 8: Constraint on vessel utilization---------------------------------------------------------------
    M = Tset[-1]
    for v in Vset:
        model.addConstr(psi[v] >= (1 / M) * gp.quicksum(y[v, l, t] for l in Lset for t in Tset), name=f"utilize_vessel_{v}")

    # ---------------------------------------------Objective Function 9: Minimizing Rebalancing Time---------------------------------------------------------------
    rebalancing_time = gp.quicksum(1 - gp.quicksum(y[v, j, t_prime] for j in Jset for t_prime in phi_results[(j, t)])
                                   for v in Vset for t in Tset )
    
    # --------------------------------------------------     penalise of the tasks  -------------------------------------------------------------------------------
    p_t = {t: model.addVar(vtype=GRB.CONTINUOUS, name=f"p_{t}") for t in Tset}

    # for t in Tset[:-1]:
    #     model.addConstr(p_t[t] >= gp.quicksum(y[v, j, t] * y[v, j, t + 1]for v in Vset for j in B + Bplus),name=f"penalty_{t}" )
    for t in Tset[:-1]:
        model.addConstr(p_t[t] >= gp.quicksum(y[v, j, t] * y[v, j_p, t + 1]for v in Vset for j in B + Bplus for j_p in B + Bplus if j_p != j),name=f"penalty_{t}")

    total_panelise = gp.quicksum(p_t[t] for t in Tset)
    rebalancing_time += total_panelise

    # ---------------------------------------------------------Weighted Equation-------------------------------------------------------------------------------------
    alpha = 0.9  # % focus on fleet size

    # Combined Weighted Objective
    weighted_objective = (alpha * (19 - vessel_utilization) / 1 +  # Fleet size
                            (1 - alpha) * rebalancing_time / 256) # rebalancing time: 168

    model.setObjective(weighted_objective, GRB.MINIMIZE)

    # ------------------------------------------Store objectives for later analysis----------------------------------------------------------------------------------
    model._vessel_utilization = vessel_utilization
    model._rebalancing_time = rebalancing_time
    model._total_panelise = total_panelise


    return psi