import gurobipy as gp
from gurobipy import GRB
from config import alpha

def set_objective_functions(model, config, y, phi_results, u):
    Vset = config.Vset
    Lset = config.Lset
    Tset = config.Tset
    Jset = config.Jset
    Wset = config.Wset
    B = config.B
    Bplus = config.Bplus

    # Objective Function 7: Minimize number of vessels utilized (psi[v])
    psi = {v: model.addVar(vtype=GRB.BINARY, name=f"psi_{v}") for v in Vset}
    vessel_utilization = gp.quicksum(psi[v] for v in Vset) 

    # ----------------------------------------Objective Function 8: Constraint on vessel utilization---------------------------------------------------------------
    M = Tset[-1]
    for v in Vset:
        model.addConstr(psi[v] >= (1 / M) * gp.quicksum(y[v, l, t] for l in Lset for t in Tset), name=f"utilize_vessel_{v}")

    # ---------------------------------------------Objective Function 9: Minimizing Rebalancing Time---------------------------------------------------------------
    rebalancing_time = gp.quicksum(1 - gp.quicksum(y[v, j, t_prime] for j in Jset for t_prime in phi_results[(j, t)])
                                   for v in Vset for t in Tset )
    
    # --------------------------------------------------     penalise of the tasks  -------------------------------------------------------------------------------
    p = {}  # p[v, t] 

    for v in Vset:
        for t in Tset[:-1]: 
            p[v, t] = model.addVar(vtype=GRB.BINARY, name=f"p_{v}_{t}")

    # count jumps
    for v in Vset:
        for w in Wset:
            for w_prime in Wset:
                if w != w_prime: ## could be revised that only w != the w_p in the same station to save simulation time !!!!!!
                    for t in Tset[:-1]:
                        model.addConstr(p[v, t] >= u[v, w, t] + u[v, w_prime, t + 1] - 1, name=f"constr_{v}_{w}_{w_prime}_{t}")

    total_penalty = gp.quicksum(p[v, t] for v in Vset for t in Tset[:-1])
    rebalancing_time += total_penalty


    # ---------------------------------------------------------Weighted Equation------------------------------------------------------------------------------------
    weighted_objective = (alpha * (19 - vessel_utilization) / 1 +  # Fleet size
                            (1 - alpha) * rebalancing_time / 256) # rebalancing time: 168

    # model.setObjective(weighted_objective, GRB.MINIMIZE)
    model.setObjective(total_penalty, GRB.MINIMIZE)

    # ------------------------------------------Store objectives for later analysis----------------------------------------------------------------------------------
    model._vessel_utilization = vessel_utilization
    model._rebalancing_time = rebalancing_time
    model._total_panelise = total_penalty


    return psi