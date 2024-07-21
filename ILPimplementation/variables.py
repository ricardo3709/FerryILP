import gurobipy as gp
from gurobipy import GRB

def define_variables(model, config, cal_C, cal_Rl, cal_C_lS):
    Lset = config.Lset
    Dset = config.Dset
    Vset = config.Vset
    Jset = config.Jset
    Tset = config.Tset

    # variable x[l, d]
    x = {}
    for l in Lset:
        for d in Dset[l]:
            x[l, d] = model.addVar(vtype=GRB.BINARY, name=f"x_{l}_{d}")

    # variable y[v, j, t]
    y = {}
    for v in Vset:
        for j in Jset:
            for t in Tset:
                y[v, j, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}_{j}_{t}")

    # variable Q[v, t]
    Q = {}
    for v in Vset:
        for t in Tset:
            Q[v, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Q_{v}_{t}")

    # variable z[j, w]
    z = {}
    for j in Jset:
        C_j = cal_C(config, j)
        for w in C_j:
            z[w, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{w}_{j}")

    # variable Z[l, w, t]
    Z = {}
    for l in Lset:
        A_l = cal_Rl(config, l)[-1]
        C_lS = cal_C_lS(config, A_l)
        for w in C_lS:
            for t in Tset:
                Z[l, w, t] = model.addVar(vtype=GRB.BINARY, name=f"Z_{l}_{w}_{t}")

    # variable Z_prime[l, w, t]
    Z_prime = {}
    for l in Lset:
        A_l = cal_Rl(config, l)[-1]
        C_lS = cal_C_lS(config, A_l)
        for w in C_lS:
            for t in Tset:
                Z_prime[l, w, t] = model.addVar(vtype=GRB.BINARY, name=f"Z_prime_{l}_{w}_{t}")

    return x, y, Q, z, Z, Z_prime
