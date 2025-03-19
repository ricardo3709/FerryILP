import gurobipy as gp
from tqdm import tqdm
from gurobipy import GRB

def add_constraints(model, config, x, y, Q, z, Z, Z_prime, u, phi_results, E_results, mu_results, taskF_results, xi_results, nu_results):
    functions = config.functions
    vessel_df = config.vessel_df

    # Constraint 1a: Each line has exactlt 1 initial departure
    for l in tqdm(config.Lset, desc='Constraint 1a'):
        model.addConstr(gp.quicksum(x[l, d] for d in config.Dset[l]) == 1, name=f"1a: departure_time_constraint_line{l}")

    # Constraint 1b: Exactly 1 vessel is chosen to do the 1st sailing at the chosen time
    for sailing in tqdm(config.Zset, desc='Constraint 1b'):
        l = int(sailing.split('_')[0])  # line
        s = int(sailing.split('_')[1])  # nth sailing
        for d in config.Dset[l]:
            h_sd = functions['cal_h'](config, s, d, l)
            t = h_sd
            model.addConstr(gp.quicksum(y[v, l, t] for v in config.Vset) == x[l, d], name=f"1b: assign_vessel_l{l}_s{s}_d{d}")

    # Constraint 1c: Vessel cannot do task at infeasible time
    for v in tqdm(config.Vset, desc='Constraint 1c'):
        for j in config.Jset:
            H_vj = functions['cal_H'](config, v, j)
            for t in [t for t in config.Tset if t not in H_vj]:
                y[v, j, t].ub = 0  # Set upper bound of y[v, j, t] to 0

    # Constraint 1d: Vessel cannot do line which is not appropariate
    for t in tqdm(config.Tset, desc='Constraint 1d'):
        for v in config.Vset:
            li_v = functions['cal_li'](config, v)
            for j in [l for l in config.Lset if l not in li_v]:
                y[v, j, t].ub = 0  # Set upper bound of y[v, j, t] to 0

    # Constraint 1e: each vessel start a day at one with 1 task in correct time
    for v in tqdm(config.Vset, desc='Constraint 1e'):
        model.addConstr(gp.quicksum(y[v, j, functions['cal_xi0'](config, v, j)] for j in config.Jset if functions['cal_xi0'](config, v, j) in config.Tset) == 1, name=f"1e: assign_task_v{v}_j{j}_t{t}")

    # Constraint 1f: Vessel can only excute 1 task at a time
    for v in tqdm(config.Vset, desc='Constraint 1f'):
        for t in config.Tset:
            model.addConstr(gp.quicksum(y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)]) <= 1, name=f"1f: task_overlap_v{v}_t{t}")

    # Constraint 1g: one line select one wharf for intemidiate stops
    for j in tqdm(config.Jset, desc='Constraint 1g'):
        if j in config.Lset:
            l = j
            R_l = functions['cal_Rl'](config, l)
            A_l = R_l[-1]  # last station
            for S in R_l[:-1]:
                C_lS = functions['cal_C_lS'](config, S)
                model.addConstr(gp.quicksum(z[w, l] for w in C_lS) == 1, name=f"1g: select_one_wharf_{l}_station_{S}")
        else:
            w = j.split('_')[-1]
            model.addConstr(z[w, j] == 1, name=f"1g: set_upper_bound_{w}_{j}")

    # Constraint 1h: 
    for l in tqdm(config.Lset, desc='Constraint 1h'):
        F_l = functions['cal_F'](config, l)
        for t in config.Tset:
            if t > F_l:
                A_l = functions['cal_Rl'](config, l)[-1]  # last station
                C_lS = functions['cal_C_lS'](config, A_l)  # available wharves at last station  # changed below "+1" 
                model.addConstr(gp.quicksum(Z[l, w, t] for w in C_lS) == gp.quicksum(y[v, l, t - F_l + 1] for v in config.Vset), name=f"1h: last_wharf_use_{l}_t{t}")

    # Constraint 1i
    for l in tqdm(config.Lset, desc='Constraint 1i'):
        A_l = functions['cal_Rl'](config, l)[-1]
        C_lS = functions['cal_C_lS'](config, A_l)
        for w in C_lS:
            muF_l = functions['cal_muF'](config, l)
            for t in config.Tset:
                if t > muF_l - 1:
                    model.addConstr(Z_prime[l, w, t] == gp.quicksum(Z[l, w, t - k] for k in range(muF_l)), name=f"1i: wharf_occupation_{l}_{w}_{t}")

    # Constraint 1j
    for v in tqdm(config.Vset, desc='Constraint 1j'):
        for w in config.Bplus:
            for t in config.Tset:
                if t > 1:
                    phi_w = f'phi_{w}'
                    model.addConstr(y[v, w, t] <= y[v, w, t - 1] + y[v, phi_w, t - 1], name=f"1j: full_period_charging_start_v{v}_w{w}_t{t}")

    # Constraint 1k
    for v in tqdm(config.Vset, desc='Constraint 1k'):
        for w in config.Bplus:
            for t in config.Tset:
                if t <= config.Tset[-1] - 1:
                    phi_w = f'phi_{w}'
                    model.addConstr(y[v, w, t] <= y[v, w, t + 1] + y[v, phi_w, t + 1], name=f"1k: full_period_charging_2_v{v}_w{w}_t{t}")


    # Combined Constraint 2
    for j in tqdm(config.Jset, desc='Constraint 2'):
        for t in config.Tset:
            follow_tasks = taskF_results[(j, t)]
            if follow_tasks:
                for v in config.Vset:
                    follow_task = gp.quicksum(y[v, j_prime, t + int(mu_results[j]) + int(xi_results[(j, j_prime)])] 
                                            for j_prime in follow_tasks)
                    model.addConstr(follow_task >= y[v, j, t], name=f"2: follow_task_v{v}_j{j}_t{t}")

    # # Constraint 3                   
    for v in tqdm(config.Vset, desc='Constraint 3'):
        for j in config.Jset:
            for t in config.Tset:
                sum_y_v_jprime_tprime = gp.quicksum(y[v, j_prime, t_prime] 
                                                    for j_prime in config.Jset 
                                                    for t_prime in nu_results[(t, j, j_prime)])            
                sum_nu_t_j_jprime = sum(len(nu_results[(t, j, j_prime)]) for j_prime in config.Jset) 
                model.addConstr(sum_y_v_jprime_tprime <= sum_nu_t_j_jprime * (1 - y[v, j, t]),name=f"3_eq_v{v}_j{j}_t{t}")

    # Constraint 4
    # time period that the F3 occupies the wharves    
    Bar1_occupied = [14, 15, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69]
    CQ5_occupied = [20, 21, 22, 25, 26, 27, 26, 27, 28, 31, 32, 33, 32, 33, 34, 35, 36, 37, 37, 38, 39, 38, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 49, 50, 51, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 67, 68]

    for w in tqdm(config.Wset, desc='Constraint 4'):
        for t in config.Tset:
            sum_yz = gp.quicksum(y[v, j, t_prime] * z[w, j] for v in config.Vset for (j, t_prime) in E_results[(w, t)])
            sum_Z_prime = gp.quicksum(Z_prime[l, w, t] for l in config.Lset if (l, w, t) in Z_prime)
            wharf_capacity = functions['cal_Cw'](config, w)
            # check F3 occupation
            if w == 'CQ5': 
                wharf_capacity -= CQ5_occupied.count(t)
            elif w == 'Bar1':
                wharf_capacity -= Bar1_occupied.count(t)
            model.addConstr(sum_yz + sum_Z_prime <= wharf_capacity, name=f"4: capacity_constraint_w{w}_t{t}")


    # # CHARGING REQUIREMENT
    # Constraint 5a
    for v in tqdm(config.Vset, desc='Constraint 5a'):
        for t in config.Tset:
            model.addConstr(Q[v, t] >= 0.5, name=f"5a: battery_non_negative_v{v}_t{t}")

    # Constraint 5b
    for v in tqdm(config.Vset, desc='Constraint 5b'):
        for t in config.Tset:
            model.addConstr(Q[v, t] <= 1, name=f"5b: battery_max_capacity_v{v}_t{t}")

    # Constraint 5c
    rv = {v: vessel_df[vessel_df['Vessel code'] == v]['rv'].iloc[0] for v in config.Vset}
    Qv0 = {v: vessel_df[vessel_df['Vessel code'] == v]['Qv0'].iloc[0] for v in config.Vset}

    for v in tqdm(config.Vset, desc='Constraint 5c'):
        for t in config.Tset:
            if t == 1:
                model.addConstr(Qv0[v]
                                + gp.quicksum(functions['cal_q'](config, v, j, t - t_prime) * y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)])
                                + rv[v] * (1 - gp.quicksum(y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)]))
                                >= Q[v, t],
                                name=f"5c: battery_update_v{v}_t{t}")
            else:
                model.addConstr(Q[v, t - 1]
                                + gp.quicksum(functions['cal_q'](config, v, j, t - t_prime) * y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)])
                                + rv[v] * (1 - gp.quicksum(y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)]))
                                >= Q[v, t],
                                name=f"5c: battery_update_v{v}_t{t}")

    # # CREW PAUSE:
    # Constraint 6a
    for v in tqdm(config.Vset, desc='Constraint 6a'):
        model.addConstr(
            gp.quicksum(y[v, j, t] for j in config.Bc for t in config.Tset) >= config.nc,
            name=f"6a: min_crew_pauses_{v}")

    # Constraint 6b
    for v in tqdm(config.Vset, desc='Constraint 6b'):
        for t in config.Tset:
            if t < (config.Tset[-1] - (config.Tc // config.period_length + 1)): # use cal_time_period function 
                for t_prime in range(1, config.Tc // config.period_length + 1):
                    model.addConstr(
                        gp.quicksum(y[v, j, t + t_prime] for j in config.Bc) >= 1,
                        name=f"6b: max_distance_pauses_v{v}_t{t}_t_prime{t_prime}")


    # new constraints for new variable records vessel location
    for v in tqdm(config.Vset, desc='vessel location'):  
        for t in config.Tset:
            for w in config.Wset:
                model.addConstr(u[v, w, t] == gp.quicksum(Z_prime[l, w, t] * y[v, l, t - mu_results[l] + 1] for l in config.Lset if (l, w, t) in Z_prime and (v, l, t - mu_results[l] + 1) in y) + gp.quicksum(y[v, j, t_prime] * z[w, j] for (j, t_prime) in E_results[(w, t)] if (v, j, t_prime) in y and (w, j) in z))


    print('All constraintrs are ready.\n')


