import gurobipy as gp
from tqdm import tqdm
from gurobipy import GRB

def add_constraints(model, config, x, y, Q, z, Z, Z_prime, phi_results, E_results, mu_results, taskF_results, xi_results):
    functions = config.functions
    vessel_df = config.vessel_df

    # Constraint 1a
    for l in tqdm(config.Lset, desc='Constraint 1a'):
        model.addConstr(gp.quicksum(x[l, d] for d in config.Dset[l]) == 1, name=f"1a: departure_time_constraint_line{l}")

    # Constraint 1b
    for sailing in tqdm(config.Zset, desc='Constraint 1b'):
        l = int(sailing.split('_')[0])  # line
        s = int(sailing.split('_')[1])  # nth sailing
        for d in config.Dset[l]:
            h_sd = functions['cal_h'](config, s, d, l)
            t = h_sd
            model.addConstr(gp.quicksum(y[v, l, t] for v in config.Vset) == x[l, d], name=f"1b: assign_vessel_l{l}_s{s}_d{d}")

    # Constraint 1c
    for v in tqdm(config.Vset, desc='Constraint 1c'):
        for j in config.Jset:
            H_vj = functions['cal_H'](config, v, j)
            for t in [t for t in config.Tset if t not in H_vj]:
                y[v, j, t].ub = 0  # Set upper bound of y[v, j, t] to 0

    # Constraint 1d
    for t in tqdm(config.Tset, desc='Constraint 1d'):
        for v in config.Vset:
            li_v = functions['cal_li'](config, v)
            for j in [l for l in config.Lset if l not in li_v]:
                y[v, j, t].ub = 0  # Set upper bound of y[v, j, t] to 0

    # Constraint 1e
    for v in tqdm(config.Vset, desc='Constraint 1e'):
        model.addConstr(gp.quicksum(y[v, j, functions['cal_xi0'](config, v, j)] for j in config.Jset if functions['cal_xi0'](config, v, j) in config.Tset) == 1, name=f"1e: assign_task_v{v}_j{j}_t{t}")

    # Constraint 1f
    for v in tqdm(config.Vset, desc='Constraint 1f'):
        for t in config.Tset:
            model.addConstr(gp.quicksum(y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)]) <= 1, name=f"1f: task_overlap_v{v}_t{t}")

    # Constraint 1g
    # for l in tqdm(config.Lset, desc='Constraint 1g'):
    #     R_l = functions['cal_Rl'](config, l)
    #     A_l = R_l[-1]  # last station
    #     for S in R_l[:-1]: # 29July revised, original code: for S in [station for station in R_l if station != A_l]:
    #         C_lS = functions['cal_C_lS'](config, S)
    #         model.addConstr(gp.quicksum(z[w, l] for w in C_lS) == 1, name=f"1g: select_one_wharf_{l}_station_{S}")

    for j in tqdm(config.Jset, desc='Constraint 1g'):
        if j in config.Lset:
            l = j
            R_l = functions['cal_Rl'](config, l)
            A_l = R_l[-1]  # last station
            for S in R_l[:-1]: # 29July revised, original code: for S in [station for station in R_l if station != A_l]:
                C_lS = functions['cal_C_lS'](config, S)
                model.addConstr(gp.quicksum(z[w, l] for w in C_lS) == 1, name=f"1g: select_one_wharf_{l}_station_{S}")
        else:
            w = j.split('_')[-1]
            model.addConstr(z[w, j] == 1, name=f"1g: set_upper_bound_{w}_{j}")

    # Constraint 1h
    for l in tqdm(config.Lset, desc='Constraint 1h'):
        F_l = functions['cal_F'](config, l)
        for t in config.Tset:
            if t > F_l:
                A_l = functions['cal_Rl'](config, l)[-1]  # last station
                C_lS = functions['cal_C_lS'](config, A_l)  # available wharves at last station
                model.addConstr(gp.quicksum(Z[l, w, t] for w in C_lS) == gp.quicksum(y[v, l, t - F_l] for v in config.Vset), name=f"1h: last_wharf_use_{l}_t{t}")

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
                    buffer = 3  # Buffer = 1 -> same functionality with rthe original expression
                    follow_task = gp.quicksum(y[v, j_prime, t_prime] 
                                            for j_prime in follow_tasks 
                                            for t_prime in range(t + int(mu_results[j]) + int(xi_results[(j, j_prime)]), 
                                                                t + int(mu_results[j]) + int(xi_results[(j, j_prime)]) + buffer)
                                            if t_prime in config.Tset)
                    model.addConstr(follow_task >= y[v, j, t], name=f"2: follow_task_v{v}_j{j}_t{t}")

    # # Constraint 3                   
    # for v in tqdm(config.Vset, desc='Constraint 3'):
    #     for j in config.Jset:
    #         for t in config.Tset:
    #             for j_prime in taskF_results[(j, t)]:
    #                 for t_prime in range(t + mu_results[j], t + mu_results[j] + xi_results[(j, j_prime)]):
    #                     if t_prime in config.Tset and xi_results[(j, j_prime)] != 1:
    #                         model.addConstr(y[v, j, t] + y[v, j_prime, t_prime] <= 1 ,name=f"3: no_overlap_v{v}_j{j}_t{t}_j_prime{j_prime}_t_prime{t_prime}")

    # constraint 3, equivalent formulation
    #  nu sets
    
    nu = {}
    for t in tqdm(config.Tset, desc='Precomputing nu'):
        for j in config.Jset:
            for j_prime in config.Jset:
                end_station = functions['get_task_location'](config, j, -1) # end 
                start_station = functions['get_task_location'](config, j_prime, 0) # start 
                if end_station != start_station:
                    nu[(t, j, j_prime)] = [t_prime for t_prime in range(t + int(mu_results[j]), t + int(mu_results[j]) + int(xi_results[(j, j_prime)])) if t_prime in config.Tset] # if xi_results[(j, j_prime)] != 1
                elif end_station == start_station: # same station can start immidiately
<<<<<<< HEAD
                    nu[(t, j, j_prime)] = [] # if xi_results[(j, j_prime)] != 1
=======
                    nu[(t, j, j_prime)] = []
>>>>>>> 60cb21148b4d8c043b1960b69e735bbb819b4824
                else:
                    print('no!')


    # equivalent formulation
    for v in tqdm(config.Vset, desc='Constraint 3 equivalent formulation'):
        for j in config.Jset:
            for t in config.Tset:
                sum_y_v_jprime_tprime = gp.quicksum(y[v, j_prime, t_prime] 
                                                    for j_prime in config.Jset 
                                                    for t_prime in nu[(t, j, j_prime)])
                
                sum_nu_t_j_jprime = sum(len(nu[(t, j, j_prime)]) for j_prime in config.Jset) 
                model.addConstr(sum_y_v_jprime_tprime <= sum_nu_t_j_jprime * (1 - y[v, j, t]),name=f"3_eq_v{v}_j{j}_t{t}")

    # Constraint 4
    for w in tqdm(config.Wset, desc='Constraint 4'):
        for t in config.Tset:
            sum_yz = gp.quicksum(y[v, j, t_prime] * z[w, j] for v in config.Vset for (j, t_prime) in E_results[(w, t)])
            sum_Z_prime = gp.quicksum(Z_prime[l, w, t] for l in config.Lset if (l, w, t) in Z_prime)
            model.addConstr(sum_yz + sum_Z_prime <= functions['cal_Cw'](config, w), name=f"4: capacity_constraint_w{w}_t{t}")

    # # Constraint 5a
    # for v in tqdm(config.Vset, desc='Constraint 5a'):
    #     for t in config.Tset:
    #         model.addConstr(Q[v, t] >= 0, name=f"5a: battery_non_negative_v{v}_t{t}")


    # # Constraint 5b
    # for v in tqdm(config.Vset, desc='Constraint 5b'):
    #     for t in config.Tset:
    #         model.addConstr(Q[v, t] <= 1, name=f"5b: battery_max_capacity_v{v}_t{t}")

    # # Constraint 5c
    # rv = {v: vessel_df[vessel_df['Vessel code'] == v]['rv'].iloc[0] for v in config.Vset}
    # Qv0 = {v: vessel_df[vessel_df['Vessel code'] == v]['Qv0'].iloc[0] for v in config.Vset}

    # for v in tqdm(config.Vset, desc='Constraint 5c'):
    #     for t in config.Tset:
    #         if t == 1:
    #             model.addConstr(
    #                 Qv0[v]
    #                 + gp.quicksum(functions['cal_q'](config, v, j, t - t_prime) * y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)])
    #                 - rv[v] * (1 - gp.quicksum(y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)]))
    #                 >= Q[v, t],
    #                 name=f"5c: battery_update_v{v}_t{t}"
    #             )
    #         else:
    #             model.addConstr(
    #                 Q[v, t - 1]
    #                 + gp.quicksum(functions['cal_q'](config, v, j, t - t_prime) * y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)])
    #                 - rv[v] * (1 - gp.quicksum(y[v, j, t_prime] for j in config.Jset for t_prime in phi_results[(j, t)]))
    #                 >= Q[v, t],
    #                 name=f"5c: battery_update_v{v}_t{t}"
    #             )


    # # Constraint 6a
    # for v in tqdm(config.Vset, desc='Constraint 6a'):
    #     model.addConstr(
    #         gp.quicksum(y[v, j, t] for j in config.Bc for t in config.Tset) >= config.nc,
    #         name=f"6a: min_crew_pauses_{v}"
    #     )

    # # Constraint 6b
    # for v in tqdm(config.Vset, desc='Constraint 6b'):
    #     for t in config.Tset:
    #         if t < (config.Tset[-1] - (config.Tc // config.period_length + 1)):
    #             for t_prime in range(1, config.Tc // config.period_length + 1):
    #                 model.addConstr(
    #                     gp.quicksum(y[v, j, t + t_prime] for j in config.Bc) >= 1,
    #                     name=f"6b: max_distance_pauses_v{v}_t{t}_t_prime{t_prime}"
    #                 )

    print('All constraintrs are ready.\n')


