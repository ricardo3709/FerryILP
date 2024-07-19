import pandas as pd
import hashlib
import json
import pickle
from tqdm import tqdm

def cal_Cw(config, w):
    try:
        wharf_data = config.wharf_df[config.wharf_df['Wharf_No'] == w]
        if wharf_data.empty:
            raise ValueError(f"No data found for wharf {w}. Please check the wharf identifier.")
        loading_berths = wharf_data['Loading_berths'].iloc[0]
        non_loading_berths = wharf_data['Non_loading_berths'].iloc[0]
        if pd.isna(loading_berths) or pd.isna(non_loading_berths):
            raise ValueError(f"Missing berth information for wharf {w}.")
        total_capacity = int(loading_berths) + int(non_loading_berths)
        return total_capacity
    except KeyError as e:
        raise KeyError(f"Missing required columns in the dataframe: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

# def cal_Rl(config, l):
#     if not isinstance(l, int):
#         raise ValueError("Line number must be an integer.")
#     if l not in config.line_df['Line_No'].values:
#         raise ValueError(f"Line number {l} is not found in the DataFrame.")
#     try:
#         route_data = config.line_df[config.line_df['Line_No'] == l][['O', 'I', 'T']].iloc[0]
#         R_l = [station for station in route_data if station not in [None, 'None', '', ' ', pd.NA]]
#         return R_l
#     except IndexError:
#         raise ValueError(f"No data available for line number {l}.")
#     except KeyError:
#         raise ValueError("DataFrame must include 'Line_No', 'O', 'I', 'T' columns.")
def cal_Rl(config, l):
    if not isinstance(l, int):
        raise ValueError("Line number must be an integer.")
    if l not in config.line_df['Line_No'].values:
        raise ValueError(f"Line number {l} is not found in the DataFrame.")
    try:
        route_data = config.line_df[config.line_df['Line_No'] == l][['O', 'I', 'T']].iloc[0]
        R_l = [station for station in route_data if not pd.isna(station) and station not in [None, 'None', '', ' ']]
        return R_l
    except IndexError:
        raise ValueError(f"No data available for line number {l}.")
    except KeyError:
        raise ValueError("DataFrame must include 'Line_No', 'O', 'I', 'T' columns.")



def cal_C_lS(config, S):
    if not isinstance(S, str):
        raise ValueError("Station name must be a string.")
    if S not in config.wharf_df['Station'].values:
        raise ValueError(f"Station {S} is not found in the DataFrame.")
    try:
        C_lS = config.wharf_df[config.wharf_df['Station'] == S]['Wharf_No'].unique().tolist()
        return C_lS
    except KeyError:
        raise ValueError("DataFrame must include 'Station' and 'Wharf_No' columns.")

def cal_Sv(config, v):
    try:
        if not isinstance(v, str):
            raise ValueError("Vessel code must be a string.")
        if v not in config.vessel_df['Vessel code'].values:
            raise ValueError(f"Vessel code {v} is not found in the DataFrame.")
        S_v = config.vessel_df[config.vessel_df['Vessel code'] == v]['Sv'].iloc[0]
        return S_v
    except KeyError:
        raise ValueError("DataFrame must include 'Vessel code' and 'Sv' columns.")
    except IndexError:
        raise ValueError(f"No data available for vessel code {v}. The vessel might not be listed.")

def cal_li(config, v):
    try:
        if not isinstance(v, str):
            raise ValueError("Vessel code must be a string.")
        if v not in config.vessel_df['Vessel code'].values:
            raise ValueError(f"Vessel code {v} is not found in the DataFrame.")
        vessel_row = config.vessel_df[config.vessel_df['Vessel code'] == v].iloc[0]
        routes_served = [route for route in config.vessel_df.columns[2:-1] if vessel_row[route] == 'Yes']
        li_v = config.line_df[config.line_df['Route_No'].isin(routes_served)]['Line_No'].tolist()
        return li_v
    except KeyError:
        raise ValueError("DataFrame must include 'Vessel code', 'Route_No', 'O', and necessary route columns.")
    except IndexError:
        raise ValueError(f"No data available for vessel code {v}.")

def cal_D(config, l):
    try:
        first_sailing_time = config.line_df[config.line_df['Line_No'] == l]['First_sailing'].iloc[0]
        delta_minutes = (first_sailing_time.hour * 60 + first_sailing_time.minute) - (config.initial_time.hour * 60 + config.initial_time.minute)
        allowed_latitude = 15
        D_l = list(range((delta_minutes - allowed_latitude) // config.period_length + 1, ((delta_minutes + allowed_latitude) // config.period_length + 1) + 1))
        return D_l
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def cal_h(config, s, d, l):
    try:
        h = config.headway_df[f'h{l}'].dropna().tolist()
        h_sd_ls = [d]  # Start the list with the initial day 'd'
        for sailing_headway in h:
            num_time_period = int(sailing_headway // config.period_length + 1)  # round up
            h_sd_ls.append(h_sd_ls[-1] + num_time_period)
        if s-1 < len(h_sd_ls):
            return h_sd_ls[s-1]
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def cal_mu(config, j):
    try:
        if not isinstance(j, (int, str)):
            raise ValueError("Task identifier must be an integer or string.")
        if j in config.Lset:
            mu_j = config.line_df[config.line_df['Line_No'] == j]['Line_duration'].iloc()[0] // config.period_length + 1
        elif j in config.Bc:
            mu_j = config.Dc // config.period_length + 1
        elif j in config.Bplus or j in config.B:
            mu_j = 1
        else:
            return None
        return mu_j
    except KeyError as e:
        raise ValueError(f"Missing data for task {j}: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

def cal_q(config, v, j, t):
    if j in config.Bplus:
        return config.rv_plus
    elif j in config.B:
        epsilon = 1 - config.pc / config.period_length
        return epsilon * config.rv_plus
    elif j in config.B:
        return 0
    elif j in config.Lset:
        l = j
        line_data = config.line_df[config.line_df['Line_No'] == l]
        R_l = cal_Rl(config, l)
        stops = R_l[1:]  # remove the origin station
        if len(stops) == 1:
            a = int(line_data['Time_underway_to_T'].iloc()[0] // config.period_length + 1)
            dw = int(line_data['dw_T'].iloc()[0] // config.period_length + 1) 
            if t in range(a, a+dw+1):
                return 0
            else:
                return -line_data['rj'].iloc()[0]
        elif len(stops) == 2:
            a1 = int(line_data['Time_underway_to_I'].iloc()[0]) // config.period_length + 1
            dw1 = int(line_data['dw_I'].iloc()[0]) // config.period_length + 1
            a2 = int(line_data['Time_underway_to_T'].iloc()[0]) // config.period_length + 1
            dw2 = int(line_data['dw_T'].iloc()[0]) // config.period_length + 1
            if t in list(range(a1, a1+dw1+1)) + list(range(a2, a2+dw2+1)):
                return 0
            else:
                return -line_data['rj'].iloc()[0]
    else:
        return 0  # the vessel is rebalancing, rv will be captured by the constraint

def get_task_location(config, j, type):
    try:
        if type not in [0, -1]:
            raise ValueError("Type parameter must be 0 for start or -1 for end.")
        if j in config.Lset:
            task_stations = cal_Rl(config, j)
            if not task_stations:
                raise ValueError(f"No stations found for line {j}.")
            return task_stations[type]
        elif j in config.Bc or j in config.B or j in config.Bplus:
            task_wharf = j.split('_')[-1]
            task_station_df = config.wharf_df[config.wharf_df['Wharf_No'] == task_wharf]
            if task_station_df.empty:
                raise ValueError(f"No station found for wharf identifier {task_wharf}.")
            return task_station_df['Station'].iloc[0]
        else:
            raise ValueError(f"Task identifier {j} is unrecognized.")
    except Exception as e:
        raise Exception(f"An error occurred retrieving location for task {j}: {str(e)}")


def cal_xi(config, j1, j2):
    try:
        end_location_j1 = get_task_location(config, j1, -1)
        start_location_j2 = get_task_location(config, j2, 0)
        if end_location_j1 not in config.tt_df.columns or start_location_j2 not in config.tt_df.columns:
            return 24*60 + 1  # Very long time to ensure no rebalancing happens
        
        travel_time = config.tt_df.loc[end_location_j1, start_location_j2]  # 29July revised
        if pd.isna(travel_time): # 29July revised
            return 24*60 + 1# 29July revised
        else:  # 29July revised
            return travel_time // config.period_length + 1 # 29July revised        
    except Exception as e:
        raise Exception(f"An error occurred calculating travel time from {j1} to {j2}: {str(e)}")


def cal_xi0(config, v, j):
    try:
        S_v = cal_Sv(config, v)
        task_station = get_task_location(config, j, 0)
        if S_v == task_station:
            return 0
        if not S_v in config.tt_df.columns or not task_station in config.tt_df.columns:
            return 24*60+1  # Max value to avoid rebalancing
        travel_time = config.tt_df.loc[S_v, task_station] # 29July revised
        if pd.isna(travel_time): # 29July revised
            return 24*60 + 1# 29July revised 
        else: # 29July revised
            return travel_time // config.period_length + 1 # 29July revised
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

def cal_C(config, j):
    try:
        C_j = []
        if isinstance(j, int) and j in config.Lset:
            R_l = cal_Rl(config, j)
            for S in R_l:
                C_lS = cal_C_lS(config, S)
                C_j.extend(C_lS)
        elif isinstance(j, str) and (j in config.Bc or j in config.B or j in config.Bplus):
            C_j.append(j.split('_')[-1])
        else:
            raise ValueError(f"Task {j} is unrecognized.")
        return C_j
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

def cal_delta(config, j, w):
    try:
        if isinstance(j, int) and j in config.Lset:
            line_data = config.line_df[config.line_df['Line_No'] == j]
            safety_buffer = line_data['Safety_buffer'].iloc[0] // config.period_length + 1  # Safety buffer = O + B + OM， 29July revised
            R_l = cal_Rl(config, j)  # Stations visited by the line

            # Exclude the last station
            stations = R_l[1:-1]
            for station in stations:
                wharves = cal_C_lS(config, station)
                if w in wharves:
                    a = int(line_data['Time_underway_to_I'].iloc[0]) // config.period_length + 1
                    dw = int(line_data['dw_I'].iloc[0]) // config.period_length + 1
                    delta_j_w = [(w, time) for time in range(a - safety_buffer, a + dw + safety_buffer)]
                    return delta_j_w
            return []
        elif isinstance(j, str) and (j in config.Bc or j in config.B or j in config.Bplus):
            if w != j.split('_')[-1]:
                raise ValueError(f"Task {j} should occupy its own wharf {j.split('_')[-1]}, not {w}.")
            mu_j = cal_mu(config, j)
            return [(w, time) for time in range(mu_j)]
        else:
            raise ValueError(f"Task {j} is unrecognized.")
    except Exception as e:
        raise Exception(f"Unexpected error occurred: {str(e)}")

def cal_F(config, l):
    try:
        if not isinstance(l, int):
            raise ValueError("Line number must be an integer.")
        time_underway_to_T = config.line_df[config.line_df['Line_No'] == l]['Time_underway_to_T'].iloc[0]
        return time_underway_to_T // config.period_length + 1
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")

def cal_muF(config, l):
    try:
        if not isinstance(l, int):
            raise ValueError("Line number must be an integer.")
        dw_T = config.line_df[config.line_df['Line_No'] == l]['dw_T'].iloc[0] 
        safety_buffer = config.line_df[config.line_df['Line_No'] == l]['Safety_buffer'].iloc[0] # 29July revised
        return (dw_T + safety_buffer) // config.period_length + 1 # 29July revised
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")

def cal_phi(config, j, t):
    try:
        if not isinstance(t, int) or t < 1:
            raise ValueError("Time period t must be a positive integer.")
        mu_j = cal_mu(config, j)
        if mu_j is None:
            raise ValueError(f"No duration found for task {j}.")
        return list(range(max(1, t - mu_j + 1), t + 1))
    except Exception as e:
        raise ValueError(f"An error occurred calculating φ(j, t): {str(e)}")

def cal_f(config, j):
    try:
        if not config.Tset:
            raise ValueError("Tset is not defined or is empty.")
        
        last_period = config.Tset[-1]
        mu_j = cal_mu(config, j)
        last_start_time = last_period + 1 - mu_j
        return last_start_time
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def cal_G(config, j):
    try:
        G_j = []
        if j in config.Lset:
            headways = config.headway_df.get(f'h{j}', pd.Series()).dropna().tolist()
            if not headways:
                raise ValueError(f"No headway data available for line {j}")
            D_l = cal_D(config, j)
            for d in D_l:
                G_j.append(d)
                current_time = d
                for h in headways:
                    num_time_period = int(h // config.period_length + 1)
                    current_time += num_time_period
                    G_j.append(current_time)
        elif j in config.Bc or j in config.B or j in config.Bplus:
            G_j = config.Tset.copy()
        else:
            raise ValueError(f"Task {j} is unrecognized or not handled.")
        return G_j
    except Exception as e:
        raise ValueError(f"An error occurred processing task {j}: {str(e)}")


def cal_H(config, v, j):
    try:
        S_v = cal_Sv(config, v)
        if pd.isna(S_v):
            print(f'The start point of the vessel {v} has not been determined. Please determine it first.')
            return None

        xi0_vj = cal_xi0(config, v, j)
        f_j = cal_f(config, j)
        G_j = cal_G(config, j)

        H_vj = [t for t in G_j if xi0_vj <= t <= f_j]
        return H_vj
    except Exception as e:
        print(f"An error occurred while calculating H(v, j): {str(e)}")
        return None

def cal_taskF(config, j, t):
    try:
        if not isinstance(t, int) or t < 0:
            raise ValueError("Start time 't' must be a non-negative integer.")

        feasible_tasks = []
        f_j = cal_f(config, j)

        if t > f_j:
            return feasible_tasks

        mu_j = cal_mu(config, j)

        for j_prime in config.Jset:
            f_j_prime = cal_f(config, j_prime)
            xi_j_j_prime = cal_xi(config, j, j_prime)

            if f_j_prime >= t + mu_j + xi_j_j_prime:
                feasible_tasks.append(j_prime)

        return feasible_tasks

    except KeyError as e:
        raise KeyError(f"Missing data for task calculation: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

def cal_E(config, w, t):
    E_wt = []

    for j in config.Jset:
        C_j = cal_C(config, j)
        if w in C_j:
            delta_jw = cal_delta(config, j, w)
            for t_prime in [time for time in config.Tset if time <= t]:
                if (t - t_prime) in [usage[1] for usage in delta_jw]:
                    E_wt.append((j, t_prime))

    return E_wt


# pkl files related
def hash_config(config):
    config_dict = {
        'period_length': config.period_length,
        'Tset': config.Tset,
        'Lset': config.Lset,
        'Bc': config.Bc,
        'B': config.B,
        'Bplus': config.Bplus,
        'Dc': config.Dc,
        'rv_plus': config.rv_plus,
        'pc': config.pc,
        'Jset': config.Jset,
        'initial_time': str(config.initial_time)
    }
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_json.encode()).hexdigest()

def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def calculate_and_save_results(config, file_prefix):
    # Calculate results
    taskF_results = {}
    for j in tqdm(config.Jset, desc='taskF_results'):
        for t in config.Tset:
            taskF_results[(j, t)] = cal_taskF(config, j, t)

    mu_results = {}
    for j in tqdm(config.Jset, desc='mu_results'):
        mu_results[j] = cal_mu(config, j)

    xi_results = {}
    for j in tqdm(config.Jset, desc='xi_results'):
        for j_prime in config.Jset:
            xi_results[(j, j_prime)] = cal_xi(config, j, j_prime)

    phi_results = {}
    for j in tqdm(config.Jset, desc='phi_results'):
        for t in config.Tset:
            phi_results[(j, t)] = cal_phi(config, j, t)

    E_results = {}
    for w in tqdm(config.Wset, desc='E_results'):
        for t in config.Tset:
            E_results[(w, t)] = cal_E(config, w, t)

    # Save results with prefix
    save_results(taskF_results, f'ILPimplementation/pkl_files/{file_prefix}_taskF_results.pkl')
    save_results(mu_results, f'ILPimplementation/pkl_files/{file_prefix}_mu_results.pkl')
    save_results(xi_results, f'ILPimplementation/pkl_files/{file_prefix}_xi_jj_results.pkl')
    save_results(phi_results, f'ILPimplementation/pkl_files/{file_prefix}_phi_results.pkl')
    save_results(E_results, f'ILPimplementation/pkl_files/{file_prefix}_E_results.pkl')

    print(f'All results matrices have been generated and saved with prefix "{file_prefix}".\n')

def load_all_results(file_prefix):
    taskF_results = load_results(f'ILPimplementation/pkl_files/{file_prefix}_taskF_results.pkl')
    mu_results = load_results(f'ILPimplementation/pkl_files/{file_prefix}_mu_results.pkl')
    xi_results = load_results(f'ILPimplementation/pkl_files/{file_prefix}_xi_jj_results.pkl')
    phi_results = load_results(f'ILPimplementation/pkl_files/{file_prefix}_phi_results.pkl')
    E_results = load_results(f'ILPimplementation/pkl_files/{file_prefix}_E_results.pkl')

    print(f'All results matrices have been loaded from files with prefix "{file_prefix}".\n')
    
    return taskF_results, mu_results, xi_results, phi_results, E_results

def manage_results(config, generate_new_files, file_prefix):
    if generate_new_files:
        calculate_and_save_results(config, file_prefix)
        return load_all_results(file_prefix)
    else:
        return load_all_results(file_prefix)