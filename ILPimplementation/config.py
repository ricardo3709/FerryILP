from datetime import time
from data_load import *

#=================================== Parameters ==================================
# Versioning & File Names 
current_version = 'v8'

# Starting files
starting_version = 'version8.0'
prefix = 'v8'

# Solver Settings 
Gap = 0.2  # Optimality gap tolerance
TimeLimit =  22 * 60 * 60  # in seconds
alpha = 0.5  # percentage of the objectives that focus on fleet size

# pkl files
generate_new_files = False
pkl_file_prefix = 'v8'

# File Paths  
# in the order of [x, y, Q, z, Z, Z_prime]
files = {'x': f'ILPimplementation/output_files/{starting_version}/{prefix}_x_ld_results.csv',
         'y': f'ILPimplementation/output_files/{starting_version}/{prefix}_y_vjt_results.csv',
         'Q': f'ILPimplementation/output_files/{starting_version}/{prefix}_Q_vt_results.csv',
        'z': f'ILPimplementation/output_files/{starting_version}/{prefix}_z_wj_results.csv',
         'Z': f'ILPimplementation/output_files/{starting_version}/{prefix}_Z_lwt_results.csv',
         'Z_prime': f'ILPimplementation/output_files/{starting_version}/{prefix}_Z_prime_lwt_results.csv'}

# Simulation Time Parameters 
initial_time = time(5, 30)
period_length = 5  # minutes
total_operation_hours = 6  # hours

# Crew Break Parameters 
nc = 1  # Minimum number of crew breaks
Dc = 30  # Crew break duration (minutes)
Tc = 6 * 60  # Maximum separation time for crew breaks (minutes)

# Charging Parameters 
rv_plus = 0.16 # 0.16  # Charging rate (as percentage per period)
pc = 1  # Plugging/Unplugging time (minutes)

# Print Simulation Configuration 
print(f"""
====================================================================
Ferry ILP Simulation Configuration
====================================================================
Time Settings:
   - Initial Time: {initial_time}
   - Period Length: {period_length} minutes
   - Total Operation Hours: {total_operation_hours} hours

Crew Break Parameters:
   - Minimum Number of Crew Breaks (nc): {nc}
   - Crew Break Duration (Dc): {Dc} minutes
   - Max Separation Time for Breaks (Tc): {Tc} minutes

Charging Parameters:
   - Charging Rate (rv_plus): {rv_plus:.2%} per period
   - Plugging/Unplugging Time (pc): {pc} minutes

Data Handling:
   - Generate New Files: {generate_new_files}
   - Output File Prefix: {pkl_file_prefix}

Versioning:
   - Current Version: {current_version}
   - Based on: {starting_version}

====================================================================
""")


# -------------------- Sets Defined -----------------------------------------

# Lset: Set of Lines
Lset = line_df['Line_No'].unique().tolist()

# Tset: Set of Time Periods
Tset = [i for i in range(1, total_operation_hours * 60 // period_length + 1)]


# BERTHS
# B+, set of wharves to charge
Bplus = [wharf for wharf in charging_berth['Wharf_No'].unique().tolist()]
# Non-loading berths
original_non_loading_berths = wharf_df[wharf_df['Non_loading_berths'] != 0]['Wharf_No'].unique().tolist()
# Bc, set of wharves to crew pause, is a copy of set B
Bc = ['cp_' + wharf for wharf in original_non_loading_berths]
# B, set of wharves to wait, any wharf with a charger belongs to B, and B contains wharves with original non-loading berths
B =  wharf_df['Wharf_No'].unique().tolist() # full Bset
for wharf in Bplus:
    if wharf in B:  # wharf with a charger
        B.remove(wharf)
        B.append(f'phi_{wharf}')  # mark as phi(w)
    else:
        B.append(f'phi_{wharf}')  # input directly

# TASKS
Jset = [ele for ele in Lset + B + Bc + Bplus]


# SAILINGS
# Zset: Set of Sailing
nl_ls = [len(headway_df[f"h{l}"].dropna().tolist()) + 1 for l in Lset]
s_ls = [list(range(1, nl + 1)) for nl in nl_ls]
Zset = [f'{line}_{sailing}' for line in Lset for sailing in s_ls[line - 1]]


# Vset: Set of Vessels
Vset = vessel_df['Vessel code'].unique().tolist()

# Wset: Set of Wharves
Wset = wharf_df['Wharf_No'].unique().tolist()

# Dset: Set of possible first sailing time
Dset = {l: list(range(
        ((line_df[line_df['Line_No'] == l]['First_sailing'].iloc[0].hour * 60 +
          line_df[line_df['Line_No'] == l]['First_sailing'].iloc[0].minute) -
         (initial_time.hour * 60 + initial_time.minute) - 15) // period_length + 1,
        ((line_df[line_df['Line_No'] == l]['First_sailing'].iloc[0].hour * 60 +
          line_df[line_df['Line_No'] == l]['First_sailing'].iloc[0].minute) -
         (initial_time.hour * 60 + initial_time.minute) + 15) // period_length + 1 + 1))
    for l in Lset
}

print(f'B+, set of wharves to charge:{Bplus}')
print(f'Bc, set of wharves to crew pause:{Bc}')
print(f'B, set of wharves to wait:{B}')

print('Vset, Wset, Tset, Jset, and Dset have been defined.\n')
