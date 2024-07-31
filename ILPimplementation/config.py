from datetime import time
from data_load import *

## -------------------- Input Parameters Values --------------------------

# Simulation time parameters
initial_time = time(5,30)
period_length = 5  # mins
total_operation_hours = 6  # hours

# nc, Minimum num of crew breaks
nc = 1

# Dc, Crew break duration (fixed)
Dc = 9  # mins

# Tc, Maximum separation time for crew breaks
Tc = 6 * 60  # mins

# rv+, charging rate
rv_plus = 2100 * period_length / 60 / 1100  # kW*h/kWh --> %

# pc, Plugging/Unplugging time
pc = 2  # mins

print(f"""
Simulation Parameters:
------------------------------------------------------
Initial Time: {initial_time}
Period Length: {period_length} minutes
Total Operation Hours: {total_operation_hours} hours

Minimum Number of Crew Breaks (nc): {nc}
Crew Break Duration (Dc): {Dc} minutes
Maximum Separation Time for Crew Breaks (Tc): {Tc} minutes

Charging Rate (rv_plus): {rv_plus:.2%} per period
Plugging/Unplugging Time (pc): {pc} minutes
------------------------------------------------------
""")
## -------------------- Sets Defined -----------------------------------------

# Lset: Set of Lines
Lset = line_df['Line_No'].unique().tolist()

# Tset: Set of Time Periods
Tset = [i for i in range(1, total_operation_hours * 60 // period_length + 1)]

# B+, set of wharves to charge
Bplus = [wharf for wharf in charging_berth['Wharf_No'].unique().tolist()]

# Non-loading berths
original_non_loading_berths = wharf_df[wharf_df['Non_loading_berths'] != 0]['Wharf_No'].unique().tolist()

# Bc, set of wharves to crew pause, is a copy of set B
Bc = ['cp_' + wharf for wharf in original_non_loading_berths]

# B, set of wharves to wait, any wharf with a charger belongs to B, and B contains wharves with original non-loading berths
B =  wharf_df['Wharf_No'].unique().tolist() # original_non_loading_berths.copy()
for wharf in Bplus:
    if wharf in B:  # wharf with a charger
        B.remove(wharf)
        B.append(f'phi_{wharf}')  # mark as phi(w)
    else:
        B.append(f'phi_{wharf}')  # input directly

Jset = [ele for ele in Lset + B + Bc + Bplus]

# Zset: Set of Sailing
nl_ls = [len(headway_df[f"h{l}"].dropna().tolist()) + 1 for l in Lset]
s_ls = [list(range(1, nl + 1)) for nl in nl_ls]
Zset = [f'{line}_{sailing}' for line in Lset for sailing in s_ls[line - 1]]

# Vset: Set of Vessels
Vset = vessel_df['Vessel code'].unique().tolist()

# Wset: Set of Wharves
Wset = wharf_df['Wharf_No'].unique().tolist()

# Dset: Set of possible first sailing time
Dset = {
    l: list(range(
        ((line_df[line_df['Line_No'] == l]['First_sailing'].iloc[0].hour * 60 +
          line_df[line_df['Line_No'] == l]['First_sailing'].iloc[0].minute) -
         (initial_time.hour * 60 + initial_time.minute) - 15) // period_length + 1,
        ((line_df[line_df['Line_No'] == l]['First_sailing'].iloc[0].hour * 60 +
          line_df[line_df['Line_No'] == l]['First_sailing'].iloc[0].minute) -
         (initial_time.hour * 60 + initial_time.minute) + 15) // period_length + 1 + 1))
    for l in Lset
}

print('Vset, Wset, Tset, Jset, and Dset have been defined.\n')
