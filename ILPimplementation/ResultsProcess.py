import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from config import *
from functions import *
from simulation_config import SimulationConfig  # Import the class
import os

version = input("Please enter the version name (e.g., versionXXX): ").strip()


# Initialize the configuration
functions = {
    'cal_C': cal_C,
    'cal_Cw': cal_Cw,
    'cal_C_lS': cal_C_lS,
    'cal_F': cal_F,
    'cal_H': cal_H,
    'cal_Rl': cal_Rl,
    'cal_h': cal_h,
    'cal_li': cal_li,
    'cal_muF': cal_muF,
    'cal_xi0': cal_xi0,
    'cal_taskF': cal_taskF,
    'cal_mu': cal_mu,
    'cal_xi': cal_xi,
    'cal_phi': cal_phi,
    'cal_E': cal_E,
    'cal_q': cal_q,
}

config = SimulationConfig(
    wharf_df, line_df, headway_df, tt_df, vessel_df,
    initial_time, period_length, Tset,
    Lset, Bc, B, Bplus, Jset, Wset, Dset, Vset, Zset,
    Dc, nc, Tc, 
    rv_plus, pc, functions)

# --------------------------------- Load results file --------------------------------------------


def load_and_process_data(filepath, split_columns, value_columns):
    """
    Load data, remove unwanted characters, filter rows, split columns, and convert types.

    Args:
    filepath (str): Path to the CSV file.
    split_columns (list): List of columns names after splitting the 'Variable' column.
    value_columns (dict): Dictionary of columns to convert with their respective types.
    """
    df = pd.read_csv(filepath)
    df = df[df['Value'] == 1]
    # df['Variable'] = df['Variable'].str.replace(r"[()']", "", regex=True)
    df['Variable'] = df['Variable'].str.replace(r"[()' ]", "", regex=True)
    df[split_columns] = df['Variable'].str.split(',', expand=True)
    for column, dtype in value_columns.items():
        df[column] = df[column].astype(dtype)
    return df



z_df = load_and_process_data(f'ILPimplementation/output_files/{file_prefix}_z_wj_results.csv',['Wharf', 'Task'],{})
Zp_df = load_and_process_data(f'ILPimplementation/output_files/{file_prefix}_Z_prime_lwt_results.csv',['Line', 'Wharf', 'Time'],{'Line': int, 'Time': int})
Z_df = load_and_process_data(f'ILPimplementation/output_files/{file_prefix}_Z_lwt_results.csv',['Line', 'Wharf', 'Time'],{'Line': int, 'Time': int})
x_df = load_and_process_data(f'ILPimplementation/output_files/{file_prefix}_x_ld_results.csv',['Line', 'Time'],{'Line': int, 'Time': int})
y_df = load_and_process_data(f'ILPimplementation/output_files/{file_prefix}_y_vjt_results.csv',['Vessel', 'Task', 'Start_Time'],{'Start_Time': int})

# Determine start and end wharfs
Start_S = dict(zip(line_df['Line_No'].astype(str), line_df['O']))
Start_S.update(dict(zip(wharf_df['Wharf_No'], wharf_df['Station'])))
End_S = dict(zip(line_df['Line_No'].astype(str), line_df['T']))
End_S.update(dict(zip(wharf_df['Wharf_No'], wharf_df['Station'])))

linels = line_df['Line_No'].unique().tolist()
Start_wharf = {}

for line in linels:
    Start_wharf[line] = z_df[z_df['Task'] == str(line)]['Wharf'].iloc[0]

# {line: z_df[z_df['Task'] == line]['Wharf'].iloc[0] for line in z_df['Line'].unique()}

line_to_route_dict = {
    '1': 'F2 - Taronga Zoo (off peak)',
    '2': 'F4 - Pyrmont Bay',
    '3': 'F5 - Neutral Bay',
    '4': 'F6 - Mosman (off peak)',
    '5': 'F7 - Double Bay',
    '6': 'F8 - Cockatoo Island',
    '7': 'F9 - Rose Bay',
    '8': 'F9 - Watsons Bay',
    '9': 'F11 - Blackwattle Bay',
    '10': 'F2 - Zoo/Mosman (Peak)',
    '11': 'F6 - Mosman (peak)'}

Lset = [str(l) for l in line_df['Line_No'].unique().tolist()]
# Lset = [str(l) for l in Lset]
Vset = vessel_df['Vessel code'].unique().tolist()
rv = {v: vessel_df[vessel_df['Vessel code'] == v]['rv'].iloc[0] for v in Vset}

# -------------------------------------------- Additional functions for processing --------------------------------------------
def cal_time(period_num):
    # Convert initial_time to a datetime object with today's date
    initial_datetime = datetime.combine(datetime.today(), initial_time)
    
    # Calculate the total minutes to add
    added_time = timedelta(minutes=int((period_num - 1) * period_length))
    new_time = initial_datetime + added_time

    return new_time

def cal_duration(minutes):
    result = (minutes // 5) + (1 if minutes % 5 != 0 else 0)
    return int(result)

def start_wharf(task):
    if task.isdigit():
        return Start_wharf[int(task)]
    return task

def end_wharf(row):
    wharf = row['End_Wharf']
    if row['Task'].isdigit():
        line = int(row['Task'])
        timetoT = int(line_df[line_df['Line_No'] == line]['Time_underway_to_T'].iloc()[0])
        end_time = row['Start_Time'] + cal_duration(timetoT)
        matching_rows = Zp_df[(Zp_df['Line'] == line) & (Zp_df['Time'] == end_time)]
        if not matching_rows.empty:
            wharf = matching_rows['Wharf'].iloc()[0]
    return wharf

def end_time(row):
    if row['Task'].isdigit():
        line = int(row['Task'])
        duration = int(line_df[line_df['Line_No'] == line]['Line_duration'].iloc()[0])
        return (cal_time(row['Start_Time']) + timedelta(minutes=duration)).strftime('%H:%M')
    elif row['Task'] in ['Waiting', 'Charging']:
        return (cal_time(row['Start_Time']) + timedelta(minutes=5)).strftime('%H:%M')
    else:
        return (cal_time(row['Start_Time']) + timedelta(minutes=Dc)).strftime('%H:%M')

Lset = line_df['Line_No'].unique().tolist()
Lset = [str(l) for l in Lset]
Vset = vessel_df['Vessel code'].unique().tolist()
rv = {v: vessel_df[vessel_df['Vessel code'] == v]['rv'].iloc[0] for v in Vset}

def cal_d_q(j):
    if j in Bc: # crew pause
        return 0  # returen change and time period

    elif j in Bplus: # Charging
        return rv_plus
    
    elif j in B :
        if 'phi_' in j: # fisr/last period of charging
            epsilon = 1 - pc / period_length
            return epsilon * rv_plus
        else:   # waiting   
            return 0
        
    elif j in Lset: # line task
        return line_df[line_df['Line_No'] == int(j)]['rj'].iloc()[0]
    
    else: # rebalancing
        return None
        # return rv[v] 
# -------------------------------------------- Generate Timetable --------------------------------------------
lines = line_df['Line_No'].unique().tolist()
solved_lines = x_df['Line'].unique().tolist()

def cal_timetable(line):
    # Check if the line exists and is solved
    if line in lines and line in solved_lines:
        # First sailing time and headways
        d = x_df.loc[x_df['Line'] == line, 'Time'].iloc[0]
        headways = headway_df[f'h{line}'].dropna().tolist()

        times, locs, wharfs = [], [], []
        
        # Calculate timings for all sailings including first and subsequent ones
        for s in range(len(headways)+1):
            period = cal_h(config, s+1, d, line)  # Assuming cal_h is defined somewhere
            sailing_time = cal_time(period)
            
            # Add sailing time, location and wharf
            times.append(sailing_time)
            locs.append(Start_S[str(line)])
            wharfs.append(z_df[z_df['Task'] == str(line)]['Wharf'].iloc[0])

            # Intermediate Stop
            Intermediate_stop = line_df[line_df['Line_No'] == line]['I'].iloc[0]
            if pd.notna(Intermediate_stop):
                timetoI = int(line_df[line_df['Line_No'] == line]['Time_underway_to_I'].iloc[0])
                arrival_time_I = sailing_time + timedelta(minutes=timetoI)
                times.append(arrival_time_I)
                locs.append(Intermediate_stop)
                wharfs.append(z_df[z_df['Task'] == str(line)]['Wharf'].iloc[1])
            
            # Terminal Stop
            timetoT = int(line_df[line_df['Line_No'] == line]['Time_underway_to_T'].iloc[0])
            arrival_time_T = sailing_time + timedelta(minutes=timetoT)
            times.append(arrival_time_T)
            locs.append(End_S[str(line)])

            filtered_df = Zp_df[(Zp_df['Line'] == line) & (Zp_df['Time'] == period + cal_duration(timetoT))]

            # Check if filtered_df is not empty before accessing .iloc[0]
            if not filtered_df.empty:
                wharfs.append(filtered_df['Wharf'].iloc[0])
            else:
                print(f"No matching data found for line {line} and time {period + cal_duration(timetoT)}")
                # Handle the case when no match is found, e.g., append a placeholder or skip
                wharfs.append('No Wharf Found')  # or any other placeholder

            # wharfs.append(Zp_df[(Zp_df['Line'] == line) & (Zp_df['Time'] == period + timetoT // period_length + 1)]['Wharf'].iloc[0])

        # Format times and create the DataFrame
        formatted_times = [time.strftime('%H:%M') for time in times]
        timetable = pd.DataFrame({
            'Time': formatted_times,
            'Station': locs,
            'Wharf': wharfs
        })
        return timetable
    
    else:
        print('Line not exist or unsolved.')
        return None

# for line in lines:
#     cal_timetable(line).to_csv(f'ILPimplementation/timetables/line_{line}_timetable.csv', index=False)

versioned_folder = os.path.join('ILPimplementation', 'timetables', version) #
os.makedirs(versioned_folder, exist_ok=True)#

# Iterate over vessels and save the itinerary files in the versioned folder
for line in lines:
    file_path = os.path.join(versioned_folder, f'line_{line}_timetable.csv')
    cal_timetable(line).to_csv(file_path, index=False)

print(f"Timetable files have been saved in the folder: {versioned_folder}")


# -------------------------------------------- Generate vessel itinerary --------------------------------------------
def cal_itinerary(vessel):
    vessel_itinerary_df = y_df.copy()
    # Extract and clean up Wharf details
    vessel_itinerary_df['Start_Wharf'] = vessel_itinerary_df['Task'].apply(lambda x: x.split('_')[-1].strip())
    vessel_itinerary_df['End_Wharf'] = vessel_itinerary_df['Start_Wharf']

    # Lookup Start and End Stations based on Wharfs
    vessel_itinerary_df['Start_Station'] = vessel_itinerary_df['Start_Wharf'].apply(lambda x: Start_S.get(x, 'Unknown Station'))
    vessel_itinerary_df['End_Station'] = vessel_itinerary_df['Start_Wharf'].apply(lambda x: End_S.get(x, 'Unknown Station'))

    # # Update Task based on specific keywords or conditions
    # vessel_itinerary_df['Task'] = vessel_itinerary_df['Task'].apply(lambda x: 'Waiting' if x in B else
    #                                                                           'Crew Break' if x in Bc else
    #                                                                           'Charging' if x in Bplus else
    #                                                                           f"{x}")


    vessel_itinerary_df['Task'] = vessel_itinerary_df['Task'].apply(lambda x: 'Waiting' if x in B and not x.startswith('phi_') else
                                                                              'Charging' if x in B and x.startswith('phi_') else
                                                                              'Crew Break' if x in Bc else
                                                                              'Charging' if x in Bplus else
                                                                              f"{x}")

    # Initial transformations for Start_Wharf and End_Wharf 
    vessel_itinerary_df['Start_Wharf'] = vessel_itinerary_df['Start_Wharf'].apply(start_wharf)
    vessel_itinerary_df['End_Wharf'] = vessel_itinerary_df['Start_Wharf']

    vessel_itinerary_df['End_Wharf'] = vessel_itinerary_df.apply(end_wharf, axis=1)
    vessel_itinerary_df['End_Time'] = vessel_itinerary_df.apply(end_time, axis=1)


    # Reorganize DataFrame columns for final output
    vessel_itinerary_df = vessel_itinerary_df[['Vessel', 'Task', 'Start_Station', 'Start_Wharf', 'Start_Time', 'End_Station', 'End_Wharf', 'End_Time']]

    # Function to calculate vessel itinerary

    itinerary = vessel_itinerary_df[vessel_itinerary_df['Vessel'] == vessel].sort_values('Start_Time')
    itinerary['Start_Time'] = itinerary['Start_Time'].apply(lambda x: cal_time(x).strftime('%H:%M'))
    itinerary.reset_index(inplace=True, drop=True)

    itinerary['Task'] = itinerary['Task'].replace(line_to_route_dict)

    return itinerary


vessels = vessel_df['Vessel code'].unique().tolist()

# Create the versioned directory if it doesn't exist
versioned_folder = os.path.join('ILPimplementation', 'vessel_itineraries', version)
os.makedirs(versioned_folder, exist_ok=True)

# Iterate over vessels and save the itinerary files in the versioned folder
for vessel in vessels:
    file_path = os.path.join(versioned_folder, f'vessel_{vessel}_itinerary.csv')
    cal_itinerary(vessel).to_csv(file_path, index=False)

print(f"Vessel files have been saved in the folder: {versioned_folder}")

# -------------------------------------------- Generate wharf utilization --------------------------------------------
def cal_wharf_utilization(wharf):
    # non line task
    non_line_tasks = y_df[~y_df['Task'].str.match(r'^\d+$')]
    data = []

    for _, row in non_line_tasks.iterrows():
        v = row['Vessel']
        j = row['Task']
        w = row['Task'].split('_')[-1]
        t = row['Start_Time'] # "12:00"
        
        start = cal_time(t)
        if j in Bc: # crew break
            end = start + timedelta(minutes=Dc)
        elif j in Bplus or j in B: # charging
            end = start + timedelta(minutes=5)
        else:
            print(f'{v},{j},{w},{t}: Error')

            # Add data for origin
        data.append({
            "Vessel": v,
            "Task": j,
            "Wharf": w,
            "Time": f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
        })

    # line task
    line_tasks = y_df[y_df['Task'].str.match(r'^\d+$')]
    for _, row in line_tasks.iterrows():
        v = row['Vessel']  # Vessel name
        j = int(row['Task'])  # Task number (Line_No)
        t = row['Start_Time']  # Start time (assuming it's in a compatible format)

        # Origin
        # w_start = line_df[line_df['Line_No'] == j]['O'].iloc[0]
        w_start = z_df[z_df['Task'] == str(j)]['Wharf'].iloc[0]
        dwell_O = int(line_df[line_df['Line_No'] == j]['dw_O'].iloc[0])

        start_O = cal_time(t)
        end_O = start_O + timedelta(minutes=dwell_O)

        # Add data for origin
        data.append({
            "Vessel": v,
            "Task": j,
            "Wharf": w_start,
            "Time": f"{start_O.strftime('%H:%M')}-{end_O.strftime('%H:%M')}"
        })

        # Intermediate Stop
        intermediate_station = line_df[line_df['Line_No'] == j]['I']
        if pd.notna(intermediate_station).any():
            w_intermediate = z_df[z_df['Task'] == str(j)]['Wharf'].iloc[1]  # Wharf
            time_to_I = int(line_df[line_df['Line_No'] == j]['Time_underway_to_I'].iloc[0])
            dwell_I = int(line_df[line_df['Line_No'] == j]['dw_I'].iloc[0])

            start_I = start_O + timedelta(minutes=time_to_I)
            end_I = start_I + timedelta(minutes=dwell_I)

            data.append({
                "Vessel": v,
                "Task": j,
                "Wharf": w_intermediate,
                "Time": f"{start_I.strftime('%H:%M')}-{end_I.strftime('%H:%M')}"
            })

        # Terminus
        time_to_T = int(line_df[line_df['Line_No'] == j]['Time_underway_to_T'].iloc[0])
        dwell_T = int(line_df[line_df['Line_No'] == j]['dw_T'].iloc[0])

        start_T = start_O + timedelta(minutes=time_to_T)
        end_T = start_T + timedelta(minutes=dwell_T)

        filtered_wharfs = Zp_df[(Zp_df['Line'] == j) & (Zp_df['Time'] == t + cal_duration(time_to_T))]['Wharf'].unique().tolist()
        if filtered_wharfs:
            w_end = filtered_wharfs[0]
        else:
            print(f"No matching wharf found for line {j} at time {t + cal_duration(time_to_T)}")
            w_end = None  # Handle missing value appropriately

        data.append({
            "Vessel": v,
            "Task": j,
            "Wharf": w_end,
            "Time": f"{start_T.strftime('%H:%M')}-{end_T.strftime('%H:%M')}"
        })

    all_wharf_df = pd.DataFrame(data)

    wharf_df = all_wharf_df[all_wharf_df['Wharf'] == wharf].copy()

    wharf_df['Task'] = wharf_df['Task'].apply(lambda x: 'Waiting' if x in B and not x.startswith('phi_') else
                                                        'Charging' if x in B and x.startswith('phi_') else
                                                        'Crew Break' if x in Bc else
                                                        'Charging' if x in Bplus else
                                                        f"{x}")



    wharf_df.reset_index(drop=True)
    wharf_df['Task'] = wharf_df['Task'].replace(line_to_route_dict)
    wharf_df = wharf_df.sort_values('Time')
    return wharf_df


wharfs = wharf_df['Wharf_No'].unique().tolist()

# Create the versioned directory if it doesn't exist
versioned_folder = os.path.join('ILPimplementation', 'wharf_utilizations', version)
os.makedirs(versioned_folder, exist_ok=True)

# Iterate over vessels and save the itinerary files in the versioned folder
for wharf in wharfs:
    file_path = os.path.join(versioned_folder, f'wharf_{wharf}_utilization.csv')
    cal_wharf_utilization(wharf).to_csv(file_path, index=False)

print(f"Wharf files have been saved in the folder: {versioned_folder}")


# -------------------------------------------- Generate battery change --------------------------------------------

def cal_battery_change(vessel):
    vesseldf = y_df[y_df['Vessel'] == vessel][['Vessel','Task','Start_Time']].copy()
    vesseldf.sort_values(by='Start_Time') 

    vesseldf['Battery_Change'] = vesseldf.apply(lambda row: cal_d_q(row['Task']), axis=1)
    filled_times = pd.DataFrame({'Start_Time': list(range(1, 73))})
    vesseldf = filled_times.merge(vesseldf, on='Start_Time', how='left')
    vesseldf

    # Handle the condition where Task is in Bc
    for idx in vesseldf.index:
        if vesseldf.loc[idx, 'Task'] in Bc: # 
            # Set Battery_Change to 0 for the next 6 time periods (if within bounds)
            vesseldf.loc[idx:idx + cal_duration(Dc) -1, 'Battery_Change'] = 0
            vesseldf.loc[idx:idx + cal_duration(Dc) -1, 'Task'] = vesseldf.loc[idx, 'Task'] + ' cont.'

            # vesseldf.loc[idx:idx + cal_duration(Dc), 'Task'] = vesseldf.loc[idx, 'Task']

    for idx in vesseldf.index:
        if vesseldf.loc[idx, 'Task'] in Lset:
            time = cal_duration(line_df[line_df['Line_No'] == int(vesseldf.loc[idx, 'Task'])]['Line_duration'].iloc[0])
            battery_value = line_df[line_df['Line_No'] == int(vesseldf.loc[idx, 'Task'])]['rj'].iloc[0]
            vesseldf.loc[idx:idx + time - 1, 'Battery_Change'] = battery_value
            vesseldf.loc[idx:idx + time - 1, 'Task'] = vesseldf.loc[idx, 'Task'] + ' cont.'


    vesseldf['Task'] = vesseldf['Task'].fillna("Rebalancing")
    vesseldf['Vessel'] = vesseldf['Vessel'].fillna(vessel)
    vesseldf['Battery_Change'] = vesseldf['Battery_Change'].fillna(rv[vessel])
    vesseldf

    # Initialize battery level column
    vesseldf['Battery_Level'] = 0  # Assuming battery starts at 0

    # Set initial battery level
    initial_battery_level = 1
    max_battery_level = 1.0

    # Calculate battery levels with constraints
    current_battery_level = initial_battery_level

    for idx in vesseldf.index:
        # Update battery level based on the previous level and the current change
        current_battery_level += vesseldf.loc[idx, 'Battery_Change']
        
        # Apply constraints
        if current_battery_level > max_battery_level:
            current_battery_level = max_battery_level

        # Save the calculated battery level
        vesseldf.loc[idx, 'Battery_Level'] = current_battery_level

    initial_row = pd.DataFrame({'Vessel': [vessel],'Start_Time': [0],'Task': [None],'Battery_Change': [0],'Battery_Level': [1]})
    vesseldf = pd.concat([initial_row, vesseldf], ignore_index=True)

    vesseldf['Time'] = vesseldf['Start_Time'].apply(lambda x: cal_time(x+1).strftime('%H:%M'))

    return vesseldf[['Task','Time','Battery_Level']]


# Create the versioned directory if it doesn't exist
versioned_folder = os.path.join('ILPimplementation', 'battery_change', version)
os.makedirs(versioned_folder, exist_ok=True)

# Iterate over vessels and save the itinerary files in the versioned folder
for vessel in vessels:
    file_path = os.path.join(versioned_folder, f'vessel_{vessel}_battery.csv')
    cal_battery_change(vessel).to_csv(file_path, index=False)

print(f"Battery files have been saved in the folder: {versioned_folder}")
