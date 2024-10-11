import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from config import *
from functions import *
from simulation_config import SimulationConfig  # Import the class


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

# file_prefix = "6htest_cyclelines_"
# file_prefix = config.file_prefix # "6htest_new_cyclelines"

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

for line in lines:
    cal_timetable(line).to_csv(f'ILPimplementation/timetables/line_{line}_timetable.csv', index=False)


# -------------------------------------------- Generate vessel itinerary --------------------------------------------
def cal_itinerary(vessel):
    vessel_itinerary_df = y_df.copy()
    # Extract and clean up Wharf details
    vessel_itinerary_df['Start_Wharf'] = vessel_itinerary_df['Task'].apply(lambda x: x.split('_')[-1].strip())
    vessel_itinerary_df['End_Wharf'] = vessel_itinerary_df['Start_Wharf']

    # Lookup Start and End Stations based on Wharfs
    vessel_itinerary_df['Start_Station'] = vessel_itinerary_df['Start_Wharf'].apply(lambda x: Start_S.get(x, 'Unknown Station'))
    vessel_itinerary_df['End_Station'] = vessel_itinerary_df['Start_Wharf'].apply(lambda x: End_S.get(x, 'Unknown Station'))

    # Update Task based on specific keywords or conditions
    vessel_itinerary_df['Task'] = vessel_itinerary_df['Task'].apply(lambda x: 'Waiting' if x in B else
                                                                              'Crew pause' if x in Bc else
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
for vessel in vessels:
    cal_itinerary(vessel).to_csv(f'ILPimplementation/vessel_itineraries/vessel_{vessel}_itinerary.csv', index=False)

# -------------------------------------------- Generate wharf utilization --------------------------------------------

def cal_wharf_utilization(wharf):
    # non line task
    non_line_tasks = y_df[~y_df['Task'].str.match(r'^\d+$')]
    utilization_data = []

    for _, row in non_line_tasks.iterrows():
        v = row['Vessel']
        j = row['Task']
        w = row['Task'].split('_')[-1]
        t = row['Start_Time']
        t_list = np.array([x[1] for x in cal_delta(config, j, w)]) + t
        utilization_data.append({'v': v, 'j': j,'w': w, 't_list': t_list.tolist()})

    # line task
    line_tasks = y_df[y_df['Task'].str.match(r'^\d+$')]
    for _, row in line_tasks.iterrows():
        v = row['Vessel']
        j = int(row['Task'])
        t = row['Start_Time']
        
        # origin
        w_start = line_df[line_df['Line_No'] == j]['O'].iloc[0]
        # safety_buffer = int(line_df[line_df['Line_No'] == j]['Safety_buffer'].iloc[0])
        safety_buffer = 0
        t_list_start = [time for time in range(t, t + cal_duration(safety_buffer))]
        utilization_data.append({'v': v, 'j': j, 'w': w_start, 't_list': t_list_start})

        # intermidiate stop
        intermediate_station = line_df[line_df['Line_No'] == j]['I']
        if pd.notna(intermediate_station).any():
            w_intermediate = z_df[z_df['Task'] == str(j)]['Wharf'].iloc[1]
            t_list_intermediate = np.array([x[1] for x in cal_delta(config, j, w_intermediate)]) + t
            utilization_data.append({'v': v, 'j': j, 'w': w_intermediate, 't_list': t_list_intermediate.tolist()})

        # terminus
        timetoT = int(line_df[line_df['Line_No'] == j]['Time_underway_to_T'].iloc[0])
        arrival_T = t + cal_duration(timetoT)

        filtered_wharfs = Zp_df[(Zp_df['Line'] == j) & (Zp_df['Time'] == arrival_T)]['Wharf'].unique().tolist()

        # Check if the list is not empty before accessing the first element
        if filtered_wharfs:
            w_end = filtered_wharfs[0]
        else:
            print(f"No matching wharf found for line {j} at time {arrival_T}")
            w_end = None  # Or set a default value or take another appropriate action

        # w_end = Zp_df[(Zp_df['Line'] == j) & (Zp_df['Time'] == arrival_T)]['Wharf'].unique().tolist()[0]

        dw_T = int(line_df[line_df['Line_No'] == j]['dw_T'].iloc[0])
        periods = cal_duration(dw_T + safety_buffer)
        t_list_end = Zp_df[(Zp_df['Line'] == j) & (Zp_df['Time'] >= arrival_T) & (Zp_df['Time'] <= arrival_T + periods)]['Time'].unique().tolist()
        utilization_data.append({'v': v, 'j': j, 'w': w_end, 't_list': t_list_end})

    # df process
    all_wharfs_utilization_df = pd.DataFrame(utilization_data, columns=['v', 'j', 'w', 't_list'])
    wharf_df = all_wharfs_utilization_df[all_wharfs_utilization_df['w'] == wharf].explode('t_list').reset_index(drop=True)
    wharf_df = wharf_df.rename(columns={'t_list': 't'}).sort_values('t')
    wharf_df['t'] = wharf_df['t'].fillna(0) # newly add causing problem?
    wharf_df['t'] = wharf_df['t'].apply(lambda x: cal_time(x).strftime('%H:%M') + '-' + cal_time(x+1).strftime('%H:%M'))

    wharf_df.rename(columns={'v': 'Vessel', 'j': 'Task', 'w': 'Wharf', 't': 'Time'}, inplace=True)
    
    # 更新任务描述
    wharf_df['Task'] = wharf_df['Task'].apply(lambda x: 'Waiting' if x in B else
                                              'Crew pause' if x in Bc else
                                              'Charging' if x in Bplus else
                                              f"{x}")

    wharf_df.reset_index(drop=True)
    wharf_df['Task'] = wharf_df['Task'].replace(line_to_route_dict)
    
    return wharf_df

wharfs = wharf_df['Wharf_No'].unique().tolist()

for wharf in wharfs:
    cal_wharf_utilization(wharf).to_csv(f'ILPimplementation/wharf_utilizations/wharf_{wharf}_utilization.csv', index=False)

