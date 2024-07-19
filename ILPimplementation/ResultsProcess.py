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

z_df = load_and_process_data('ILPimplementation/output_files/z_wj_results.csv',['Wharf', 'Line'],{'Line': int})
Zp_df = load_and_process_data('ILPimplementation/output_files/Z_prime_lwt_results.csv',['Line', 'Wharf', 'Time'],{'Line': int, 'Time': int})
Z_df = load_and_process_data('ILPimplementation/output_files/Z_lwt_results.csv',['Line', 'Wharf', 'Time'],{'Line': int, 'Time': int})
x_df = load_and_process_data('ILPimplementation/output_files/x_ld_results.csv',['Line', 'Time'],{'Line': int, 'Time': int})
y_df = load_and_process_data('ILPimplementation/output_files/y_vjt_results.csv',['Vessel', 'Task', 'Start_Time'],{'Start_Time': int})


# Determine start and end wharfs
Start_S = dict(zip(line_df['Line_No'].astype(str), line_df['O']))
Start_S.update(dict(zip(wharf_df['Wharf_No'], wharf_df['Station'])))
End_S = dict(zip(line_df['Line_No'].astype(str), line_df['T']))
End_S.update(dict(zip(wharf_df['Wharf_No'], wharf_df['Station'])))
Start_wharf = dict(zip(z_df['Line'].astype(str), z_df['Wharf']))


# -------------------------------------------- Additional functions for processing --------------------------------------------
def cal_time(period_num):

    # Convert initial_time to a datetime object with today's date
    initial_datetime = datetime.combine(datetime.today(), initial_time)
    
    # Calculate the total minutes to add
    added_time = timedelta(minutes=int((period_num - 1) * period_length))
    new_time = initial_datetime + added_time

    return new_time


def start_wharf(wharf):
    if wharf.isdigit():
        return Start_wharf.get(wharf, wharf)
    return wharf

def end_wharf(row):
    wharf = row['End_Wharf']
    if row['Task'].isdigit():
        line = int(row['Task'])
        timetoT = int(line_df[line_df['Line_No'] == line]['Time_underway_to_T'].iloc()[0])
        end_time = row['Start_Time'] + timetoT // period_length + 1
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
            wharfs.append(z_df[z_df['Line'] == line]['Wharf'].iloc[0])

            # Intermediate Stop
            Intermediate_stop = line_df[line_df['Line_No'] == line]['I'].iloc[0]
            if pd.notna(Intermediate_stop):
                timetoI = int(line_df[line_df['Line_No'] == line]['Time_underway_to_I'].iloc[0])
                arrival_time_I = sailing_time + timedelta(minutes=timetoI)
                times.append(arrival_time_I)
                locs.append(Intermediate_stop)
                wharfs.append(z_df[z_df['Line'] == line]['Wharf'].iloc[1])
            
            # Terminal Stop
            timetoT = int(line_df[line_df['Line_No'] == line]['Time_underway_to_T'].iloc[0])
            arrival_time_T = sailing_time + timedelta(minutes=timetoT)
            times.append(arrival_time_T)
            locs.append(End_S[str(line)])
            wharfs.append(Zp_df[(Zp_df['Line'] == line) & (Zp_df['Time'] == period + timetoT // period_length + 1)]['Wharf'].iloc[0])

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


cal_timetable(1).to_csv('ILPimplementation/test_timetable.csv')


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
    return itinerary

cal_itinerary('10M').to_csv('ILPimplementation/test_itinerary.csv')

# -------------------------------------------- Generate wharf utilization --------------------------------------------

def cal_wharf_utilization(wharf, y_df, Zp_df):
    wharf_utilization_df = y_df.copy()
    wharf_utilization_df['Start_Wharf'] = wharf_utilization_df['Task'].apply(lambda x: x.split('_')[-1].strip())
    wharf_utilization_df = wharf_utilization_df[wharf_utilization_df['Start_Wharf'] == wharf].sort_values('Start_Time')
    wharf_utilization_df['Occupied_time'] = wharf_utilization_df.apply(lambda row: [int(row['Start_Time']) + x[1] for x in cal_delta(config, row['Task'], wharf)], axis=1)

    #  Zp_df

    # This part is unfinished


    return wharf_utilization_df



cal_wharf_utilization('Bar1', y_df, Zp_df).to_csv('ILPimplementation/test_wharf_utilization.csv', index=True)

