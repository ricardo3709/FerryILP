import pandas as pd
from datetime import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tqdm import tqdm

## -------------------- Load csv files --------------------
# Station and wharves dataframe
wharf_df = pd.read_csv('ILPimplementation/wharf_info.csv')
# lines dataframe
line_df = pd.read_csv('ILPimplementation/line_info.csv')
line_df['First_sailing'] = pd.to_datetime(line_df['First_sailing'], format='%H:%M')
# Wharf to wharf transit time dataframe
tt_df = pd.read_csv('ILPimplementation/rebalancing_times.csv',index_col='From/To')
# Headways dataframe
headway_df = pd.read_csv('ILPimplementation/headways.csv')
# vessels
vessel_df = pd.read_csv('ILPimplementation/vessel_info.csv')
# charging berths dataframe
charging_berth = pd.read_csv('ILPimplementation/charging_berths.csv')

print('All .csv files have been loaded successfully.\n')


## Other input required
# Simulation time parameters
initial_time = time(5,0)
period_length = 5 # min
total_operation_hours = 24 # hours

# nc, Minimum num of crew break
nc = 5 

# Dc, Crew break duration (fixed)
Dc = 60

# Tc, Maximum seperation time for crew breakings
Tc = 4*60

# rv+, charging rate
rv_plus = 2100 # kW

# rv, discharging rate for revalancing, based on max speed of the vessel
vessel_df['rv']

# pc. Plugging/Unplugging time
pc = 2 # min


## -------------------- Sets defined --------------------

# Lset: Set of Lines
Lset = line_df['Line_No'].unique().tolist()

# Zset: Set of Sailing 
nl_ls= [len(headway_df[f"h{l}"].dropna().tolist())+1 for l in Lset]
s_ls = [list(range(1,nl+1)) for nl in nl_ls]
Zset = []

for line in Lset:
    for sailing in s_ls[line-1]:
        Zset.append(f'{line}_{sailing}')

# Vset: Set of Vessels
Vset = vessel_df['Vessel code'].unique().tolist()

# Wset: Set of Wharves
Wset = wharf_df['Wharf_No'].unique().tolist()

# Tset: Set of Time Periods
Tset = [i for i in range(1, total_operation_hours * 60 // period_length + 1)]

# Jset: Set of tasks (Defined based on the Set B, Bc, and Bplus)
# B+, set of wharves to charge
Bplus = [wharf for wharf in charging_berth['Wharf_No'].unique().tolist()]
# print(f'The set of wharves for vessels can charge, B+: {Bplus}')

# Non loading berths
original_non_loading_berths = wharf_df[wharf_df['Non_loading_berths'] != 0]['Wharf_No'].unique().tolist()

# Bc, set of wharves to crew pause, is copy of set B
Bc = ['cp_' + wharf for wharf in original_non_loading_berths] # or Bc = wharf_df[wharf_df['Crew_pause'] == 'Yes']['Wharf_No'].unique().tolist()    they result in the same set
# print(f'The set of wharves for vessels can do crew pause, Bc: {Bc}')

# B, set of wharves to wait, any wharf with a charger belongs to B, and B contains wharves with original non loading berths
B = original_non_loading_berths.copy() 

for wharf in Bplus:
    if wharf in B: # wharf with a charger
        B.remove(wharf)
        B.append(f'phi_{wharf}') # mark as phi(w)
    else:
        B.append(f'phi_{wharf}')  # input directly

# print(f'The set of wharves for vessels can wait, B: {B}')

# Jset = Lset + B + B+ + Bc
Jset = [ele for ele in Lset + B + Bc + Bplus]

# Dset: Set of possible first sailing time (Defined later after the functions)



## -------------------- Functions --------------------

def cal_Cw(w):
    """
    Calculate the total capacity (number of berths) of a specific wharf.

    Parameters:
    w (str): The wharf identifier.

    Returns:
    int: The total capacity of the wharf, including loading and non-loading berths.

    Raises:
    ValueError: If the wharf identifier is not found or if required data is missing.
    """
    try:
        # Extract rows corresponding to the wharf
        wharf_data = wharf_df[wharf_df['Wharf_No'] == w]
        
        # Check if the wharf exists in the data
        if wharf_data.empty:
            raise ValueError(f"No data found for wharf {w}. Please check the wharf identifier.")
        
        # Calculate total capacity
        loading_berths = wharf_data['Loading_berths'].iloc[0]
        non_loading_berths = wharf_data['Non_loading_berths'].iloc[0]

        # Handle possible missing values for berths
        if pd.isna(loading_berths) or pd.isna(non_loading_berths):
            raise ValueError(f"Missing berth information for wharf {w}.")

        total_capacity = int(loading_berths) + int(non_loading_berths)
        return total_capacity

    except KeyError as e:
        raise KeyError(f"Missing required columns in the dataframe: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while calculating the capacity for wharf {w}: {str(e)}")


def cal_Rl(l):
    """
    Calculate the route for a given line number from a DataFrame.
    
    Parameters:
    l (int): The line number for which the route is to be extracted.
    
    Returns:
    list: A list containing the sequence of stations for the line, excluding any NaN entries.
    
    Raises:
    ValueError: If the line number is not found in the DataFrame or input is not an integer.
    """
    
    # Check input type to be an integer
    if not isinstance(l, int):
        raise ValueError("Line number must be an integer.")
    
    # Check if line number exists
    if l not in line_df['Line_No'].values:
        raise ValueError(f"Line number {l} is not found in the DataFrame.")
    
    try:
        route_data = line_df[line_df['Line_No'] == l][['O', 'I', 'T']].iloc[0]
        # Filter out NaN values and convert to list
        R_l = [station for station in route_data if station not in [ None,'None', '', ' ', np.nan, np.NaN]]
        return R_l
    except IndexError:
        raise ValueError(f"No data available for line number {l}.")
    except KeyError:
        raise ValueError("DataFrame must include 'Line_No', 'O', 'I', 'T' columns.")
    

def cal_C_lS(S):
    """
    Calculate the set of usable wharves for a given station of a line.
    
    Parameters:
    S (str): The station name for which the set of wharves is to be calculated.
    
    Returns:
    list: A list of unique wharf numbers that can be used at the given station.
    
    Raises:
    ValueError: If the station does not exist in the DataFrame or the input is not a string.
    """
    # Check if input is a string
    if not isinstance(S, str):
        raise ValueError("Station name must be a string.")
    
    # Check if station exists in DataFrame
    if S not in wharf_df['Station'].values:
        raise ValueError(f"Station {S} is not found in the DataFrame.")
    
    try:
        # Extract unique wharves for the station
        C_lS = wharf_df[wharf_df['Station'] == S]['Wharf_No'].unique().tolist()
        return C_lS
    except KeyError:
        raise ValueError("DataFrame must include 'Station' and 'Wharf_No' columns.")
    

def cal_Sv(v):
    """
    Retrieve the starting station for a given vessel identified by its vessel code.
    
    Parameters:
    v (str): The vessel code for which the starting station is to be retrieved.
    
    Returns:
    str: The starting station of the vessel.
    
    Raises:
    ValueError: If the vessel code does not exist in the DataFrame or the input type is incorrect.
    """
    # Validate input type
    if not isinstance(v, str):
        raise ValueError("Vessel code must be a string.")
    
    # Check if the vessel code exists in the DataFrame
    if v not in vessel_df['Vessel code'].values:
        raise ValueError(f"Vessel code {v} is not found in the DataFrame.")
    
    try:
        # Extract the starting station for the vessel
        S_v = vessel_df[vessel_df['Vessel code'] == v]['Sv'].iloc[0]
        return S_v
    except KeyError:
        raise ValueError("DataFrame must include 'Vessel code' and 'Sv' columns.")
    except IndexError:
        raise ValueError(f"No data available for vessel code {v}. The vessel might not be listed.")   


def cal_li(v):
    """
    Calculate the lines that a given vessel can serve, based on its starting station and the routes it can serve.
    
    Parameters:
    v (str): The vessel code for which lines are to be calculated.
    
    Returns:
    list: A list of line numbers that the vessel can serve starting from its designated station.
    
    Raises:
    ValueError: If the vessel code does not exist in the DataFrame or if the DataFrame structure is incorrect.
    """
    # Validate input type
    if not isinstance(v, str):
        raise ValueError("Vessel code must be a string.")
    
    # Check if vessel code exists in the DataFrame
    if v not in vessel_df['Vessel code'].values:
        raise ValueError(f"Vessel code {v} is not found in the DataFrame.")
    
    try:
        # Extract rows for the vessel and find routes served
        vessel_row = vessel_df[vessel_df['Vessel code'] == v].iloc[0]
        routes_served = [route for route in vessel_df.columns[2:-1] if vessel_row[route] == 'Yes']
        li_v = line_df[line_df['Route_No'].isin(routes_served)]['Line_No'].tolist() # DO NOT DEPEND ON THE Sv

        return li_v
    except KeyError:
        raise ValueError("DataFrame must include 'Vessel code', 'Route_No', 'O', and necessary route columns.")
    except IndexError:
        raise ValueError(f"No data available for vessel code {v}.")


def cal_D(l):
    '''
    Determines the allowable time periods for the first sailing of a specified line, 
    relative to the given initial simulation time and allowed_latitude.

    Parameters:
    l (int): Line number.

    Returns:
    list: A list of allowable time periods, D(l), during which the specified line's
          first sailing is considered permissible.

    Example:
    For line l1 and an initial simulation time of 5:00 AM, D(l1) might include time periods
    where the first sailing time of line l1 falls within a certain allowable range.
    
    '''
    first_sailing_time = line_df[line_df['Line_No'] == l]['First_sailing'].iloc()[0]
    delta_minutes = (first_sailing_time.hour * 60 + first_sailing_time.minute) - (initial_time.hour * 60 + initial_time.minute)

    allowed_latitude = 15 # min
    # Create a set of allowable time period numbers
    D_l = list(range((delta_minutes - allowed_latitude) // period_length + 1, ((delta_minutes + allowed_latitude) // period_length + 1) + 1))
    return D_l

def cal_h(s, d, line):
    """
    Calculate the s-th sailing time starting from time 'd' for a specified 'line'.

    This function retrieves headway periods for a specific line from a dataframe,
    constructs a list of sailing times starting from time 'd', and returns the s-th
    sailing time based on these intervals. It is used to project sailing schedules
    based on a specified headway and start day.

    Parameters:
    s (int): The order of the sailing time to be retrieved (1st, 2nd, etc.).
    d (int): The day from which sailing starts.
    line (int): The line number for which headway data is to be used.

    Returns:
    int or None: The s-th sailing time in terms of the number of time periods
                 since day 'd', or None if the index 's' is out of range or
                 an error occurs in processing.

    Note:
    The function assumes `period_length` as a global variable that denotes the
    length of each time period within the operational schedule.
    """

    try:
        # Retrieve the headway periods for the specified line and drop missing values
        h = headway_df[f'h{line}'].dropna().tolist()
        h_sd_ls = [d]  # Start the list with the initial day 'd'

        # Calculate subsequent sailing times based on headway periods
        for sailing_headway in h:
            num_time_period = int(sailing_headway // period_length + 1)  # round up
            h_sd_ls.append(h_sd_ls[-1] + num_time_period)

        # Return the s-th sailing time if within bounds
        if s-1 < len(h_sd_ls):
            return h_sd_ls[s-1]
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def cal_mu(j):
    """
    Calculate the duration in time periods for a given task j.
    
    Parameters:
    j (int or str): The task identifier which can be a line number, crew pause, or waiting task.
    
    Returns:
    int or None: The number of time periods the task takes, or None if the task is unrecognized.
    
    Raises:
    ValueError: If the input or required global variables are improperly configured.
    """
    try:
        if not isinstance(j, (int, str)):
            raise ValueError("Task identifier must be an integer or string.")
        
        # Check if j is in lists
        if j in Lset:
            mu_j = line_df[line_df['Line_No'] == j]['Line_duration'].iloc()[0] // period_length + 1
        elif j in Bc:
            mu_j = Dc // period_length + 1
        elif j in Bplus or j in B:
            mu_j = 1
        else:
            # If task j is unrecognized
            return None
        return mu_j
    except KeyError as e:
        raise ValueError(f"Missing data for task {j}: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")
    

def cal_q(v, j, t):
    """
    Calculate the battery change rate qv,j,t for a vessel v performing task j at time t.

    Parameters:
    v (str): Identifier for the vessel.
    j (str or int): Identifier for the task.
    t (int): Time unit after the start of the task.

    Returns:
    float: The rate of battery change at time t.
    """

    if j in Bplus: # the task is charging, return the charging rate
        return rv_plus
    elif j in B: # the task is the first or last period of charging
        epsilon = 1-pc/period_length # REQUIRE CHECK: Do I understand this correctly ????????????????????????????????
        return epsilon * rv_plus
    elif j in B: # the task is waiting
        return 0
    elif j in Lset: # the task is a line
        l = j
        line_data = line_df[line_df['Line_No'] == l]
        R_l = cal_Rl(l)
        stops = R_l[1:] # remove the origin station
        # Check if it's during a stop
        if len(stops) == 1: # No intermediate stops
            a = int(line_data['Time_underway_to_T'].iloc[0] // period_length + 1)
            dw = int(line_data['dw_T'].iloc[0] // period_length + 1)
            if t in list(range(a,(a+dw)+1)): # at stop
                return 0 
            else:
                rj = line_data['rj'].iloc()[0]
        elif len(stops) == 2: # with intermediate stops
            a1 = int(line_data['Time_underway_to_I'].iloc[0] // period_length + 1)
            dw1 = int(line_data['dw_I'].iloc[0] // period_length + 1)
            a2 = int(line_data['Time_underway_to_T'].iloc[0] // period_length + 1)
            dw2 = int(line_data['dw_T'].iloc[0] // period_length + 1)

            if t in list(range(a1,(a1+dw1)+1)) + list(range(a2,(a2+dw2)+1)):
                return 0
            
            else:
                rj = line_data['rj'].iloc()[0]
        return -rj  # Negative because it's a consumption rate
    else: # the vessel is rebalncing, rv will be captured by the constraint
        return 0


def get_task_location(j, type):
    """
    Retrieve the starting station or wharf location for a given task.

    Parameters:
    j (str or int): Task identifier, which could be a line number or a specific wharf/task identifier.
    type (int): Type of location required (0 for start or -1 for end).

    Returns:
    str: The starting station or wharf associated with the task.

    Raises:
    ValueError: If the task identifier is unrecognized, necessary data is missing, or parameters are invalid.
    IndexError: If expected data is not available in the DataFrame.
    """
    try:
        if type not in [0, -1]:
            raise ValueError("Type parameter must be int, 0 for start or -1 for end.")

        if j in Lset:
            task_stations = cal_Rl(j)
            if not task_stations:
                raise ValueError(f"No stations found for line {j}.")
            task_station = task_stations[type]
        elif j in Bc or j in B or j in Bplus:
            task_wharf = j.split('_')[-1]
            task_station_df = wharf_df[wharf_df['Wharf_No'] == task_wharf]
            if task_station_df.empty:
                raise ValueError(f"No station found for wharf identifier {task_wharf} from task {j}.")
            task_station = task_station_df['Station'].iloc[0]
        else:
            raise ValueError(f"Task identifier {j} is unrecognized or does not belong to known task sets.")
        
        return task_station
    except IndexError as e:
        raise IndexError(f"Data retrieval error for task {j}: {str(e)}")
    except Exception as e:
        raise Exception(f"An error occurred while retrieving the location for task {j}: {str(e)}")


def cal_xi(j1, j2):
    """
    Calculate the rebalancing time needed to travel from the end of task j1 to the start of task j2.

    Parameters:
    j1 (str or int): Task identifier for the first task.
    j2 (str or int): Task identifier for the second task.

    Returns:
    int: The number of time units required to transition from the end of j1 to the start of j2.

    Raises:
    ValueError: If any of the task identifiers are unrecognized, if location data is missing,
                or if there is no travel time data available between the two tasks.
    """
    try:
        # Retrieve the ending location of j1 and starting location of j2
        end_location_j1 = get_task_location(j1, -1)
        start_location_j2 = get_task_location(j2, 0)

        # Check if any location data is missing
        if not end_location_j1 in tt_df.columns or not start_location_j2 in tt_df.columns:
            travel_time = 24*60+1 # Very long time --> ensure no rebalancing happend between the two station

        # Fetch travel time from the DataFrame based on the locations
        else:
            travel_time = tt_df.loc[end_location_j1, start_location_j2]

        # Calculate time units required for transition
        xi_j1_j2 = travel_time // period_length + 1
        return xi_j1_j2
    except Exception as e:
        raise Exception(f"An error occurred while calculating transition time from {j1} to {j2}: {str(e)}")
    

def cal_xi0(v, j):
    """
    Calculate the time periods required for vessel v to travel from its starting position to the starting point of task j.
    If the starting position is the same as the task location, xi0 is zero.
    
    Parameters:
    v (str): The vessel identifier.
    j (str or int): The task identifier, which could be a line or a specific wharf.
    
    Returns:
    int: The time period required to travel from the vessel's starting wharf to the wharf where task j begins, or zero if they are the same.
    
    Raises:
    ValueError: If input data is missing or incorrect, or if travel times are not found in the DataFrame.
    """
    try:
        # Fetch the starting station for vessel v
        S_v = cal_Sv(v)
        # Fetch the starting station of the task using the refined function
        task_station = get_task_location(j,0)

        # If the starting point and task location are the same, return zero
        if S_v == task_station:
            return 1
        
        elif not S_v in tt_df.columns or not task_station in tt_df.columns:
            travel_time = 24*60+1 # Very long time --> ensure no rebalancing happend between the two station 

        else: # Fetch travel time from the travel time DataFrame
            travel_time = tt_df.loc[S_v, task_station]

        # Calculate periods
        xi0 = travel_time // period_length + 1
        return int(xi0)
    except KeyError as e:
        raise KeyError(f"DataFrame column missing: {str(e)}")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")
    

def cal_C(j):
    """
    Calculate all wharves that could be used given task j.
    If j is a line, aggregates wharves from all stations visited by the line.
    If j is a wharf (either in B or Bc), returns just that wharf.

    Parameters:
    j (int): The task identifier, which could be a line number or a wharf.

    Returns:
    list: A list of all wharves usable for the task, or None if task is unrecognized.

    Raises:
    Exception: Raises an exception with a descriptive message if any error occurs.
    """
    try:
        C_j = []
        if isinstance(j, int) and j in Lset:
            R_l = cal_Rl(j)  # stations visited by the line
            # print(R_l)
            for S in R_l:
                C_lS = cal_C_lS(S)
                C_j.extend(C_lS)  # Use extend to avoid nested lists
        elif isinstance(j, str) and (j in Bc or B or Bplus):
            C_j.append(j.split('_')[-1])
        else:
            raise ValueError(f"Task {j} is unrecognized or inappropriate data type")

        return C_j
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")


def cal_delta(j, w):
    """
    Calculate the set of times the wharf w will be occupied due to task j starting at time t0.
    For lines, excludes the last station from consideration and includes a safety buffer.
    For specific tasks like B, Bplus, or Bc, directly uses the task's duration.

    Parameters:
    j (int or str): The task identifier, which could be a line number or a specific wharf/task identifier.
    w (str): The wharf identifier.

    Returns:
    list: A list of tuples (wharf, time) indicating times the wharf is occupied.

    Raises:
    ValueError: If the task identifier or wharf is unrecognized, or if the wharf does not belong to the task.
    KeyError: If necessary data columns are missing from the data frames.
    IndexError: If data extraction based on indices fails.
    """
    
    try:
        if isinstance(j, int) and j in Lset:
            l = j # task is a line
            line_data = line_df[line_df['Line_No'] == l]
            safety_buffer = 1  # Assume a safety buffer of 1 time period
            R_l = cal_Rl(j)  # Stations visited by the line

            # Exclude the first and last station
            stations = R_l[1:-1]

            # Iterate through intermediate stations only
            for station in stations: # only one element here
                wharves = cal_C_lS(station)
                if w in wharves:
                    a = int(line_data['Time_underway_to_I'].iloc[0] // period_length + 1)
                    dw = int(line_data['dw_I'].iloc[0] // period_length + 1)
                    delta_j_w = [(w, time) for time in range(a - safety_buffer, (a + dw - 1 + safety_buffer) + 1)]
                    return delta_j_w

            # raise ValueError(f"Wharf {w} is not available for the task {j} at any intermediate stops.")
            return []

        elif isinstance(j, str) and (j in Bc or j in B or j in Bplus):
            if w != j.split('_')[-1]:
                raise ValueError(f"Task {j} should occupy its own wharf {j.split('_')[-1]}, not {w}.")
            mu_j = cal_mu(j)
            delta_j_w = [(w, time) for time in range(mu_j)] 

        else:
            raise ValueError(f"Task {j} is unrecognized or has an inappropriate data type.")

    except KeyError as e:
        raise KeyError(f"Missing data column: {str(e)}")
    except IndexError as e:
        raise IndexError(f"Data extraction error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error occurred: {str(e)}")


def cal_F(l):
    """
    Calculate the number of time periods from the start of a sailing until arrival at the last station of line l.
    
    Parameters:
    l (int): Line
    
    Returns:
    int: Time periods until arrival at the last station
    
    Raises:
    ValueError: If line number does not exist or required data is missing.
    """
    try:
        if not isinstance(l, int):
            raise ValueError("Line number must be an integer.")
        
        time_underway_to_T = line_df[line_df['Line_No'] == l]['Time_underway_to_T'].iloc()[0]
        F_l = time_underway_to_T // period_length + 1
        return F_l
    except IndexError:
        raise ValueError(f"No data available for line number {l}.")
    except KeyError:
        raise ValueError("Missing 'Time_underway_to_T' in line_df.")


def cal_muF(l):
    """
    Calculate the number of time periods a wharf is occupied at the last station by line l, including any safety buffer.
    
    Parameters:
    l (int): Line number
    
    Returns:
    int: Time periods a wharf is occupied
    
    Raises:
    ValueError: If line number does not exist or required data is missing.
    """
    try:
        if not isinstance(l, int):
            raise ValueError("Line number must be an integer.")
        
        dw_T = line_df[line_df['Line_No'] == l]['dw_T'].iloc()[0]
        muF_l = dw_T // period_length + 1
        return muF_l
    except IndexError:
        raise ValueError(f"No data available for line number {l}.")
    except KeyError:
        raise ValueError("Missing 'dw_T' in line_df.")


def cal_phi(j, t):
    """
    Calculate a set of time periods within which if a task j starts, it will still be ongoing at time t.
    
    Parameters:
    j (int or str): The task identifier.
    t (int): The time period at which task j is still ongoing if started within the returned set.
    
    Returns:
    list: A list of time periods representing possible start times for task j to be ongoing at time t.
    
    Raises:
    ValueError: If the inputs are not valid or the time period is out of expected range.
    """
    try:
        # Check if t is a valid input
        if not isinstance(t, int) or t < 1:
            raise ValueError("Time period t must be a positive integer.")

        mu_j = cal_mu(j)  # Duration that task j occupies
        if mu_j is None:
            raise ValueError(f"No duration found for task {j}. It may be unrecognized.")

        # Compute the range of start times
        phi_j_t = list(range(max(1, t - mu_j + 1), t + 1))
        return phi_j_t
    except Exception as e:
        raise ValueError(f"An error occurred calculating φ(j, t): {str(e)}")


def cal_f(j):
    """
    Determines the latest feasible start time for task `j` so that it finishes before the day ends.

    Parameters:
    j (str or int): Task identifier. Can be an integer for line tasks or a string for specific wharves or tasks.

    Returns:
    int or None: The last valid starting period for the task, or None if the task cannot be completed in a day or an error occurs in processing.
    """
    try:
        # Validate that Tset is properly defined and not empty
        if not Tset:
            raise ValueError("Tset is not defined or is empty.")
        
        last_period = Tset[-1]  # Last time period in the set
        
        # Calculate mu(j) based on the type of task
        if j in Lset:
            mu_j = line_df[line_df['Line_No'] == j]['Line_duration'].iloc()[0] // period_length + 1
        elif j in Bc:
            mu_j = Dc // period_length + 1
        elif j in Bplus or j in B:
            mu_j = 1  # Fixed duration for waiting tasks
        else:
            # If task j is unrecognized
            return None

        # Calculate the latest feasible start time
        last_start_time = last_period + 1 - mu_j
        return last_start_time
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def cal_G(j):
    """
    Calculate the set of valid start times for task j.
    
    If j is a line, G(j) includes times based on headways and initial days from D(l).
    If j is a crew pause, waiting, or charging, G(j) includes all times in Tset.

    Parameters:
    j (int or str): The task identifier.

    Returns:
    list: List of valid start times for the task j.

    Raises:
    ValueError: If task j is unrecognized or essential data is missing.
    """
    
    try:
        G_j = []
        if j in Lset:  # a line
            headways = headway_df.get(f'h{j}', pd.Series()).dropna().tolist()
            if not headways:
                raise ValueError(f"No headway data available for line {j}")
            D_l = cal_D(j) 
            for d in D_l:
                G_j.append(d)
                current_time = d
                for h in headways:
                    num_time_period = int(h // period_length + 1)
                    current_time += num_time_period
                    G_j.append(current_time)
        elif j in Bc or j in B or j in Bplus:  # crew pause / waiting / charging
            G_j = Tset.copy() 
        else:
            raise ValueError(f"Task {j} is unrecognized or not handled.")

        return G_j
    except Exception as e:
        raise ValueError(f"An error occurred processing task {j}: {str(e)}")


def cal_H(v, j):
    """
    Calculate the set of feasible start times H(v,j) for vessel v to start task j.
    
    Parameters:
    v (int): Index of the vessel.
    j (int or str): Index of the task, a line or a wharf.
    
    Returns:
    list: List of feasible times or None if start point is not determined.
    """
    try:
        S_v = cal_Sv(v)
        if pd.isna(S_v):
            print(f'The start point of the vessel {v} has not been determined. Please determine it first.')
            return None

        xi0_vj = cal_xi0(v, j)
        f_j = cal_f(j)
        G_j = cal_G(j)

        # Filter G(j) to find all t such that xi0(v, j) ≤ t ≤ f(j)
        H_vj = [t for t in G_j if xi0_vj <= t <= f_j]

        return H_vj
    except Exception as e:
        print(f"An error occurred while calculating H(v, j): {str(e)}")
        return None
    
def cal_taskF(j, t):
    """
    Calculate the set of tasks that can be performed and finished after task j if j started at time t.

    Parameters:
    j (str or int): The task identifier for which subsequent tasks are calculated.
    t (int): The start time of task j.

    Returns:
    list: A list of tasks that can be started and completed after task j.

    Raises:
    ValueError: If the start time 't' is not valid or if 'j' does not have a feasible completion time.
    KeyError: If there are any missing required data fields or calculations.
    """
    try:
        # Validate input
        if not isinstance(t, int) or t < 0:
            raise ValueError("Start time 't' must be a non-negative integer.")
        
        feasible_tasks = []
        f_j = cal_f(j)  # Calculate the latest feasible start time for task j

        if t > f_j:
            return feasible_tasks  # Return an empty list if j can't be completed

        mu_j = cal_mu(j)  # Duration of task j

        # Loop through all tasks in the global set Jset to find feasible subsequent tasks
        for j_prime in Jset:
            f_j_prime = cal_f(j_prime)  # Latest feasible start time for task j'
            xi_j_j_prime = cal_xi(j, j_prime)  # Travel time from task j to task j'

            # Check if task j' can start after j considering travel time and its own constraints
            if f_j_prime >= t + mu_j + xi_j_j_prime:
                feasible_tasks.append(j_prime)

        return feasible_tasks

    except KeyError as e:
        raise KeyError(f"Missing data for task calculation: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")
    

def cal_E(w, t):
    """
    Calculate the set of pairs (j, t') such that starting task j at time t' results in using a wharf
    at station associated with wharf w at time t.

    Parameters:
    w (str): Wharf identifier.
    t (int): Specific time point.

    Returns:
    set: Set of pairs (j, t') meeting the specified conditions.
    """
    E_wt = []

    for j in Jset:
        C_j = cal_C(j)  # Set of wharves usable for task j
        # print(f'target wharf: {w}, current task {j}, wharf available for task:{C_j}') 
        if w in C_j:
            delta_jw = cal_delta(j, w)  # Set of time units allowed between t' and t for task j and wharf w
            for t_prime in [time for time in Tset if time <= t]:
                if (t - t_prime) in [usage[1] for usage in delta_jw]:
                    E_wt.append((j, t_prime)) # store the task and it's start time

    return E_wt


D_l = [cal_D(l) for l in Lset]
Dset = {l: d for l, d in zip(Lset, D_l)}

print('Vset, Wset, Tset, Jset, and Dset have been defined.\n')



## -------------------- Varibles --------------------
print('start creating model\n')
# Create model
model = gp.Model("Ferry ILP")

print('model created!\n')

print('Start loading variable!')
# variable x[l, d]
x = {}
for l in Lset:
    for d in Dset[l]:
        x[l, d] = model.addVar(vtype=GRB.BINARY, name=f"x_{l}_{d}")
print('Variable x is ready.')


# variable y[v, j, t]
y = {}
for v in Vset:
    for j in Jset:
        for t in Tset:
            y[v, j, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}_{j}_{t}")
print('Variable y is ready.')

# variable Q[v, t]
Q = {}
for v in Vset: 
    for t in Tset:
        Q[v, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Q_{v}_{t}")
print('Variable Q is ready.')

# variable z[j, w]
z = {}
for j in Jset:
    print(j)
    C_j = cal_C(j)
    for w in C_j: 
        z[w,j] = model.addVar(vtype=GRB.BINARY, name=f"z_{w}_{j}")
print('Variable z is ready.')

# variable Z[l, w, t]
Z = {}
for l in Lset:
    A_l = cal_Rl(l)[-1]
    C_lS = cal_C_lS(A_l)
    for w in C_lS:
        for t in Tset:
            Z[l, w, t] = model.addVar(vtype=GRB.BINARY, name=f"Z_{l}_{w}_{t}")
print('Variable Z is ready.')

# variable Z'[l, w, t]
Z_prime = {}
for l in Lset:
    A_l = cal_Rl(l)[-1]
    C_lS = cal_C_lS(A_l)
    for w in C_lS:
        for t in Tset:
            Z_prime[l, w, t] = model.addVar(vtype=GRB.BINARY, name=f"Z_prime_{l}_{w}_{t}")
print("Variable Z' is ready.\n")


## -------------------- Constraints --------------------

# Constraint 1a
for l in tqdm(Lset, desc='Constraint 1a'):
    model.addConstr(sum(x[l, d] for d in Dset[l]) == 1, name=f"departure_time_constraint_{l}")
# print('constraint 1a ok.')

# Constraint 1b
for sailing in tqdm(Zset, desc='Constraint 1b'):  
    l = int(sailing.split('_')[0]) # line
    s = int(sailing.split('_')[1]) # nth sailing
    # print(f'line:{l}')
    for d in Dset[l]:  #  
        h_sd = cal_h(s,d,l)
        t = h_sd
        # print(f'If the first sailing start at {d}, the {s}th sailing departure time is {t}')
        model.addConstr(sum(y[v, l, t] for v in Vset) == x[l, d],name=f"assign_vessel_s{s}_d{d}")
# print('constraint 1b ok.')

# Constraint 1c
for v in tqdm(Vset, desc='Constraint 1c'):
    for j in Jset:
        H_vj = cal_H(v,j)
        for t in [t for t in Tset if t not in H_vj]:
            y[v, j, t].ub = 0  # Set upper bound of y[v,j,t] to 0
# print('constraint 1c ok.')

# Constraint 1d
for t in tqdm(Tset, desc='Constraint 1d'):
    for v in Vset:
        li_v = cal_li(v)
        for j in [l for l in Lset if l not in li_v]:
            y[v, j, t].ub = 0  # Set upper bound of y[v,j,t] to 0
# print('constraint 1d ok.')

# Constraint 1e
for v in tqdm(Vset, desc='Constraint 1e'):   
    for j in Jset:
        xi0_v_j = cal_xi0(v,j)
        t = xi0_v_j
        if t in Tset:
            model.addConstr(sum(y[v, j, t] for j in Jset) == 1,name=f"assign_task_j{j}_t{t}")
# print('constraint 1e ok.')

# Constraint 1f 
for v in tqdm(Vset, desc='Constraint 1f'):
    for t in Tset:
        model.addConstr(sum(y[v, j, t_prime] for j in Jset for t_prime in cal_phi(j, t)) <= 1,name=f"task_overlap_v{v}_t{t}")
# print('constraint 1f ok.')

# Constraint 1g
for l in tqdm(Lset, desc='Constraint 1g'):
    R_l = cal_Rl(l)
    A_l = R_l[-1] # last station
    for S in [station for station in R_l if station != A_l]:
        C_lS = cal_C_lS(S)
        model.addConstr(sum(z[w, l] for w in C_lS) == 1,name=f"select_one_wharf_{l}_station_{S}")
# print('constraint 1g ok.')

# Constraint 1h
for l in tqdm(Lset, desc='Constraint 1h'):
    F_l = cal_F(l)
    for t in Tset:
        if t > F_l:
            A_l = cal_Rl(l)[-1] # last station
            C_lS = cal_C_lS(A_l) # available wharves at last station
            model.addConstr(sum(Z[l, w, t] for w in C_lS) == sum(y[v, l, t - F_l] for v in Vset),name=f"last_wharf_use_{l}_t{t}")
# print('constraint 1h ok.')

# Constraint 1i
for l in tqdm(Lset, desc='Constraint 1i'):
    A_l = cal_Rl(l)[-1]
    C_lS = cal_C_lS(A_l)
    for w in C_lS:
        muF_l = cal_muF(l)
        for t in Tset:
            if t > muF_l: 
                model.addConstr(Z_prime[l, w, t] == sum(Z[l, w, t - k] for k in range(muF_l)), name=f"wharf_occupation_{l}_{w}_t{t}")
# print('constraint 1i ok.')

# Constraint 1j
for v in tqdm(Vset, desc='Constraint 1j'):
    for w in Bplus:
        for t in Tset: 
            if t > 1:
                phi_w = f'phi_{w}'
                # j = w
                model.addConstr(y[v, w, t] <= y[v, w, t - 1] + y[v, phi_w, t - 1], name=f"full_period_charging_start_v{v}_w{w}_t{t}")
# print('constraint 1j ok.')

# Constraint 1k
for v in tqdm(Vset, desc='Constraint 1k'):
    for w in Bplus:
        for t in Tset: 
            if t <= Tset[-1]-1:
                phi_w = f'phi_{w}'
                # j = w
                model.addConstr(y[v, w, t] <= y[v, w, t + 1] + y[v, phi_w, t + 1], name=f"full_period_charging_2_v{v}_w{w}_t{t}")
# print('constraint 1k ok.')

# Constraint 2 
# This constraint requires a very long time to run.
for v in tqdm(Vset, desc='Constraint 2'):
    for j in Jset:
        for t in Tset:
            print(f'Current vessel: {v}; current task: {j}; current time {t}; number of possible following tasks: {len(cal_taskF(j, t))}')
            if cal_taskF(j, t) != []:
                model.addConstr(sum(y[v, j_prime, t + cal_mu(j) + cal_xi(j, j_prime)] for j_prime in cal_taskF(j, t)) >= y[v, j, t], name=f"follow_task_v{v}_j{j}_t{t}")
print('constraint 2 ok.')

# Constraint 3
# This constraint requires a very long time to run.
for v in tqdm(Vset, desc='Constraint 3'):
    for j in Jset:
        for t in Tset:
            for j_prime in cal_taskF(j, t):
                for t_prime in range(t + cal_mu(j), t + cal_mu(j) + cal_xi(j, j_prime)):
                    print(f"Current vessel: {v}; current task: {j}; current time {t}; current j'{j_prime}; current t' {t_prime}")
                    model.addConstr(y[v, j, t] + y[v, j_prime, t_prime] <= 1, name=f"no_overlap_v{v}_j{j}_t{t}_j_prime{j_prime}_t_prime{t_prime}")
print('constraint 3 ok.')

# Constraint 4
# This constraint requires a very long time to run.
for w in tqdm(Wset, desc='Constraint 4'):
    for t in Tset:
        print(w, t)
        # Sum over y and z
        sum_yz = sum(y[v, j, t_prime] * z[w, j] for v in Vset for (j, t_prime) in cal_E(w, t))
        # Sum over Z_prime with key check
        sum_Z_prime = sum(Z_prime[l, w, t] for l in Lset if (l, w, t) in Z_prime) 
        model.addConstr(sum_yz + sum_Z_prime <= cal_Cw(w), name=f"capacity_constraint_w{w}_t{t}")
print('constraint 4 ok.')

# Constraint 5a, 5b
for v in tqdm(Vset, desc='Constraint 5a'):
    for t in Tset:
        model.addConstr(Q[v, t] >= 0, name=f"battery_non_negative_v{v}_t{t}") 
print('constraint 5a ok.')

for v in tqdm(Vset, desc='Constraint 5b'):
    for t in Tset:
        model.addConstr(Q[v, t] <= 1, name=f"battery_max_capacity_v{v}_t{t}") 
print('constraint 5b ok.')

# Constraint 5c
# This constraint requires a very long time to run.
for v in tqdm(Vset, desc='Constraint 5c'):
    for t in Tset:
        rv = vessel_df[vessel_df['Vessel code'] == v]['rv'].iloc()[0]
        if t == 1:
            Qv0 = vessel_df[vessel_df['Vessel code'] == v]['Qv0'].iloc()[0]
            model.addConstr( Qv0 
                        + sum(cal_q(v, j, t - t_prime) * y[v, j, t_prime] for j in Jset for t_prime in cal_phi(j, t)) 
                        - rv * (1 - sum(y[v, j, t_prime] for j in Jset for t_prime in cal_phi(j, t))) 
                        >= Q[v, t], 
                        name=f"battery_update_v{v}_t{t}")     
        else:
            model.addConstr(Q[v, t - 1] 
                            + sum(cal_q(v, j, t - t_prime) * y[v, j, t_prime] for j in Jset for t_prime in cal_phi(j, t)) 
                            - rv * (1 - sum(y[v, j, t_prime] for j in Jset for t_prime in cal_phi(j, t))) 
                            >= Q[v, t], 
                            name=f"battery_update_v{v}_t{t}")
print('constraint 5c ok.')

# Constraint 6a
for v in tqdm(Vset, desc='Constraint 6a'):
    model.addConstr(sum(y[v, j, t] for j in Bc for t in Tset) >= nc, name=f"min_crew_pauses_v{v}")
print('constraint 6a ok.')

# Constraint 6b
for v in tqdm(Vset, desc='Constraint 6b'):
    model.addConstr(sum(y[v, j, t + t_prime] for j in Bc for t_prime in range(1, Tc//period_length+1) for t in Tset if t < (Tset[-1] - (Tc//period_length+1))) >= 1, name=f"max_distance_pauses_v{v}_t{t}")
print('constraint 6b ok.')
print('All constraintrs are ready.\n')


## -------------------- Objective Functions --------------------

# Variable psi[v]
psi = {}
for v in Vset:
    psi[v] = model.addVar(vtype=GRB.BINARY, name=f"psi_{v}")

# Objective Function 7
model.setObjective(gp.quicksum(psi[v] for v in Vset), GRB.MINIMIZE)

# Objective Function 8
M = Tset[-1] # ??

for v in Vset:
    model.addConstr(psi[v] >= (1 / M) * gp.quicksum(y[v, l, t] for l in Lset for t in Tset), name=f"utilize_vessel_{v}")


# Objective Function 9: Minimizing Rebalancing Time
# This constraint requires a very long time to run.
rebalancing_time = gp.quicksum(
    1 - gp.quicksum(y[v, j, t_prime] for j in Jset for t_prime in cal_phi(j, t))
    for v in Vset for t in Tset)
model.setObjective(rebalancing_time, GRB.MINIMIZE)


print('Model is ready to run now.')

## -------------------- Optimization --------------------
print('Starting optimization...')
model.optimize()
print('Optimization call completed.')


def save_variable_results(var_dict, filename):
    results = {k: var_dict[k].X for k in var_dict.keys()}  # Save all values
    df = pd.DataFrame(list(results.items()), columns=['Variable', 'Value'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}.")

# Check if the model has been solved
if model.status == GRB.OPTIMAL:
    print("Optimization was successful. Saving results...")
    
    # Save results for different variables
    save_variable_results(x, 'x_variable_results.csv')
    save_variable_results(y, 'y_variable_results.csv')
    save_variable_results(Q, 'Q_variable_results.csv')
    save_variable_results(z, 'z_variable_results.csv')
    save_variable_results(Z, 'Z_variable_results.csv')
    save_variable_results(Z_prime, 'Z_prime_variable_results.csv')
else:
    print("Optimization did not reach optimality.")


# Adding progress prints
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    print(f'\r{prefix} |{bar}| {percents}% {suffix}', end='\r')
    if iteration == total:
        print()

# Usage example in a loop
total_iterations = len(Tset)
for i, t in enumerate(Tset, 1):
    print_progress(i, total_iterations, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    # your code here, e.g., adding constraints
