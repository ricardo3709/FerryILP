import pandas as pd
from datetime import time

# Station and wharves dataframe

wharf_df = pd.read_csv('ILPimplementation/csv_inputs/wharf_info.csv')

# lines dataframe
line_df = pd.read_csv('ILPimplementation/csv_inputs/line_info.csv').assign(First_sailing=lambda df: pd.to_datetime(df['First_sailing'], format='%H:%M'))

# Wharf to wharf transit time dataframe
tt_df = pd.read_csv('ILPimplementation/csv_inputs/rebalancing_times.csv', index_col='From/To')

# Headways dataframe
headway_df = pd.read_csv('ILPimplementation/csv_inputs/headways.csv')

# vessels
vessel_df = pd.read_csv('ILPimplementation/csv_inputs/vessel_info.csv')

# charging berths dataframe
charging_berth = pd.read_csv('ILPimplementation/csv_inputs/charging_berths.csv')

print('All .csv files have been loaded successfully.\n')