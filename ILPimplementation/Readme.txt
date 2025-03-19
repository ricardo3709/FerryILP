# Ferry Time tableing ILP Gurobi Solver


## Files

|        .py files       |                                       Description                                   |
|------------------------|-------------------------------------------------------------------------------------|
| `main.py`              | Execute the ILP model.                                                              | 
| `config.py`            | Configuration settings for the optimization model.                                  | 
| `simulation_config.py` | Centralises simulation parameters, datasets, and functions                          | 
| `constraints.py`       | Defines model constraints.                                                          | 
| `objectives.py`        | Specifies the objective function.                                                   | 
| `optimization.py`      | Optimization logic using Gurobi.                                                    | 
| `data_load.py`         | Handles input data loading from the folder 'csv_inputs'.                            | 
| `functions.py`         | Functions for other parapmeters.                                                    | 
| `ResultsProcess.py`    | Processes variable solutions into readable tables and,                              |
|                        | stores them in 'timetable', 'vessel_itineraries', and 'wharf_utilizations' folders. | 


|         .ipynb files         |                                             Description                                               |
|------------------------------|-------------------------------------------------------------------------------------------------------|
| `AllCodes.ipynb`             | Notebook contains all fuuntionality of the program in order to check in details. (original draft file)|
|                              |, there are detailed explaination and example of use for all the functions.                            |
| `result_visualization.ipynb` | Jupyter Notebook for visualizing processed results.                                                   |
| `ResultsProcessDraft.ipynb`  | Notebook contains almost same code for ResultsProcess.py in order to check steps in details.          |

---

## Getting Started

### 1️. Check input files in folder 'csv_inputs'
    there are 6 files:
        1. 'line_info.csv', contains the cyclines information (Route and Line Number, Stop stations, Dwelling times, First sailing time,	underway times, average discharging rate) 
        2. 'vessel_info.csv', contains the vessel's type, available serving lines, starting location, discharging rate and the initial battery level.
        3. 'rebalancing_times.csv', contains the travel time amongs the stations.
        4. 'headways.csv', defines the headways of each sailing for each line.
        5. 'wharf_info.csv', contains the information that wharf number, number of the berths and if available to the crew pause task 
        6. 'charging_berths.csv', indicates the berths that can charge.

### 2. Check the parameters in 'config.py'
    there are some parameters:
        # current version and start version
        1. 'current_version'
        2. 'starting_version' 
        3. 'prifix', this is the prifix of the starting version files, depends on the .pkl files they use.

        # pkl files
        4. 'generate_new_files', True or False to indicate if we want to create new pkl files
        5. 'pkl_file_prefix', prefix of the pkl files to load the files or generate new files.

        # solver
        6. 'Gap', the optimality gap tolerance
        7. 'TimeLimit', simulation time limit
        8. 'alpha', % of the objectives that focus on fleet size

        # simulation
        9. 'initial_time', simulation start time
        10. 'period_length', 1 time period = 5mins now
        11. 'total_operation_hours', simulation durtaion

        # crew break
        11. nc  # Minimum number of crew breaks
        12. Dc  # Crew break duration (minutes)
        13. Tc  # Maximum separation time for crew breaks (minutes)

        # charging 
        14. rv_plus # Charging rate (as percentage per period)
        15. pc # Plugging/Unplugging time (minutes)

### 3. Run 'mian.py'
    The results would be store in the folder ILPimplementation directly. the results are only values of varibales to be process to visualise.

### 4. Results process
    run 'ResultsProcess.py' directly, follow the prompts to input the version name of the processed results.
    the mannully put the original variable reuslts into a new folder with same version name.

### 5. Results visualisation
    Modify the version number in 'result_visualization.ipynb', then run the cells get the full visualisation for all vessel itenarary and wharf utilisation


