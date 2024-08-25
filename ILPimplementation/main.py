import gurobipy as gp
import pickle
from config import *
from functions import *
from simulation_config import SimulationConfig  # Import the class
from constraints import add_constraints  # Import the constraints function
from variables import define_variables  # Import the variables function
from objectives import set_objective_functions
from optimization import run_optimization, save_all_results, save_relaxed_variable_results  # Import optimization functions

# Create model
model = gp.Model("Ferry ILP")

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
    'get_task_location': get_task_location,
    'hash_config': hash_config,
    'save_results': save_results,
    'load_results': load_results,
    'calculate_and_save_results': calculate_and_save_results,
    'load_all_results': load_all_results,
    'manage_results': manage_results,
}

config = SimulationConfig(
    wharf_df, line_df, headway_df, tt_df, vessel_df,
    initial_time, period_length, Tset,
    Lset, Bc, B, Bplus, Jset, Wset, Dset, Vset, Zset,
    Dc, nc, Tc, 
    rv_plus, pc, functions)

# Define variables
x, y, Q, z, Z, Z_prime = define_variables(model, config, cal_C, cal_Rl, cal_C_lS)

# Flag to decide whether to generate new files
# generate_new_files = True  
generate_new_files = False 


# Prefix for file names
file_prefix = "6htest_cyclelines"  # You can change the prefix as needed

# Manage results based on the flag
results = manage_results(config, generate_new_files, file_prefix)

if results:
    taskF_results, mu_results, xi_results, phi_results, E_results, nu_results = results
    print('Results loaded successfully.\n')
else:
    print('Results generated and saved successfully.\n')
    # Load results from pkl files
    with open(f'ILPimplementation/pkl_files/{file_prefix}_taskF_results.pkl', 'rb') as f:
        taskF_results = pickle.load(f)
    with open(f'ILPimplementation/pkl_files/{file_prefix}_mu_results.pkl', 'rb') as f:
        mu_results = pickle.load(f)
    with open(f'ILPimplementation/pkl_files/{file_prefix}_xi_jj_results.pkl', 'rb') as f:
        xi_results = pickle.load(f)
    with open(f'ILPimplementation/pkl_files/{file_prefix}_phi_results.pkl', 'rb') as f:
        phi_results = pickle.load(f)
    with open(f'ILPimplementation/pkl_files/{file_prefix}_E_results.pkl', 'rb') as f:
        E_results = pickle.load(f)
    with open(f'ILPimplementation/pkl_files/{file_prefix}_nu_results.pkl', 'rb') as f:
        nu_results = pickle.load(f)


# Add constraints
add_constraints(model, config, x, y, Q, z, Z, Z_prime, phi_results, E_results, mu_results, taskF_results, xi_results, nu_results)

# Set objective functions
psi = set_objective_functions(model, config, y, phi_results)

# Run optimization
run_optimization(model)

# Save relaxed variable results if infeasible
# save_relaxed_variable_results(model, x, 'ILPimplementation/output_files/relaxed_x_variable_results.csv')

# Save results if optimal
save_all_results(model, x, y, Q, z, Z, Z_prime,file_prefix)
