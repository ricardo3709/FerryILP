import gurobipy as gp
import pickle
from config import *
from functions import *
from simulation_config import SimulationConfig
from constraints import add_constraints
from variables import define_variables
from objectives import set_objective_functions
from optimization import run_optimization, save_all_results

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
    'load_partial_solution':load_partial_solution,  #
    'set_partial_solution':set_partial_solution #
}

config = SimulationConfig(
    wharf_df, line_df, headway_df, tt_df, vessel_df,
    initial_time, period_length, Tset,
    Lset, Bc, B, Bplus, Jset, Wset, Dset, Vset, Zset,
    Dc, nc, Tc, 
    rv_plus, pc, functions)

# Define variables
x, y, Q, z, Z, Z_prime,u, p = define_variables(model, config, cal_C, cal_Rl, cal_C_lS)

# Starting Points 
partial_solutions = {key: load_partial_solution(file) for key, file in files.items()}
for var, partial in zip([x, y, Q, z, Z, Z_prime], partial_solutions.values()):
    set_partial_solution(var, partial, fix_values=False) 

# pkl files 
results = manage_results(config, generate_new_files, pkl_file_prefix)

if results:
    taskF_results, mu_results, xi_results, phi_results, E_results, nu_results = results
    print('Results loaded successfully.\n')
else:
    print('Results generated and saved successfully.\n')
    file_ids = ["taskF", "mu", "xi_jj", "phi", "E", "nu"]
    # load results
    results = {f"{file_id}_results": pickle.load(open(f'ILPimplementation/pkl_files/{pkl_file_prefix}_{file_id}_results.pkl', 'rb')) for file_id in file_ids}
    # Assign to local variables
    taskF_results, mu_results, xi_results, phi_results, E_results, nu_results = (results[f"{file_id}_results"] for file_id in file_ids)

# Add constraints 
add_constraints(model, config, x, y, Q, z, Z, Z_prime, u, p, phi_results, E_results, mu_results, taskF_results, xi_results, nu_results)

# Set objective functions  
psi = set_objective_functions(model, config, y, phi_results, p)

# Run optimization 
run_optimization(model, Gap, TimeLimit)

# Save results if optimal 
print(f'\Current version is {current_version}, and the model results is based on the {starting_version} files')

save_all_results(model, x, y, Q, z, Z, Z_prime,u, pkl_file_prefix)

