import gurobipy as gp
import pandas as pd
import os

def run_optimization(model, Gap, TimeLimit):
    print(f"Starting optimization with {model.NumVars} variables and {model.NumConstrs} constraints.\n")

    # Modify model parameters

    # Basic settings
    model.setParam('OutputFlag', 1) # Enable output logs
    model.setParam('InfUnbdInfo', 1) # Output information on infeasible or unbounded models

    # Presolve and numerical stability settings
    model.setParam('Presolve', 1) # Enable presolve to simplify the model
    model.setParam('ScaleFlag', 0) # Disable automatic model scaling, which might help avoid numerical issues
    model.setParam('NumericFocus', 3) # Increase numerical stability at the cost of performance (range: 0-3, higher = more stable)

    # Optimization method selection
    model.setParam('Method', 2) # Use the dual simplex method, more stable

   # Gap and time limit
    model.setParam('MIPGap', Gap)
    model.setParam('TimeLimit', TimeLimit) 

    model.optimize()

    print(f"Optimization completed with status: {model.Status}")

    # Check if a solution was found or partial solution exists
    objects_name = {0: "Vessel utilization", 1: "Total rebalancing time"}

    if model.SolCount > 0:  # Handle partial solutions as well
        print("Solution found or partially solved!")
        # Retrieve and print individual objective components if available
        try:
            print(f"Vessel Utilization Value: {model._vessel_utilization.getValue()}")
            print(f"Rebalancing Time Value: {model._rebalancing_time.getValue()}")
            print(f'Total penalise of tasks: {model._total_panelise.getValue()}') 
            
        except AttributeError:
            print("Objective components are not accessible or not defined in the model.")

        # Save the best solution found so far
        solution = {v.VarName: v.X for v in model.getVars()}
        # print("Solution Variables:")
        # for var_name, value in solution.items():
        #     print(f"{var_name}: {value}")

    else:
        print("No feasible solution found within the time limit.")

    # CHANGED: Handle cases where optimization is interrupted due to the time limit
    if model.Status == gp.GRB.TIME_LIMIT:
        # print("\nOptimization stopped due to time limit.")
        if model.SolCount > 0:
            print("A feasible solution was found within the time limit.")
            print(f"Best objective value within time limit: {model.ObjVal}")
        else:
            print("No feasible solution found within the time limit.")

    # Calculate IIS if model is infeasible
    if model.Status == gp.GRB.INFEASIBLE:
        print("Model is infeasible; computing IIS...")
        model.computeIIS()
        print("The following constraints and/or bounds are contributing to the infeasibility:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"{c.ConstrName} is in the IIS.")
        model.write("model.ilp")


def save_variable_results(var_dict, filename):
    results = {k: (var_dict[k].X if var_dict[k].X <= var_dict[k].UB and var_dict[k].X >= var_dict[k].LB else "Out of bounds") for k in var_dict.keys()}
    df = pd.DataFrame(list(results.items()), columns=['Variable', 'Value'])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename} with {len(results)} entries.")


def save_all_results(model, x, y, Q, z, Z, Z_prime,u, file_prefix):
    if model.Status == gp.GRB.OPTIMAL or model.SolCount > 0:  # Allow saving files even if not optimal
        print("\nOptimization was successful or partially solved. Saving results...")
        
        output_dir = 'ILPimplementation/output_files'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        save_variable_results(x, os.path.join(output_dir, f'{file_prefix}_x_ld_results.csv'))
        save_variable_results(y, os.path.join(output_dir, f'{file_prefix}_y_vjt_results.csv'))
        save_variable_results(Q, os.path.join(output_dir, f'{file_prefix}_Q_vt_results.csv'))
        save_variable_results(z, os.path.join(output_dir, f'{file_prefix}_z_wj_results.csv'))
        save_variable_results(Z, os.path.join(output_dir, f'{file_prefix}_Z_lwt_results.csv'))
        save_variable_results(Z_prime, os.path.join(output_dir, f'{file_prefix}_Z_prime_lwt_results.csv'))
        save_variable_results(u, os.path.join(output_dir, f'{file_prefix}_u_vwt_results.csv'))
    else:
        print("\nOptimization did not reach optimality or find a feasible solution.")
