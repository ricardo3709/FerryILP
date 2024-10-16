import gurobipy as gp
import pandas as pd
import os

def run_optimization(model):
    print(f"Starting optimization with {model.NumVars} variables and {model.NumConstrs} constraints.\n")

    # Modify model parameters

    # Basic settings
    model.setParam('OutputFlag', 1)       # Enable output logs
    model.setParam('InfUnbdInfo', 1)      # Output information on infeasible or unbounded models
    model.setParam('Presolve', 1)  
    model.setParam('ScaleFlag', 0)  

    # Optimization strategies
    model.setParam('NumericFocus', 3)  # Improve numerical stability
    model.setParam('MIPGap', 0.2)  # Optimality gap, change to smaller value later
    model.setParam('Method', 2)  # Try dual simplex to avoid barrier instability

    model.optimize()

    print(f"Optimization completed with status: {model.Status}")

    # Check if a solution was found
    objects_name = {0:"Vessel utilization", 1:"Total rebalancing time"}

    if model.SolCount > 0:
        print("Solution found!")
        # Retrieve the solution
        print(f"Objective Function Value: {model.ObjVal}") ## single obj function

        # # Check obj seperately
        # for i in range(model.NumObj): 
        #     model.setParam('ObjNumber', i)
        #     print(f"{objects_name[i]} Value: {model.ObjNVal}")
        # solution = {v.VarName: v.X for v in model.getVars()}

        # Further processing of the solution
    else:
        print("No solution found within the time limit.")

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

def save_relaxed_variable_results(model, var_dict, filename):
    # Check if the model is infeasible
    if model.Status == gp.GRB.INFEASIBLE:
        print("Model is infeasible. Attempting to compute a feasibility relaxation.")
        # Compute a feasibility relaxation of the model
        model.feasRelaxS(0, False, False, True)
        model.optimize()

        if model.Status == gp.GRB.OPTIMAL:
            results = {
                k: (var_dict[k].X if var_dict[k].X <= var_dict[k].UB and var_dict[k].X >= var_dict[k].LB else "Out of bounds")
                for k in var_dict.keys()
            }
            df = pd.DataFrame(list(results.items()), columns=['Variable', 'Value'])
            df.to_csv(filename, index=False)
            print(f"Feasibility relaxation results saved to {filename} with {len(results)} entries.")
        else:
            print("Unable to find a feasible relaxed solution.")
    else:
        print("\nModel status is not infeasible. No need to relax anything.")

def save_all_results(model, x, y, Q, z, Z, Z_prime,file_prefix):
    if model.Status == gp.GRB.OPTIMAL:
        print("\nOptimization was successful. Saving results...")
        
        output_dir = 'ILPimplementation/output_files'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        save_variable_results(x, os.path.join(output_dir, f'{file_prefix}_x_ld_results.csv'))
        save_variable_results(y, os.path.join(output_dir, f'{file_prefix}_y_vjt_results.csv'))
        save_variable_results(Q, os.path.join(output_dir, f'{file_prefix}_Q_vt_results.csv'))
        save_variable_results(z, os.path.join(output_dir, f'{file_prefix}_z_wj_results.csv'))
        save_variable_results(Z, os.path.join(output_dir, f'{file_prefix}_Z_lwt_results.csv'))
        save_variable_results(Z_prime, os.path.join(output_dir, f'{file_prefix}_Z_prime_lwt_results.csv'))
    else:
        print("\nOptimization did not reach optimality.")

