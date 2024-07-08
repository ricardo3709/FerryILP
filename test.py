# test_gurobi.py
import gurobipy as gp

try:
    # Create a new model
    m = gp.Model("test_model")
    # Create variables
    x = m.addVar(name="x")
    y = m.addVar(name="y")
    # Set objective
    m.setObjective(x + y, gp.GRB.MAXIMIZE)
    # Add constraint
    m.addConstr(x + 2 * y <= 8, "c0")
    m.optimize()

    for v in m.getVars():
        print(f'{v.varName}, {v.x}')

    print('Obj:', m.objVal)
except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
