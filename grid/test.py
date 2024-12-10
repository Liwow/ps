import json

import gurobipy
from gurobipy import GRB
from pyscipopt import Model as SCIPModel
import gurobipy as gp
import warnings
import random
import time

import helper


def get_tight_scip(model):
    pass


def fix_constraints_scip(model, fixed_constraints):
    pass


def get_tight_gurobi(model):
    # Get non-equality tight constraints
    tight_constraints = []
    for constr in model.getConstrs():
        if constr.Sense != "=":  # Non-equality constraint
            slack = constr.Slack
            if abs(slack) < 1e-6:  # Tight constraint
                tight_constraints.append(constr)

    print(f"Number of tight constraints: {len(tight_constraints)}")
    return tight_constraints


def solve_lp_with_solver(lp_file, solver="scip", fix_percentage=0.3, max_time=1000):
    """
    Solve an LP problem using SCIP or Gurobi, identify tight constraints, fix a percentage, and re-solve.

    Parameters:
    - lp_file (str): Path to the LP file.
    - solver (str): Solver to use ("scip" or "gurobi").
    - fix_percentage (float): Percentage of tight constraints to fix.
    """
    if solver.lower() == "scip":
        print(f"Using SCIP to solve: {lp_file}")
        # Initialize SCIP model
        model = SCIPModel()
        model.readProblem(lp_file)
        model.setParam("limits/time", max_time)
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        time1 = end_time - start_time

        status = model.getStatus()
        if status in ["optimal", "timelimit"]:
            print(status)
            print("Objective value:", model.getObjVal())

            tight_constraints = get_tight_scip(model)

            # Randomly select constraints to fix
            num_to_fix = int(len(tight_constraints) * fix_percentage)
            fixed_constraints = random.sample(tight_constraints, min(num_to_fix, len(tight_constraints)))
            print(f"Randomly fixing {len(fixed_constraints)} constraints.")

            m = SCIPModel()
            m.readProblem(lp_file)
            fix_constraints_scip(m, fixed_constraints)

            m.setParam("limits/time", max_time)
            start_time = time.time()
            m.optimize()
            end_time = time.time()

            # Print new results
            print(f"Re-solved in {end_time - start_time:.2f} seconds. first solve in {time1:.2f}")
            print(f"Objective value after fixing: {m.getObjVal()}")

        else:
            print(f"Solver finished with status: {status}")

    elif solver.lower() == "gurobi":
        print(f"Using Gurobi to solve: {lp_file}")
        try:
            # Initialize Gurobi model
            model = gp.read(lp_file)
            model.Params.TimeLimit = max_time
            start_time = time.time()
            model.optimize()
            end_time = time.time()
            time1 = end_time - start_time

            # Check for optimality
            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                print("Optimal solution found!")
                print("Objective value:", model.objVal)

                tight_constraints = get_tight_gurobi(model)
                # Randomly select constraints to fix
                num_to_fix = int(len(tight_constraints) * fix_percentage)
                fixed_constraints = random.sample(tight_constraints,
                                                  min(num_to_fix, len(tight_constraints))) if num_to_fix != 0 else []
                print(f"Randomly fixing {len(fixed_constraints)} constraints.")

                # Fix constraints and re-solve
                model.reset()
                for constr in fixed_constraints:
                    row = model.getRow(constr)
                    model.remove(constr)
                    rhs_value = constr.getAttr("RHS")
                    new_constr = model.addConstr(row, GRB.EQUAL, rhs_value)

                start_time = time.time()
                model.update()
                model.optimize()
                end_time = time.time()
                binary_vars_0 = 0
                binary_vars_1 = 0
                for var in model.getVars():
                    # 检查变量是否为二元变量
                    if var.vType == 'B':  # 'B' 表示二元变量
                        if var.X == 0:
                            binary_vars_0 += 1
                        elif var.X == 1:
                            binary_vars_1 += 1
                print("binary_vars_0:", binary_vars_0)
                print("binary_vars_1:", binary_vars_1)
                # Print new results
                print(f"Re-solved in {end_time - start_time:.2f} seconds. first solve in {time1:.2f}")
                print(f"Objective value after fixing: {model.objVal}")

            else:
                print(f"Solver finished with status: {model.status}")
        except Exception as e:
            print(f"Error with Gurobi: {e}")
    else:
        print("Invalid solver choice! Please choose 'scip' or 'gurobi'.")


def fix_lp_and_solve(lp_file, solver, fix_set, fix_number=2000, max_time=1000):
    if solver.lower() == "scip":
        pass
    elif solver.lower() == "gurobi":
        model = gp.read(lp_file)
        model.Params.TimeLimit = max_time
        n_fix = 0
        cons = model.getConstrs()
        for constr in cons:
            if not fix_set:
                break
            if n_fix >= fix_number:
                break
            if constr.ConstrName in fix_set:
                row = model.getRow(constr)
                index_in_fix_set = fix_index_map[constr.ConstrName]
                if index_in_fix_set in wrong_indices:
                    print(f"{constr.ConstrName} is not tight, wrong")
                    continue
                coeffs = []
                vars = []
                for i in range(row.size()):
                    coeffs.append(row.getCoeff(i))
                    vars.append(row.getVar(i))
                model.remove(constr)
                model.addConstr(gurobipy.LinExpr(coeffs, vars) == constr.RHS, name=f"{constr.ConstrName}_tight")
                fix_set.remove(constr.ConstrName)
                n_fix += 1

        start_time = time.time()
        model.update()
        model.optimize()
        end_time = time.time()
        fix_solve_time = end_time - start_time
        print(f"固定了{n_fix}个约束")
        print(f"Fix-solved in {fix_solve_time:.2f} seconds.")
        print(f"Objective value after fixing: {model.objVal}")
    else:
        print("Invalid solver choice! Please choose 'scip' or 'gurobi'.")


def get_map(lp_file):
    model = gp.read(lp_file)
    index1, index2 = helper.map_model_to_filtered_indices(model)
    return index1, index2


# Example usage
lp_file = "case2869pegase_13.lp"
solver = "gurobi"
fix_number = 5000
model_to_filtered_index, filtered_index_to_model = get_map(lp_file)
with open("../case2869pegase_2017-12-31.json", "r") as file:
    data = json.load(file)
wrong_indices = data["wrong_indices"]
fixed_constraints = data["fixed_constraints_name"]

fixed_constraints_set = set(fixed_constraints[:fix_number])
fix_index_map = {name: idx for idx, name in enumerate(fixed_constraints)}
#5k(-54) 191  4k(-4) 158 3k(-3) 195  2k 213  0 213
fix_lp_and_solve(lp_file, solver, fixed_constraints_set, fix_number=fix_number)
# solve_lp_with_solver(lp_file, solver, fix_percentage=0.012)  # 141，213
