import gurobipy as gp
from gurobipy import GRB

import gurobipy as gp
from gurobipy import GRB


def relax_constraints(constraints):
    """ Temporarily relax constraints by setting their bounds to infinity. """
    original_bounds = {}
    for constr in constraints:
        original_bounds[constr] = (constr.getAttr(GRB.Attr.RHS), constr.getAttr(GRB.Attr.Sense))
        if constr.getAttr(GRB.Attr.Sense) == GRB.LESS_EQUAL:  # a_i x <= b_i
            constr.setAttr(GRB.Attr.RHS, GRB.INFINITY)
        elif constr.getAttr(GRB.Attr.Sense) == GRB.GREATER_EQUAL:  # a_i x >= b_i
            constr.setAttr(GRB.Attr.RHS, -GRB.INFINITY)
        elif constr.getAttr(GRB.Attr.Sense) == GRB.EQUAL:  # a_i x = b_i
            constr.setAttr(GRB.Attr.RHS, GRB.INFINITY)  # Effectively disable the equality

    return original_bounds


def restore_constraints(constraints, original_bounds):
    """ Restore original constraint bounds. """
    for constr in constraints:
        rhs, sense = original_bounds[constr]
        constr.setAttr(GRB.Attr.RHS, rhs)
        constr.setAttr(GRB.Attr.Sense, sense)


def detect_redundant_constraints(lp_file, k=10):
    """
    Detect redundant constraints of type 'worker_used_ct_$i_$j'.
    :param lp_file: Path to the .lp file.
    :param k: Number of most violated constraints to add per iteration.
    :return: List of redundant constraints.
    """
    # Step 1: Load the model
    model = gp.read(lp_file)
    model.setParam("OutputFlag", 1)  # Suppress solver output for cleaner logs

    # Step 2: Identify all constraints of type 'worker_used_ct_$i_$j'
    all_constraints = model.getConstrs()
    target_constraints = [c for c in all_constraints if 'worker_used_ct' in c.ConstrName]

    # Step 3: Initialize
    active_constraints = []  # Active constraints during the iterations
    redundant_constraints = []  # Store redundant constraints

    # Step 4: Initial solve (relaxed problem)
    original_bounds = relax_constraints(target_constraints)  # Relax only target constraints
    model.update()
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError("Initial relaxed solve failed. Check model feasibility.")

    # Step 5: Iterative constraint screening
    while True:
        violated_constraints = []
        violation_values = []

        for constr in target_constraints:
            if constr not in active_constraints:
                slack = constr.getAttr(GRB.Attr.Slack)  # Slack value
                if slack < 0:  # Constraint is violated
                    violated_constraints.append(constr)
                    violation_values.append(abs(slack))  # Store violation magnitude

        if not violated_constraints:
            break

        # Sort violated constraints by the magnitude of violation
        sorted_constraints = sorted(zip(violation_values, violated_constraints), key=lambda x: -x[0])
        selected_constraints = [item[1] for item in sorted_constraints[:k]]  # Select top k violated constraints

        # Add top k violated constraints to the active set
        for constr in selected_constraints:
            active_constraints.append(constr)
            constr.setAttr(GRB.Attr.RHS, original_bounds[constr][0])  # Restore original RHS
            constr.setAttr(GRB.Attr.Sense, original_bounds[constr][1])  # Restore original Sense
        model.update()
        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError("Solve failed during iteration. Check model feasibility.")

    # Step 6: Identify redundant constraints
    for constr in target_constraints:
        if constr not in active_constraints:
            redundant_constraints.append(constr)

    # Step 7: Restore original constraints
    restore_constraints(target_constraints, original_bounds)
    model.update()

    # Step 8: Output results
    print(f"Total target constraints: {len(target_constraints)}")
    print(f"Redundant constraints: {len(redundant_constraints)}")
    print(f"Active constraints: {len(active_constraints)}")

    return redundant_constraints


# Example usage
lp_file = "2017-01-01.lp"
k = 5000  # Number of most violated constraints to add per iteration
redundant_constraints = detect_redundant_constraints(lp_file, k)

# Output redundant constraints
print(len(redundant_constraints))
