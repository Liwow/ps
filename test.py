import os
import shutil
import gurobipy as gp
import torch
from gurobipy import GRB


def ss():
    # 原始文件夹和目标文件夹路径
    src_folder = './instance/test/CA_m'
    dst_folder = './instance/test/CA'

    # 遍历并移动重命名文件
    for i in range(100):
        src_filename = f'CA_m_{i}.lp'
        dst_filename = f'CA_{i}.lp'

        # 源文件和目标文件的完整路径
        src_path = os.path.join(src_folder, src_filename)
        dst_path = os.path.join(dst_folder, dst_filename)

        # 检查源文件是否存在
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Source file {src_path} not found.")


def project_to_feasible_region_and_get_tight_constraints(model, x_pred):
    """
    使用Gurobi将预测变量x_pred投影到可行域，并返回可行解和紧约束集。
    """
    # 创建新的Gurobi模型
    proj_model = gp.Model("projection")

    # 设置参数以静默运行
    proj_model.setParam('OutputFlag', 0)

    # 从原模型复制变量，并构建映射：添加新的决策变量
    x = proj_model.addVars(len(x_pred), lb=0, ub=1, vtype=GRB.BINARY, name="x")

    # 添加约束：从原模型复制所有约束
    constraints = []  # 用于存储新模型中的约束引用
    for constr in model.getConstrs():
        lin_expr = model.getRow(constr)
        new_expr = gp.LinExpr()
        for i in range(lin_expr.size()):
            var = lin_expr.getVar(i)  # 获取第 i 个变量
            coeff = lin_expr.getCoeff(i)  # 获取对应的系数
            new_expr.addTerms(coeff, x[var.index])  # 添加到新的表达式中
        new_constr = proj_model.addConstr(new_expr <= constr.RHS)
        constraints.append(new_constr)

    # 构造投影的目标函数：min ||x - x_pred||_2
    objective = gp.quicksum((x[i] - x_pred[i]) ** 2 for i in range(len(x_pred)))

    # 设置模型的目标函数
    proj_model.setObjective(objective, GRB.MINIMIZE)

    # 求解模型
    proj_model.optimize()

    # 检查模型是否找到最优解
    if proj_model.status != GRB.OPTIMAL:
        raise ValueError("无法找到可行解")

    print("proj_obj", proj_model.objVal)
    x_feasible = torch.tensor([x[i].x for i in range(len(x_pred))])

    # 找到紧约束集（dual slack = 0）
    tight_constraints = []
    for i, constr in enumerate(constraints):
        if constr.Sense in ['<', '>'] and constr.getAttr('slack') == 0:  # 检查松弛度
            tight_constraints.append(i)

    return x_feasible, tight_constraints


def get_most_tight_constraints(slacks, tight_indices):
    """
    根据给定的 slacks 字典和 tight_indices 列表，
    找到松弛度为 0 且出现在这两个集合中的约束，并返回这些约束的列表。

    参数：
    - slacks: dict，{constr: slack_value} 表示每个约束及其对应的松弛度
    - tight_indices: list，包含紧约束的索引（从 0 开始）

    返回：
    - most_tight_constraints: list，松弛度为 0 且在索引列表中的约束
    - count: int，满足条件的约束的数量
    """
    # 遍历 slacks 字典，并根据索引过滤出紧约束
    most_tight_constraints = [
        constr for i, constr in enumerate(slacks.keys())
        if i in tight_indices and slacks[constr] == 0
    ]

    # 返回满足条件的约束列表及其数量
    return most_tight_constraints, len(most_tight_constraints)


def validate_solution_in_original_model(original_model, new_solution):
    for constr in original_model.getConstrs():
        # 获取当前约束的左侧系数行
        row = original_model.getRow(constr)

        # 计算左侧值 (LHS) = Σ (系数 * 变量的取值)
        lhs_value = 0
        for i in range(row.size()):
            var = row.getVar(i)  # 获取变量
            coeff = row.getCoeff(i)  # 获取系数
            if var.VarName in new_solution:
                lhs_value += coeff * new_solution[var.VarName]  # 新解的取值

        # 根据约束类型 (<=, >=, =) 来验证是否满足约束
        if constr.Sense == GRB.LESS_EQUAL and lhs_value > constr.RHS:
            print(f"Constraint {constr.ConstrName} violated: LHS = {lhs_value}, RHS = {constr.RHS}")
            return False
        elif constr.Sense == GRB.GREATER_EQUAL and lhs_value < constr.RHS:
            print(f"Constraint {constr.ConstrName} violated: LHS = {lhs_value}, RHS = {constr.RHS}")
            return False
        elif constr.Sense == GRB.EQUAL and lhs_value != constr.RHS:
            print(f"Constraint {constr.ConstrName} violated: LHS = {lhs_value}, RHS = {constr.RHS}")
            return False

    print("New solution is feasible in the original model.")
    return True


def change_model(model, n=0, m=0):
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print("No optimal solution found.")
        return None

    original_obj_val = model.ObjVal

    # 获取所有约束的松弛度
    slacks = {constr: constr.slack for constr in model.getConstrs() if constr.Sense in ['<', '>']}

    # 根据松弛度进行排序，取最松弛的 n 个约束
    sorted_slacks = sorted(slacks.items(), key=lambda item: abs(item[1]), reverse=True)
    most_relaxed_constraints = [constr for constr, slack in sorted_slacks[:n]]
    print(f"Removing {n} most relaxed constraints.")
    for constr in most_relaxed_constraints:
        model.remove(constr)

    most_tight_constraints = [constr for constr, slack in sorted_slacks[-m:]] if m >= 0 else []
    for constr in most_tight_constraints:
        row = model.getRow(constr)
        coeffs = []
        vars = []
        for i in range(row.size()):
            coeffs.append(row.getCoeff(i))
            vars.append(row.getVar(i))
        model.remove(constr)
        model.addConstr(gp.LinExpr(coeffs, vars) == constr.RHS + 1e-6, name=f"{constr.ConstrName}_tight")

    # 更新模型以反映约束的变化
    model.update()
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        new_obj_val = model.ObjVal
        print(f"Original Objective Value: {original_obj_val}")
        print(f"New Objective Value after removing {n} most relaxed constraints: {new_obj_val}")
        new_solution = {var.VarName: var.X for var in model.getVars()}

        return new_solution
    else:
        print("No feasible solution found after removing constraints.")


def main(file_path, n=0, m=0):
    # n: 去掉的松弛约束的个数
    # m: 转为紧约束的个数
    try:
        model = gp.read(file_path)
        original_model = model.copy()

        new_solution = change_model(model, n, m)
        if new_solution is not None:
            feasible_in_original = \
                validate_solution_in_original_model(original_model, new_solution)
            if feasible_in_original:
                print("The new solution is feasible in the original model.")
            else:
                print("The new solution is not feasible in the original model.")

    except gp.GurobiError as e:
        print(f"Gurobi Error: {str(e)}")
    except Exception as e:
        print(f"General Error: {str(e)}")


if __name__ == "__main__":
    file = "./instance/test/CA_m/CA_m_69.lp"
    main(file, n=0, m=30)

