import json
import os
import pickle
import re
import shutil
import gurobipy as gp
import torch
import numpy as np
from gurobipy import GRB
from helper import map_model_to_filtered_indices


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


def top_k_importance_sampling(predict, k, m=None):
    if m is None:
        m = min(2 * k, predict.size(0))

    _, top_m_indices = torch.topk(predict, m)  # 取前 m 个高概率的约束索引
    candidate_probs = predict[top_m_indices]  # 获取候选集的概率值

    weights = candidate_probs / candidate_probs.sum()

    selected_indices = torch.multinomial(weights, k, replacement=False)
    selected_constraints = top_m_indices[selected_indices]

    return selected_constraints.tolist()


def constrain_search():
    # 从topk集合中，采样（如何采样）一个子集，作为
    pass


def enhance_solve(m, o_m, new_solution, BD, tight_pred, k0, k1, kc):
    model_to_filtered_index = map_model_to_filtered_indices(o_m)
    pre_cons, pre_sols = BD
    vars_list = o_m.getVars()
    assert len(vars_list) == len(pre_sols), "x_pred 的长度必须与模型变量数一致"

    sorted_indices = torch.argsort(pre_sols)
    topk0_indices = sorted_indices[:k0] if k0 != 0 else torch.tensor([])
    topk1_indices = sorted_indices[-k1:] if k1 != 0 else torch.tensor([])

    # k = 0
    # fixed_vars = set()
    # for idx in topk0_indices.tolist():
    #     var_name = vars_list[idx].VarName
    #     if var_name in new_solution and new_solution[var_name] == 0:
    #         k += 1
    #         var = vars_list[idx]
    #         var.LB = 0
    #         var.UB = 0
    #         fixed_vars.add(var_name)
    #
    # for idx in topk1_indices.tolist():
    #     var_name = vars_list[idx].VarName
    #     if var_name in new_solution and new_solution[var_name] == 1:
    #         k += 1
    #         var = vars_list[idx]
    #         var.LB = 1
    #         var.UB = 1
    #         fixed_vars.add(var_name)

    tight_constraints = {}
    num = len(o_m.getConstrs())
    for cind, constr in enumerate(m.getConstrs()):
        num -= 1
        slack = constr.getAttr(GRB.Attr.Slack)
        if abs(slack) < 1e-6 and constr.Sense != GRB.EQUAL:
            idx = model_to_filtered_index[cind]
            if idx not in tight_pred:
                tight_constraints[idx] = constr
        if num == 0:
            break
    k = 0
    min_topk = min(200, pre_cons.size(0))
    topk_indices = torch.topk(pre_cons, min_topk).indices
    for i in range(len(topk_indices) - kc):
        index = topk_indices[i + kc].item()
        if index in tight_constraints.keys():
            k += 1
            constr = tight_constraints[index]
            constr_name = constr.ConstrName
            o_constr = o_m.getConstrByName(constr_name)
            o_constr.Sense = GRB.EQUAL
    # for constr in tight_constraints:
    #     # 怎么取有讲究
    #     sample = np.random.normal()
    #     if sample > 0.5:
    #         k += 1
    #         constr_name = constr.ConstrName
    #         o_constr = o_m.getConstrByName(constr_name)
    #         o_constr.Sense = GRB.EQUAL
    o_m.update()

    return o_m


def project_to_feasible_region_and_get_tight_constraints(model, x_pred):
    """
    使用Gurobi将预测变量x_pred投影到可行域，并返回可行解和紧约束集。
    """
    proj_model = gp.Model("projection")

    proj_model.setParam('OutputFlag', 0)

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


def search(m, scores, delta):
    instance_variabels = m.getVars()
    instance_variabels.sort(key=lambda v: v.VarName)
    variabels_map = {}
    for v in instance_variabels:  # get a dict (variable map), varname:var clasee
        variabels_map[v.VarName] = v
    alphas = []
    for i in range(len(scores)):
        tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
        if x_star < 0:
            continue
        # tmp_var = m1.addVar(f'alp_{tar_var}', 'C')
        tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
        alphas.append(tmp_var)
        m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
    all_tmp = 0
    for tmp in alphas:
        all_tmp += tmp
    m.addConstr(all_tmp <= delta, name="sum_alpha")
    return m


def get_gp_best_objective(log_folder):
    best_objectives = []
    pattern = r"Best objective ([\d.e+-]+)"  # 正则表达式匹配 "Best objective " 后面的值
    filenames = sorted(os.listdir(log_folder))
    # 遍历文件夹中的所有文件
    for filename in filenames:
        filepath = os.path.join(log_folder, filename)

        # 确保只处理文件
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        # 查找匹配的值
                        match = re.search(pattern, line)
                        if match:
                            # 提取第一次出现的值并保存到列表中
                            best_objectives.append(float(match.group(1)))
                            break  # 只提取第一次出现的值
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    return best_objectives


def primal_integral_callback(model, where):
    """
    回调函数，用于记录求解过程中的 Primal Gap 和时间点
    """
    if where == gp.GRB.Callback.MIP:
        # 获取当前已用时间
        time_elapsed = model.cbGet(gp.GRB.Callback.RUNTIME)

        # 获取当前的上界和下界
        ub = model.cbGet(gp.GRB.Callback.MIP_OBJBST)  # Best solution (upper bound)
        lb = model.cbGet(gp.GRB.Callback.MIP_OBJBND)  # Best bound (lower bound)

        # 如果上下界合法，计算 Primal Gap 并记录
        if lb < gp.GRB.INFINITY and ub < gp.GRB.INFINITY and lb != 0:
            primal_gap = abs((ub - lb) / abs(lb))
            # 记录时间点和对应的 Primal Gap
            primal_integral_callback.gap_records.append((time_elapsed, primal_gap))


def pred_error(scores, ins_name_to_read, InstanceName, delta):
    TaskName = ins_name_to_read.split('_')[0]
    sols_files = f"./logs/{InstanceName}/{TaskName}_GRB_sols"
    sols_file = sols_files + "/" + ins_name_to_read + ".sol"
    with open(sols_file, 'rb') as f:
        sols = pickle.load(f)

    sorted_scores = sorted(scores, key=lambda x: x[0])
    total_count = 0
    correct_count = 0
    for score, sol in zip(sorted_scores, sols):
        variable_value = score[3]  # scores 中的变量值
        if variable_value in [0, 1]:  # 排除 -1 的情况
            total_count += 1
            if variable_value == sol:  # 比较值是否相同
                correct_count += 1
            else:
                continue
    error = total_count - correct_count
    return error


if __name__ == "__main__":
    file = "./instance/test/CA/CA_69.lp"
    log_file = "./logs/CA/CA_GRB_Predect&Search"
    # main(file, n=0, m=30)
    primal_integral_callback.gap_records = []

    # 创建模型并优化
    model = gp.read(file)
    model.optimize(primal_integral_callback)

    # 计算 Primal Integral
    gap_records = primal_integral_callback.gap_records
    primal_integral = 0.0

    # 确保记录有数据
    if len(gap_records) > 1:
        for i in range(1, len(gap_records)):
            t1, gap1 = gap_records[i - 1]
            t2, gap2 = gap_records[i]
            # 梯形积分法计算 Primal Integral
            primal_integral += (gap1 + gap2) / 2 * (t2 - t1)

    print(f"Primal Integral: {primal_integral:.4f}")
