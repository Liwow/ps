import pickle

import gurobipy
import json
from gurobipy import GRB
import argparse
import gc
import helper
import gp_tools
import random
import os
import numpy as np
import torch
from time import time
from helper import get_a_new2
import logging

# todo device设置，需要与model统一
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# def modify(model, n=0, k=0, fix=0):
#     # fix 0:no fix 1:随机 2:排序 3: 交集
#     if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
#         print("No optimal solution found.")
#         return
#     slacks = {constr: constr.slack for constr in model.getConstrs() if constr.Sense in ['<', '>']}
#     # filter_indices = [
#     #     model_to_filtered_index[i] for i in tc_1 if i in model_to_filtered_index
#     # ]
#     # most_tight_constraints, count_tight = test.get_most_tight_constraints(slacks, filter_indices)
#     # print(f"最优解和预测投影解中松弛度都为 0 的相同约束个数: {count_tight}")
#     if fix == 0:
#         print("****** do nothing! *********")
#         return
#     sorted_slacks = sorted(slacks.items(), key=lambda item: abs(item[1]), reverse=True)
#     if n != 0:
#         most_relaxed_constraints = [constr for constr, slack in sorted_slacks[:n]]
#         print(f"Removing {n} most relaxed constraints.")
#         for constr in most_relaxed_constraints:
#             m.remove(constr)
#
#     tight_constraints = [constr for constr, slack in slacks.items() if slack == 0]
#
#     if fix == 1:
#         print("groundtruth 固定约束,固定方式:随机")
#         random.shuffle(tight_constraints)
#         most_tight_constraints = tight_constraints[:k]
#     elif fix == 2:
#         print("groundtruth 固定约束,固定方式:排序")
#         most_tight_constraints = [constr for constr, slack in sorted_slacks[-k:0]] if k > 0 else []
#     elif fix == 3:
#         print("groundtruth 固定约束,固定方式:交集")
#     for constr in most_tight_constraints:
#         row = model.getRow(constr)
#         coeffs = []
#         vars = []
#         for i in range(row.size()):
#             coeffs.append(row.getCoeff(i))
#             vars.append(row.getVar(i))
#         model.remove(constr)
#         model.addConstr(gurobipy.LinExpr(coeffs, vars) == constr.RHS, name=f"{constr.ConstrName}_tight")


def modify_by_predict(model, predict, k=0, fix=0, th=30, n=0):
    # min_topk = min(250, pre_t.size(0))
    # top_indices = torch.topk(pre_t, min_topk).indices
    # tight_mask = torch.zeros_like(pre_t)
    # tight_mask[top_indices] = 1
    # tight_constraints = torch.nonzero(tight_mask == 1, as_tuple=True)[0]

    min_topk = min(k, predict.size(0))
    topk_indices = torch.topk(predict, min_topk).indices
    all_indices = torch.topk(predict, 3 * kc).indices.tolist()
    critical_mask = torch.zeros_like(predict)
    critical_mask[topk_indices] = 1
    critical_constraints = torch.nonzero(critical_mask == 1, as_tuple=True)[0]
    ct_constraints = critical_constraints
    wrong_indices = []
    # ct_constraints = critical_constraints[torch.isin(critical_constraints, tight_constraints)].tolist()
    # filter_indices = [
    #     model_to_filtered_index[i] for i in tc_1 if i in model_to_filtered_index
    # ]
    # most_tight_constraints = list(set(tight_constraints) & set(filter_indices))
    # print(f"预测约束和预测投影解中松弛度都为 0 的相同约束个数: {len(most_tight_constraints)}")
    # todo 这里是调试的，跑结果可以注释掉，与grb有关
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        slacks = {constr: constr.slack for constr in model.getConstrs() if constr.Sense in ['<', '>']}
        slacks_indices = [i for i, constr in enumerate(model.getConstrs())
                          if constr.Sense in ['<', '>'] and constr.Slack == 0]
        slacks_indices = [
            model_to_filtered_index[i] for i in slacks_indices if i in model_to_filtered_index
        ]
        all_indices = torch.topk(predict, len(slacks_indices)).indices.tolist()
        correct_tight_list = list(set(slacks_indices) & set(all_indices))
        # print("correct_tight_list number: ", len(correct_tight_list))
        wrong_indices = [i for i, value in enumerate(all_indices) if value not in slacks_indices]
        print(f"预测错的元素索引: {wrong_indices[:200]}")
        with open(log_file, 'a') as f:
            f.write(f"预测错的元素索引: {wrong_indices[:200]}")
        most_tight_constraints, count_tight = gp_tools.get_most_tight_constraints(slacks, ct_constraints)
        # print(f"最优解和预测约束中松弛度都为 0 的相同约束个数: {count_tight}")
    if fix == 0:
        print("****** predict do nothing! *********")
        return
    # todo 下面是固定约束部分，需要改成optv
    m.reset()
    cons = model.getConstrs()
    remove_num = 0
    fixed_constraints = []
    for i in all_indices[:2 * kc]:
        idx_in_model = filtered_index_to_model[i]
        c = cons[idx_in_model]
        fixed_constraints.append(c.ConstrName)

    for idx, c in enumerate(cons):
        if idx in model_to_filtered_index.keys() and model_to_filtered_index[idx] in ct_constraints:
            row = model.getRow(c)
            coeffs = []
            vars = []
            for i in range(row.size()):
                coeffs.append(row.getCoeff(i))
                vars.append(row.getVar(i))
            if c.Sense in ['<', '>'] and len(vars) < th:
                remove_num += 1
                model.remove(c)
                model.addConstr(gurobipy.LinExpr(coeffs, vars) == c.RHS, name=f"{c.ConstrName}_tight")
    print("remove_num: ", remove_num, ", 阈值：", th)
    # todo 这里保存了固定的约束索引和前200个预测错的索引，调试需要
    data_to_save = {
        "fixed_constraints_name": fixed_constraints,
        "wrong_indices": wrong_indices[:200]
    }
    if not os.path.exists(f"{results_dir}/{TaskName}_{test_ins_name.split('.')[0]}.json"):
        print("store tight")
        with open(f"{results_dir}/{TaskName}_{test_ins_name.split('.')[0]}.json", "w") as data_file:
            json.dump(data_to_save, data_file, ensure_ascii=False, indent=4)
    return ct_constraints


random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
multimodal = False
is_solver = False
# TaskName = "CA_m_ccon"
TaskName = "case118"
if TaskName == "CA_multi":
    multimodal = True
# load pretrained model
if TaskName == "IP":
    # Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy, postion_get
elif multimodal:
    from GCN import GNNPolicy_multimodal as GNNPolicy
else:
    from GCN import GNNPolicy_constraint as GNNPolicy

model_name = f'{TaskName}.pth'
pathstr = f'./models/{model_name}'
policy = GNNPolicy().to(DEVICE)
state = torch.load(pathstr, map_location=DEVICE)
policy.load_state_dict(state)

policy.eval()
# todo 在这里修改测试数据集 case1951rte case2868rte case2869pegase
TaskName = 'case2869pegase'


def test_hyperparam(task):
    '''
    set the hyperparams
    k_0, k_1, delta, kc
    '''
    if task == "IP":
        return 400, 5, 10
    elif task == "IS":
        return 300, 300, 20
    elif task == "WA":
        return 0, 500, 10
    elif task == "CA":  # 600 0 1
        return 600, 0, 1, 10
    elif task == "beasley":
        return 50, 17, 10
    elif task == "ns":
        return 120, 18, 20
    elif task == "binkar":
        return 54, 24, 10
    elif task == "neos":
        return 20129, 569, 700  # 20741 609
    elif task == "mas":
        return 136, 14, 10
    elif task == "markshare":
        return 14, 12, 9
    elif task == "case118":  # 10000,1164,18878
        return 2000, 0, 100, 500
    elif task == "case300":  # 13137,1767,22470
        return 2000, 0, 100, 500
    elif task == "case1951rte":  # 80000, 2000, 15w
        return 20000, 50, 300, 4000
    elif task == "case6515rte":
        return 50000, 1200, 1200, 10000
    elif task == "case2868rte":
        return 30000, 50, 500, 4000
    elif task == "case2869pegase":
        return 22000, 50, 600, 4000


k_0, k_1, delta, kc = test_hyperparam(TaskName)
# kc = 4000
# set log folder
solver = 'GRB'
test_task = f'{TaskName}_{solver}_Predict&Search'
if not os.path.isdir(f'./logs'):
    os.mkdir(f'./logs')
if not os.path.isdir(f'./logs/{TaskName}'):
    os.mkdir(f'./logs/{TaskName}')
if not os.path.isdir(f'./logs/{TaskName}/{test_task}'):
    os.mkdir(f'./logs/{TaskName}/{test_task}')
log_folder = f'./logs/{TaskName}/{test_task}'

# todo 这里是设置结果保存的文件夹，路径需要修改
results_dir = f"/home/ljj/project/predict_and_search/results/{TaskName}/"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

sample_names = sorted(os.listdir(f'./instance/test/{TaskName}'))

acc = 0
subop_total = 0
time_total_ps = 0
time_totol_solver = 0

# todo 这里可以修改测试的instance数量
ALL_Test = len(sample_names)
epoch = 1
TestNum = round(ALL_Test / epoch)

if not is_solver:
    gp_obj_list = gp_tools.get_gp_best_objective(f'./logs/{TaskName}/{test_task}')
else:
    gp_obj_list = []

for e in range(epoch):
    for ins_num in range(TestNum):
        test_ins_name = sample_names[e * TestNum + ins_num]
        ins_name_to_read = f'./instance/test/{TaskName}/{test_ins_name}'
        # get bipartite graph as input
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_name_to_read)
        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = 1  # remove nan value
        variable_features = v_nodes
        if TaskName == "IP":
            variable_features = postion_get(variable_features)
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        # edge_features = torch.ones(edge_features.shape)

        m = gurobipy.read(ins_name_to_read)
        cons = m.getConstrs()
        c_masks = [1 if con.sense in ['<', '>'] else 0 for con in cons]
        c_masks.append(0)  # add obj
        c_masks = torch.tensor(c_masks, dtype=torch.float32)

        # prediction
        BD = policy(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
            c_masks.to(DEVICE)
        )

        pre_cons, pre_sols = BD
        pre_cons = pre_cons.cpu().squeeze()
        pre_sols = pre_sols.cpu().squeeze()
        # pre_t = pre_t.cpu().squeeze()
        x_pred = torch.round(pre_sols)

        # align the variable name between the output and the solver
        all_varname = []
        for name in v_map:
            all_varname.append(name)
        binary_name = [all_varname[i] for i in b_vars]
        scores = []  # get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(v_map)):
            type = "C"
            if all_varname[i] in binary_name:
                type = 'BINARY'
            scores.append([i, all_varname[i], pre_sols[i].item(), -1, type])

        scores.sort(key=lambda x: x[2], reverse=True)

        scores = [x for x in scores if x[4] == 'BINARY']  # get binary

        fixer = 0
        # fixing variable picked by confidence scores
        count1 = 0
        for i in range(len(scores)):
            if count1 < k_1:
                scores[i][3] = 1
                count1 += 1
                fixer += 1
        scores.sort(key=lambda x: x[2], reverse=False)
        count0 = 0
        for i in range(len(scores)):
            if count0 < k_0:
                scores[i][3] = 0
                count0 += 1
                fixer += 1

        print(f'instance: {test_ins_name}, '
              f'fix {k_0} 0s and '
              f'fix {k_1} 1s, delta {delta}. ')

        # todo 这里是得到非等式约束集合与全部约束集合 2个集合索引间的map 用到了grb，需要修改为optv
        model_to_filtered_index, filtered_index_to_model = helper.map_model_to_filtered_indices(m)
        # 修复预测初始解，得到初始可行解
        # _, tc_1 = test.project_to_feasible_region_and_get_tight_constraints(m, x_pred)
        # o_m = m.copy()
        # todo 下面是grb的代码，需要修改为optv
        m.Params.TimeLimit = 1000
        m.Params.MIPFocus = 1
        gurobipy.setParam('LogToConsole', 1)
        log_file = f'{log_folder}/{test_ins_name}.log'
        m.Params.LogFile = log_file

        t_start_1 = time()
        # todo is_solver为True，即需要使用求解器求解一遍，会保存结果和sol，第二次就不用再重新求解器求解了，需要改成optv, 如果改的麻烦也可以注释掉，最好还是改了方便后面调试
        if is_solver:
            output_folder = f"./logs/{TaskName}/{TaskName}_GRB_sols"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"{test_ins_name}.sol")
            m.optimize()
            obj = m.objVal
            integer_sols = sorted(
                [(v.varName, v.x) for v in m.getVars() if v.vType in ['I', 'B']],  # 获取变量名和值
                key=lambda x: x[0]  # 按变量名字典序排序
            )
            integer_sols = [value for _, value in integer_sols]
            if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                with open(output_file, "wb") as f:
                    pickle.dump(integer_sols, f)
        else:
            obj = gp_obj_list[(0 + e) * TestNum + ins_num] if len(gp_obj_list) > 0 else 0
        time_totol_solver += (time() - t_start_1)

        error = gp_tools.pred_error(scores, test_ins_name, TaskName)
        acc = acc + 1 if error <= delta else acc
        print(f"gurobi 最优解：{obj}; pred_error is: {error}")

        # fix 0:no fix 1:随机 2:排序 3: 交集
        # modify(m, n=0, k=100, fix=0)  # if fix=0  do nothing
        # todo 这里是固定约束函数，需要改成optv
        tight_constraints = modify_by_predict(m, pre_cons, k=kc, fix=1)

        # trust region method implemented by adding constraints
        # todo 这里是固定变量函数，需要改成optv，gp_tools里的函数基本都跟grb有关，用不到的函数可以删掉，用到的函数需要改成optv
        m = gp_tools.search(m, scores, delta)
        # todo 求解，需要改为optv
        m.update()
        m.Params.TimeLimit = 1000
        m.Params.MIPFocus = 1
        t_start_2 = time()
        m.optimize()
        # new_solution = {var.VarName: var.X for var in m.getVars()}
        # test.validate_solution_in_original_model(o_m, new_solution)
        # o_m = test.enhance_solve(m, o_m, new_solution, BD, tight_constraints, k_0, k_1, kc)
        # o_m = test.search(o_m, scores, delta)
        # o_m.update()
        # o_m.optimize()
        # print(f"search 最优值：{o_m.objVal}")

        del BD, A, v_nodes, c_nodes, edge_indices, edge_features, b_vars, c_masks, constraint_features, pre_cons, pre_sols, variable_features, x_pred
        torch.cuda.empty_cache()
        gc.collect()

        t_ps = time() - t_start_2
        time_total_ps += t_ps

        # todo 这里是计算subopt，需要改成optv
        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            pre_obj = m.objVal
            subop = (pre_obj - obj) / (obj + 1e-8) if TaskName == "CA" else (obj - pre_obj) / (obj + 1e-8)
            subop_total += subop
            print(
                f"solver 最优值：{obj}; ps 最优值：{pre_obj}; subopt: {round(subop / (ins_num + 1), 6)}; pred_error: {round(acc / (ins_num + 1), 4)}")
        else:
            print("不可行")
        torch.cuda.empty_cache()

total_num = TestNum * epoch
results = {
    "avg_subopt": round(subop_total / total_num, 6),
    "mean_time_pred_ps": round(time_total_ps / total_num, 6),
    "mean_time_solver": round(time_totol_solver / total_num, 6),
}

with open(results_dir + "results.json", "a") as file:
    json.dump(results, file)
    file.write("\n")

print("avg_time_pred_ps: ", results['mean_time_pred_ps'])
print("avg_time_solver: ", results['mean_time_solver'])
print("avg_subopt： ", results['avg_subopt'])
