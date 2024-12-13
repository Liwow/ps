import gurobipy
import json
from gurobipy import GRB
import argparse
import utils
import helper
import test
from test import validate_solution_in_original_model, change_model
import random
import os
import numpy as np
import torch
from time import time
from helper import get_a_new2, get_bigraph, get_pattern

# def modify(model, n=0, k=0, fix=0):
#     # fix 0:no fix 1:随机 2:排序 3: 交集
#     if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
#         print("No optimal solution found.")
#         return
#     slacks = {constr: constr.slack for constr in model.getConstrs() if constr.Sense in ['<', '>']}
#     filter_indices = [
#         model_to_filtered_index[i] for i in tc_1 if i in model_to_filtered_index
#     ]
#     most_tight_constraints, count_tight = test.get_most_tight_constraints(slacks, filter_indices)
#     print(f"最优解和预测投影解中松弛度都为 0 的相同约束个数: {count_tight}")
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
#         most_tight_constraints = [constr for constr, slack in sorted_slacks[-k:]] if k > 0 else []
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


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
multimodal = False
TaskName = "CA___"
if TaskName == "CA_multi":
    multimodal = True
# load pretrained model
if TaskName == "IP":
    # Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy, postion_get
elif multimodal:
    from GCN import GNNPolicy_multimodal as GNNPolicy
else:
    # from GCN import GNNPolicy as GNNPolicy
    from GCN_class import GNNPolicy_class as GNNPolicy

model_name = f'{TaskName}.pth'
pathstr = f'./models/{model_name}'
policy = GNNPolicy().to(DEVICE)
state = torch.load(pathstr, map_location=torch.device('cuda:0'))
policy.load_state_dict(state)

# 4 public datasets, IS, WA, CA, IP
TaskName = 'CA'


def test_hyperparam(task):
    '''
    set the hyperparams
    k_0, k_1, delta
    '''
    if task == "IP":
        return 400, 5, 10
    elif task == "IS":
        return 300, 300, 20
    elif task == "WA":
        return 0, 500, 10
    elif task == "CA" or task == "CA_m":  # 600 0 1
        return 600, 0, 1
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


k_0, k_1, delta = test_hyperparam(TaskName)

# set log folder
solver = 'GRB'
test_task = f'{TaskName}_{solver}_Predect&Search'
if not os.path.isdir(f'./logs'):
    os.mkdir(f'./logs')
if not os.path.isdir(f'./logs/{TaskName}'):
    os.mkdir(f'./logs/{TaskName}')
if not os.path.isdir(f'./logs/{TaskName}/{test_task}'):
    os.mkdir(f'./logs/{TaskName}/{test_task}')
log_folder = f'./logs/{TaskName}/{test_task}'

sample_names = sorted(os.listdir(f'./instance/test/{TaskName}'))
count = 0
subop_total = 0
infeas_total = 0
time_total = 0
max_subop = -1
max_infeas = -1
max_time = 0
gap_total = 0

ALL_Test = 100
epoch = 3
TestNum = round(ALL_Test / epoch)

for e in range(epoch):
    for ins_num in range(TestNum):
        t_start = time()
        test_ins_name = sample_names[(0 + e) * TestNum + ins_num]
        ins_name_to_read = f'./instance/test/{TaskName}/{test_ins_name}'
        # get bipartite graph as input
        v_class_name, c_class_name = get_pattern("./task_config.json", TaskName)
        A, v_map, v_nodes, c_nodes, b_vars, v_class, c_class, _ = get_bigraph(ins_name_to_read, v_class_name, c_class_name)
        # A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_name_to_read)
        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = 1  # remove nan value
        variable_features = v_nodes
        if TaskName == "IP":
            variable_features = postion_get(variable_features)
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        v_class = utils.convert_class_to_labels(v_class, variable_features.shape[0])
        c_class = utils.convert_class_to_labels(c_class, constraint_features.shape[0])

        # prediction
        BD = policy(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
            torch.LongTensor(v_class).to(DEVICE),
            torch.LongTensor(c_class).to(DEVICE),
        ).cpu().squeeze()
        x_pred = (BD > 0.5).float()

        # align the variable name betweend the output and the solver
        all_varname = []
        for name in v_map:
            all_varname.append(name)
        binary_name = [all_varname[i] for i in b_vars]
        scores = []  # get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(v_map)):
            type = "C"
            if all_varname[i] in binary_name:
                type = 'BINARY'
            scores.append([i, all_varname[i], BD[i].item(), -1, type])

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

        # read instance
        m = gurobipy.read(ins_name_to_read)
        model_to_filtered_index = helper.map_model_to_filtered_indices(m)
        # 修复预测初始解，得到初始可行解
        # _, tc_1 = test.project_to_feasible_region_and_get_tight_constraints(m, x_pred)

        m.Params.TimeLimit = 1000
        m.Params.Threads = 32
        m.Params.LogFile = f'{log_folder}/{test_ins_name}.log'
        gurobipy.setParam('LogToConsole', 1)

        # m.optimize()
        # obj = m.objVal
        obj = 23791.7
        print("gurobi 最优解：", obj)
        m.reset()
        m.Params.TimeLimit = 1000
        m.Params.Threads = 8

        # trust region method implemented by adding constraints
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
        m.update()
        m.optimize()

        # new_solution = {var.VarName: var.X for var in m.getVars()}
        # validate_solution_in_original_model(o_m, new_solution)

        t = time() - t_start
        time_total += t
        pre_obj = m.objVal
        if max_time <= t:
            max_time = t
        if m.status == GRB.OPTIMAL:
            subop = abs(pre_obj - obj) / abs(obj)
            subop_total += subop
            if subop > max_subop:
                max_subop = subop
            if subop < 1e-4:
                count += 1
            print(f"ps 最优值：{pre_obj}; subopt: {round(subop, 4)}")
        if m.status == GRB.TIME_LIMIT:
            mip_gap = m.MIPGap
            gap_total += mip_gap
            print(f"ps 最优值：{pre_obj}; gap: {round(mip_gap, 4)}")
        else:
            print("不可行")

total_num = TestNum * epoch
mean_gap = gap_total / TestNum / epoch
results = {
    "acc": count / total_num,
    "avg_subopt": round(subop_total / total_num, 6),
    "max_subopt": round(max_subop, 6),
    "mean_time_pred": round(time_total / total_num, 6),
    "max_time": round(max_time, 6)
}
results_dir = f"/home/ljj/project/predict_and_search/results/{TaskName}/"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
with open(results_dir + "results.json", "w") as file:
    json.dump(results, file)
print("acc: ", results['acc'])
print("avg_subopt： ", results['avg_subopt'])
print("max_subopt: ", results['max_subopt'])