import pickle
import gurobipy
import json
from gurobipy import GRB
from get_logits import plot_logits
import utils
from utils import test_hyperparam
import helper
import gp_tools
import gc
import random
import os
import numpy as np
import torch
from time import time
from helper import get_a_new2, get_bigraph, get_pattern
from GCN_class import getPE
from gp_tools import primal_integral_callback, get_gp_best_objective, pred_error


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
multimodal = False
position = False
gp_solve = False
ps_solve = False
Threads = 24
TimeLimit = 3600
gap_threshold = 0.005
ModelName = "WA_anchor"
model_name = f'{ModelName}.pth'
if ModelName == "CA_multi":
    multimodal = True
# load pretrained model
if ModelName.startswith("IP"):
    # Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy, postion_get
    position = True
elif multimodal:
    from GCN import GNNPolicy_multimodal as GNNPolicy
else:
    # from GCN import GNNPolicy as GNNPolicy
    from GCN_class import GNNPolicy_class as GNNPolicy

TaskName = ModelName.split("_")[0]
pathstr = f'./models/{model_name}'
policy = GNNPolicy(TaskName, position=position).to(DEVICE)
state = torch.load(pathstr, map_location=DEVICE)
policy.load_state_dict(state)
policy.eval()

# set log folder
solver = 'GRB'
instanceName = 'WA'
test_task = f'{instanceName}_{solver}_Predict&Search'
if not os.path.isdir(f'./logs'):
    os.mkdir(f'./logs')
if not os.path.isdir(f'./logs/{instanceName}'):
    os.mkdir(f'./logs/{instanceName}')
if not os.path.isdir(f'./logs/{instanceName}/{test_task}'):
    os.mkdir(f'./logs/{instanceName}/{test_task}')
log_folder = f'./logs/{instanceName}/{test_task}'
k_0, k_1, delta = test_hyperparam(instanceName)
sample_names = sorted(os.listdir(f'./instance/test/{instanceName}'))

subop_total = 0
time_total = 0
max_subop = -1
max_time = 0
ps_int_total = 0
gp_int_total = 0
gp_gap_time = 0
ps_gap_time = 0
acc = 0
acc_local = 0
mse_total = 0
ALL_Test = 15  # 30/60
epoch = 1
TestNum = round(ALL_Test / epoch)
if not gp_solve:
    gp_obj_list = get_gp_best_objective(f'./logs/{instanceName}/{test_task}')
else:
    gp_obj_list = []
for e in range(epoch):
    for ins_num in range(TestNum):
        test_ins_name = sample_names[(0 + e) * TestNum + ins_num]
        ins_name_to_read = f'./instance/test/{instanceName}/{test_ins_name}'
        # get bipartite graph as input
        v_class_name, c_class_name = get_pattern("./task_config.json", TaskName)
        A, v_map, v_nodes, c_nodes, b_vars, v_class, c_class, _ = get_bigraph(ins_name_to_read, v_class_name,
                                                                              c_class_name)
        # A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_name_to_read)
        constraint_features = c_nodes.cpu()
        variable_features = v_nodes
        constraint_features[torch.isnan(constraint_features)] = 1  # remove nan value
        # variable_features = getPE(v_nodes, position)

        # if TaskName == "IP":
        #     variable_features = postion_get(variable_features)
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        # edge_features = torch.ones(edge_features.shape)

        v_class = utils.convert_class_to_labels(v_class, variable_features.shape[0])
        c_class = utils.convert_class_to_labels(c_class, constraint_features.shape[0])

        # prediction
        get_logits = False
        BD = policy(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
            torch.LongTensor(v_class).to(DEVICE),
            torch.LongTensor(c_class).to(DEVICE),
            get_logits=get_logits
        )
        if get_logits:
            variable_features, constraint_features, BD = BD
            plot_logits(variable_features, constraint_features, v_class, c_class)
        BD = BD.cpu().squeeze()

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

        m.Params.TimeLimit = TimeLimit
        m.Params.Threads = Threads
        m.Params.MIPFocus = 1
        m.Params.LogFile = f'{log_folder}/{test_ins_name}.log'
        gurobipy.setParam('LogToConsole', 1)

        if gp_solve:
            primal_integral_callback.gap_records = []
            primal_integral_callback.gap_threshold = gap_threshold
            primal_integral_callback.point = None
            output_folder = f"./logs/{instanceName}/{TaskName}_GRB_sols"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"{test_ins_name}.sol")
            m.optimize(primal_integral_callback)
            obj = m.objVal
            integer_sols = sorted(
                [(v.varName, v.x) for v in m.getVars() if v.vType in ['I', 'B']],  # 获取变量名和值
                key=lambda x: x[0]  # 按变量名字典序排序
            )
            integer_sols = [value for _, value in integer_sols]
            if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                with open(output_file, "wb") as f:
                    pickle.dump(integer_sols, f)
            if m.status == GRB.TIME_LIMIT:
                primal_integral_callback.gap_records.append(
                    (m.Runtime, abs((m.objVal - m.ObjBound) / abs(m.ObjBound))))
            gp_gap_records = primal_integral_callback.gap_records
            ps_gap_time += primal_integral_callback.point[0] if primal_integral_callback.point is not None else 0
            primal_integral = 0.0
            if len(gp_gap_records) > 1:
                for i in range(1, len(gp_gap_records)):
                    t1, gap1 = gp_gap_records[i - 1]
                    t2, gap2 = gp_gap_records[i]
                    # 梯形积分法计算 Primal Integral
                    primal_integral += (gap1 + gap2) / 2 * (t2 - t1)
            gp_int_total += primal_integral
            print("gp_int_total：", gp_int_total)
        else:
            obj = gp_obj_list[(0 + e) * TestNum + ins_num]

        error_local, error_all, mse = pred_error(scores, test_ins_name, instanceName, BD[b_vars])
        acc += (1 - error_all / len(scores))
        acc_local += (1 - error_local / (k_0+k_1))
        mse_total += mse
        print(f"gurobi 最优解：{obj}; pred_error_all is: {error_all}; mse is: {mse}")
        m.reset()
        m.Params.TimeLimit = TimeLimit
        m.Params.Threads = Threads
        m.Params.MIPFocus = 1

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
        t_start = time()
        m.update()
        primal_integral_callback.gap_records = []
        primal_integral_callback.gap_threshold = gap_threshold
        primal_integral_callback.point = None

        if ps_solve:
            m.optimize(primal_integral_callback)
            pre_obj = m.objVal
        else:
            pre_obj = 0

        if m.status == GRB.TIME_LIMIT:
            primal_integral_callback.gap_records.append(
                (m.Runtime, abs((m.objVal - m.ObjBound) / abs(m.ObjBound))))
        ps_gap_records = primal_integral_callback.gap_records
        ps_gap_time += primal_integral_callback.point[0] if primal_integral_callback.point is not None else 0
        primal_integral = 0.0
        if len(ps_gap_records) > 1:
            for i in range(1, len(ps_gap_records)):
                t1, gap1 = ps_gap_records[i - 1]
                t2, gap2 = ps_gap_records[i]
                # 梯形积分法计算 Primal Integral
                primal_integral += (gap1 + gap2) / 2 * (t2 - t1)

        ps_int_total += primal_integral
        print(f"ps_int_total: {ps_int_total}")

        # new_solution = {var.VarName: var.X for var in m.getVars()}
        # validate_solution_in_original_model(o_m, new_solution)

        t = time() - t_start
        time_total += t
        if max_time <= t:
            max_time = t
        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            subop = (pre_obj - obj) / (obj + 1e-8) if TaskName != "CA" else (obj - pre_obj) / (obj + 1e-8)
            subop_total += subop
            print(
                f"ps 最优值：{pre_obj}; subopt: {round(subop, 4)}; subop_total: {round(subop_total / (ins_num + 1), 4)}; pred_error: {round(acc / (ins_num + 1), 4)}")
        m.dispose()
        gc.collect()

total_num = TestNum * epoch
results = {
    "avg_subopt": round(subop_total / total_num, 6),
    "mean_time_pred": round(time_total / total_num, 6),
    "max_time": round(max_time, 6),
    "gurobi_integral": round(gp_int_total / total_num, 6) if gp_solve else 0,
    "ps_gap_integral": round(ps_int_total / total_num, 6) if ps_solve else 0,
    "gurobi_gap_bound": round(gp_gap_time / total_num, 6) if gp_solve else 0,
    "ps_gap_bound": round(ps_gap_time / total_num, 6) if ps_solve else 0,
}
results_dir = f"/home/ljj/project/predict_and_search/results/{TaskName}/"
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
with open(results_dir + "results.json", "a") as file:
    json.dump(results, file)
    file.write("\n")
print("avg_subopt： ", results['avg_subopt'])
print("ps_gap_integral： ", results['ps_gap_integral'])
print(f"pred_error_all: {round(acc / total_num, 4)}")
print(f"pred_error_local: {round(acc_local / total_num, 4)}")
print(f"mse: {mse_total / total_num}")
