import os.path
import pickle
import random
from multiprocessing import Process, Queue
import gurobipy as gp
import numpy as np
import argparse
from helper import get_a_new2, get_bigraph, get_pattern
from datetime import datetime
import gc


def solve_grb(filepath, log_dir, settings):
    gp.setParam('LogToConsole', 0)
    m = gp.read(filepath)

    m.Params.PoolSolutions = settings['maxsol']
    m.Params.PoolSearchMode = settings['mode']
    m.Params.TimeLimit = settings['maxtime']
    m.Params.Threads = settings['threads']
    log_path = os.path.join(log_dir, os.path.basename(filepath) + '.log')
    with open(log_path, 'w'):
        pass

    m.Params.LogFile = log_path
    m.setParam("MIPFocus", 3)
    m.optimize()

    sols = []
    objs = []
    slacks = []
    solc = m.getAttr('SolCount')

    mvars = m.getVars()
    # get variable name,
    oriVarNames = [var.varName for var in mvars]

    varInds = np.arange(0, len(oriVarNames))

    if m.Status not in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT]:
        print("*********No optimal found !*************")
        m.dispose()
        return None

    for sn in range(solc):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))
        objs.append(m.PoolObjVal)
        con_slacks = {constr: constr.slack for constr in m.getConstrs()}
        new_slacks = [(c[0].ConstrName, c[1], c[0].Sense) for c in con_slacks.items()]
        # sort
        # cons = m.getConstrs()
        # cons_map = [[x, m.getRow(x).size()] for x in cons]
        # cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
        # cons = [x[0].ConstrName for x in cons_map]
        # cons_dict = {c: i for i, c in enumerate(cons)}
        # new_slacks = sorted(new_slacks, key=lambda x: cons_dict[x[0]])

        new_slacks.append(("obj_node", 1e20, None))  # obj节点
        slacks.append(new_slacks)

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)

    sol_data = {
        'var_names': oriVarNames,
        'sols': sols,
        'objs': objs,
        'slacks': slacks
    }
    m.dispose()
    gc.collect()

    return sol_data


def collect(ins_dir, q, sol_dir, log_dir, bg_dir, settings):
    while True:
        filename = q.get()
        if not filename:
            break
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"开始处理: {filename} 时间: {current_time}")

        filepath = os.path.join(ins_dir, filename)

        # get bipartite graph , binary variables' indices
        bg_file = os.path.join(bg_dir, filename + '.bg')
        if not os.path.isfile(bg_file):
            A2, v_map2, v_nodes2, c_nodes2, b_vars2, v_class, c_class, _ = get_bigraph(filepath, v_class_name,
                                                                                       c_class_name)
            BG_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2, v_class, c_class]
            with open(bg_file, 'wb') as bg_f:
                pickle.dump(BG_data, bg_f)
            del A2, v_map2, v_nodes2, c_nodes2, b_vars2, v_class, c_class, BG_data
            gc.collect()

        sol_file = os.path.join(sol_dir, filename + '.sol')
        if not os.path.isfile(sol_file):
            sol_data = solve_grb(filepath, log_dir, settings)
            if sol_data is not None:
                with open(sol_file, 'wb') as sol_f:
                    pickle.dump(sol_data, sol_f)
            del sol_data
            gc.collect()

    gc.collect()


if __name__ == '__main__':
    sizes = ["CA"]
    # sizes=["IP","WA","IS","CA","NNV"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./')
    parser.add_argument('--nWorkers', type=int, default=16)
    parser.add_argument('--maxTime', type=int, default=2400)
    parser.add_argument('--maxStoredSol', type=int, default=100)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()

    for size in sizes:

        dataDir = args.dataDir

        INS_DIR = os.path.join(dataDir, f'instance/train/{size}')

        if not os.path.isdir(f'./dataset/{size}'):
            os.mkdir(f'./dataset/{size}')
        if not os.path.isdir(f'./dataset/{size}/solution'):
            os.mkdir(f'./dataset/{size}/solution')
        if not os.path.isdir(f'./dataset/{size}/NBP'):
            os.mkdir(f'./dataset/{size}/NBP')
        if not os.path.isdir(f'./dataset/{size}/logs'):
            os.mkdir(f'./dataset/{size}/logs')
        if not os.path.isdir(f'./dataset/{size}/BG'):
            os.mkdir(f'./dataset/{size}/BG')

        SOL_DIR = f'./dataset/{size}/solution'
        LOG_DIR = f'./dataset/{size}/logs'
        BG_DIR = f'./dataset/{size}/BG'
        os.makedirs(SOL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        os.makedirs(BG_DIR, exist_ok=True)

        N_WORKERS = args.nWorkers

        # gurobi settings
        SETTINGS = {
            'maxtime': args.maxTime,
            'mode': 2,
            'maxsol': args.maxStoredSol,
            'threads': args.threads,

        }
        v_class_name, c_class_name = get_pattern("./task_config.json", size)
        filenames = os.listdir(INS_DIR)
        num = min(300, len(filenames))
        random.seed(42)
        random.shuffle(filenames)
        filenames = filenames[:num]

        q = Queue()
        # add ins
        for filename in filenames:
            BGFilepath = os.path.join(BG_DIR, filename + '.bg')
            solFilePath = os.path.join(SOL_DIR, filename + '.sol')
            if not os.path.exists(BGFilepath) or not os.path.exists(solFilePath):
                q.put(filename)
            elif not os.path.getsize(BGFilepath) > 0:
                print(BGFilepath, "is empty")
                os.remove(BGFilepath)
                q.put(filename)
            elif not os.path.getsize(solFilePath) > 0:
                print(solFilePath, "is empty")
                os.remove(solFilePath)
                q.put(filename)
        # add stop signal
        for i in range(N_WORKERS):
            q.put(None)

        ps = []
        for i in range(N_WORKERS):
            p = Process(target=collect, args=(INS_DIR, q, SOL_DIR, LOG_DIR, BG_DIR, SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()


        for p in ps:
            p.close()  # 关闭进程资源
        gc.collect()

        print('done')
