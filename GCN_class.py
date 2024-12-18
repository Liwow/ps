import json
import os
from sklearn.cluster import KMeans
from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
import torch_geometric
import pickle
import numpy as np
import math
import pyscipopt as scip
import utils

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        b = torch.cat([self.post_conv_module(output), right_features], dim=-1)
        a = self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        # node_features_i,the node to be aggregated
        # node_features_j,the neighbors of the node i

        # print("node_features_i:",node_features_i.shape)
        # print("node_features_j",node_features_j.shape)
        # print("edge_features:",edge_features.shape)

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output


class SELayerGraph(nn.Module):
    def __init__(self, feature_dim, reduction=16):
        super(SELayerGraph, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, num_nodes, feature_dim)
        mean = x.mean(dim=0, keepdim=True)  # 对节点的特征均值化 (batch_size, feature_dim)
        weights = self.fc(mean)  # 通过全连接网络生成权重
        return x * weights


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            v_class,
            c_class,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.v_class = v_class
        self.c_class = c_class

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        # elif key == "v_class":
        #     return [self.variable_features.size(0)] * len(self.v_class)
        # elif key == "c_class":
        #     return [self.constraint_features.size(0)] * len(self.c_class)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GNNPolicy_class(torch.nn.Module):
    def __init__(self, TaskName, position=False):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6 if not position else 18
        self.temperature = 0.6
        self.dropout = nn.Dropout(p=0.3)
        self.se_con = SELayerGraph(emb_size)
        self.se_var = SELayerGraph(emb_size)
        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.con_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.var_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.anchor_gnn = AnchorGNN(TaskName, emb_size).to(DEVICE)

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features, v_class, c_class, c_mask=None
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        constraint_features = self.se_con(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        variable_features = self.se_var(variable_features)

        constraint_features = self.dropout(constraint_features)
        variable_features = self.dropout(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )

        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        variable_features, constraint_features = self.anchor_gnn(
            variable_features, constraint_features, v_class, c_class
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )

        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        var_output = torch.sigmoid(self.var_mlp(variable_features).squeeze(-1) / self.temperature)

        return var_output


class GraphDataset_class(torch_geometric.data.Dataset):

    def __init__(self, sample_files, position=False):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.position = position

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        try:
            with open(solFilePath, 'rb') as f:
                solData = pickle.load(f)
        except Exception as e:
            print(f"Error: {e}, file: {solFilePath}")

        BG = bgData
        varNames = solData['var_names']

        sols = solData['sols'][:50]  # [0:300]
        objs = solData['objs'][:50]  # [0:300]
        slacks = solData['slacks'][:50]
        sols = np.round(sols, 0)
        return BG, sols, objs, varNames, slacks

    def get_critical(self, path, method="kmeans", n=15):
        file = os.path.basename(path[0])[:-3]
        task_name = path[0].split('/')[-3]
        instance_path = './instance/train/' + task_name + '/' + file
        model = scip.Model()
        model.setParam('display/verblevel', 0)  # 禁用显示信息
        model.setParam('display/freq', 0)
        model.readProblem(instance_path)
        critical_list = []
        num_vars_per_constr = []
        if method == "fix":
            for constr in model.getConss():
                rhs = model.getRhs(constr)
                lhs = model.getLhs(constr)
                if lhs != rhs:
                    lin_expr = model.getValsLinear(constr)
                    num_vars = len(lin_expr)
                    num_vars_per_constr.append(num_vars)
                    critical_list.append(1 if num_vars <= n else 0)
        elif method == "kmeans":
            for constr in model.getConss():
                rhs = model.getRhs(constr)
                lhs = model.getLhs(constr)
                if lhs != rhs:
                    lin_expr = model.getValsLinear(constr)
                    num_vars_per_constr.append(len(lin_expr))
            sparse_cluster, labels = utils.get_label_by_kmeans(num_vars_per_constr)
            critical_list = [1 if label == sparse_cluster else 0 for label in labels]
        else:
            print("no select way")
            return None

        return critical_list, num_vars_per_constr

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames, slacks = self.process_sample(self.sample_files[index])
        critical_list_by_sparse, var_num_list = self.get_critical(self.sample_files[index], method="kmeans")

        A, v_map, v_nodes, c_nodes, b_vars, v_class, c_class = BG
        # sparse_cluster, labels = utils.get_label_by_kmeans(coupling_degrees)
        # critical_list_by_coupling = [1 if label == sparse_cluster else 0 for label in labels]

        n_v_class = len(v_class)
        n_c_class = len(c_class)

        critical_list = critical_list_by_sparse

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        variable_features = getPE(variable_features, self.position)

        constraint_features[torch.isnan(constraint_features)] = 1

        v_class_list = utils.convert_class_to_labels(v_class, variable_features.shape[0])
        c_class_list = utils.convert_class_to_labels(c_class, constraint_features.shape[0])

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.LongTensor(v_class_list),
            torch.LongTensor(c_class_list),
        )

        c_mask = []
        c_labels = []
        for slack in slacks:
            mask = [1 if con[2] in ['<', '>'] else 0 for con in slack]
            labels = [1 if con[1] == 0 and con[2] in ['<', '>'] else 0 for con in slack]
            mask = torch.tensor(mask, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int)
            labels = labels[mask == 1]
            labels = torch.bitwise_and(labels, torch.tensor(critical_list, dtype=torch.int))
            c_mask.append(mask)
            c_labels.append(labels.float())
        graph.c_labels = torch.cat(c_labels).reshape(-1)
        graph.c_mask = torch.cat(c_mask).reshape(-1)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.ncons = len(slacks[1])
        graph.nlabels = c_labels[1].size()[0]
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames
        varname_dict = {}
        varname_map = []
        i = 0
        for iter in varNames:
            varname_dict[iter] = i
            i += 1
        for iter in v_map:
            varname_map.append(varname_dict[iter])

        varname_map = torch.tensor(varname_map)

        graph.varInds = [[varname_map], [b_vars]]

        return graph


def get_by_semantics(task, tokenizer, text_encoder):
    # get var_fea, con_fea, edge, edge_feature
    # edge为抽象模型边的索引，edge_feature为边的特征
    var_fea = []
    con_fea = []
    edge = []
    edge_feature = []
    config_path = './task_config.json'
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    if "task" not in config_json or task not in config_json["task"]:
        raise ValueError(f"Task '{task}' not found in the JSON configuration.")
    task_details = config_json["task"][task]
    task_description = task_details.get("task_description", "No description available")
    task_text = f"task: {task}\ntask_description: {task_description}\n\n"
    var_text = []
    con_text = []
    if "variable_type" in task_details:
        for var_name, var_details in task_details["variable_type"].items():
            var_description = var_details.get("description", "No description available")
            var_type = var_details.get("type", "No type specified")
            var_index = var_details.get("index", "No index specified")
            var_range = var_details.get("range", "No range specified")
            var_constraints = var_details.get("constraints", "No constraints specified")

            var_text.append(
                task_text + f"variable_name: {var_name}, variable_index: {var_index}, variable_type: {var_type}, variable_description: {var_description}, variable_range: {var_range}, variable_constraints: {var_constraints}\n\n")
    if "constraint_type" in task_details:
        for con_name, con_details in task_details["constraint_type"].items():
            con_description = con_details.get("description", "No description available")
            con_type = con_details.get("type", "No type specified")
            con_index = con_details.get("index", "No index specified")
            con_expression = con_details.get("expression", "No expression specified")
            con_constraints = con_details.get("constraints", "No constraints specified")

            con_text.append(
                task_text + f"constraint_name: {con_name}, constraint_index: {con_index}, constraint_type: {con_type}, constraint_description: {con_description}, constraint_expression: {con_expression}, constraint_constraints: {con_constraints}\n\n")

    for var in var_text:
        text_tokenizer = tokenizer(var, return_tensors="pt").to(DEVICE)
        text_embedding = text_encoder(**text_tokenizer).last_hidden_state[:, -1, :]
        var_fea.append(text_embedding)
    var_fea = torch.stack(var_fea)

    for con in con_text:
        text_tokenizer = tokenizer(con, return_tensors="pt").to(DEVICE)
        text_embedding = text_encoder(**text_tokenizer).last_hidden_state[:, -1, :]
        con_fea.append(text_embedding)
    con_fea = torch.stack(con_fea)

    return var_fea, con_fea, edge, edge_feature


class AnchorGNN(torch.nn.Module):
    def __init__(self, task, emb_size=64):
        super().__init__()
        self.emb_size = emb_size
        self.layer_norm = nn.LayerNorm(self.emb_size)
        path = "../../local_models/t5-base"
        tokenizer = T5Tokenizer.from_pretrained(path, legacy=False)
        text_encoder = T5EncoderModel.from_pretrained(path).to(DEVICE)
        var_fea, con_fea, edge, edge_feature = get_by_semantics(task, tokenizer, text_encoder)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(768, 2 * emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_size, emb_size),
        )
        self.v_sem_fea = var_fea
        # self.v_sem_fea = nn.parameter.Parameter(var_fea, requires_grad=True)
        self.v_n_class = var_fea.shape[0]
        self.c_sem_fea = con_fea
        # self.c_sem_fea = nn.parameter.Parameter(con_fea, requires_grad=True)
        self.c_n_class = con_fea.shape[0]
        self.edge = edge
        self.edge_fea = edge_feature

        self.self_att = torch.nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)

        self.send_var = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
        )
        self.send_con = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
        )
        self.rec_var = torch.nn.Sequential(
            torch.nn.Linear(self.v_n_class, emb_size),
        )
        self.rec_con = torch.nn.Sequential(
            torch.nn.Linear(self.c_n_class, emb_size),
        )

        self.anchor1 = Anchor1(self.v_n_class, self.c_n_class, emb_size)
        # self.anchor2 = Anchor2(self.v_n_class, self.c_n_class, emb_size)
        self.anchor3 = Anchor3(self.v_n_class, self.c_n_class, emb_size)

    def forward(self, v, c, v_class, c_class, mpnn=False):
        # v_class: [[indices],...,[indices]]   get batch
        v_class = v_class.to(v.device)
        c_class = c_class.to(c.device)
        v = self.layer_norm(v)
        c = self.layer_norm(c)
        fea = torch.concat([self.v_sem_fea, self.c_sem_fea], dim=0)
        fea = self.proj(fea)  # 768 -> 64
        fea_sem = self.self_att(fea, fea, fea)[0]
        fea_sem = self.layer_norm(fea_sem + fea).squeeze(1)
        v_sem = fea_sem[:self.v_n_class]
        c_sem = fea_sem[-self.c_n_class:]
        # send
        v_s = self.send_var(v)
        c_s = self.send_con(c)

        if not mpnn:
            # v_updates, c_updates = self.anchor3(v_s, c_s, v_sem, c_sem, v_class, c_class)
            # v_updates = self.layer_norm(v_updates)
            # v_s = self.rec_var(torch.cat([v_s, v_updates], dim=-1))
            # c_updates = self.layer_norm(c_updates)
            # c_s = self.rec_con(torch.cat([c_s, c_updates], dim=-1))

            # v_updates, c_updates = self.anchor2(v_s, c_s, v_sem, c_sem, v_class, c_class)

            v_final, c_final = self.anchor1(v_s, c_s, v_sem, c_sem, v_class, c_class)
            self.layer_norm(v_final)
            v_updates = v_s @ v_final.transpose(0, 1)
            v_updates = torch.sin(self.rec_var(v_updates))
            v_s = v_s * v_updates
            self.layer_norm(c_final)
            c_updates = c_s @ c_final.transpose(0, 1)
            c_updates = torch.sin(self.rec_con(c_updates))
            c_s = c_s * c_updates

        return v_s, c_s


class Anchor1(torch.nn.Module):
    def __init__(self, v_n, c_n, emb_size=64):
        super().__init__()
        self.emb_size = emb_size
        self.v_n = v_n
        self.c_n = c_n
        self.layer_norm = nn.LayerNorm(self.emb_size)
        self.cross_att_var = torch.nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)
        self.cross_att_con = torch.nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)

    def forward(self, v_s, c_s, v_sem, c_sem, v_class, c_class):
        v_final_list = []
        c_final_list = []

        for v_i in range(self.v_n):
            v_i_indices = torch.nonzero(v_class == v_i, as_tuple=False).squeeze(1)
            if len(v_i_indices) == 0:
                continue
            v_i_fea = v_s[v_i_indices].unsqueeze(1)
            v_i_sem = v_sem[v_i].unsqueeze(0).unsqueeze(1)
            v_i_final = self.cross_att_var(v_i_sem, v_i_fea, v_i_fea)[0].squeeze(1)
            v_final_list.append(v_i_final)
        v_final = torch.cat(v_final_list)

        for c_i in range(self.c_n):
            c_i_indices = torch.nonzero(c_class == c_i, as_tuple=False).squeeze(1)
            if len(c_i_indices) == 0:
                continue
            c_i_fea = c_s[c_i_indices].unsqueeze(1)
            c_i_sem = c_sem[c_i].unsqueeze(0).unsqueeze(1)
            c_i_final = self.cross_att_con(c_i_sem, c_i_fea, c_i_fea)[0].squeeze(1)
            c_final_list.append(c_i_final)
        c_final = torch.cat(c_final_list)

        return v_final, c_final


class Anchor3(torch.nn.Module):
    def __init__(self, v_n, c_n, emb_size=64):
        super().__init__()
        self.emb_size = emb_size
        self.v_n = v_n
        self.c_n = c_n
        self.layer_norm = nn.LayerNorm(self.emb_size)
        self.cross_att_var = torch.nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)
        self.cross_att_con = torch.nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)

    def forward(self, v_s, c_s, v_sem, c_sem, v_class, c_class):
        v_mask = (v_class.unsqueeze(1) == torch.arange(self.v_n_class, device=DEVICE).unsqueeze(0))
        v_mask = v_mask.float()
        v_class_count = v_mask.sum(dim=0, keepdim=True) + 1e-8
        v_i_fea = torch.matmul(v_mask.T, v_s) / v_class_count.T  # 平均池化 [n_classes, dim]
        v_i_fea = v_i_fea.unsqueeze(1)  # [n_classes, 1, dim]
        v_i_sem = v_sem.unsqueeze(1)  # [n_classes, 1, dim]
        v_i_final = self.cross_att_var(v_i_sem, v_i_fea, v_i_fea)[0].squeeze(1)  # [n_classes, dim]
        v_updates = torch.matmul(v_mask, v_i_final)  # [n_variables, dim]

        c_mask = (c_class.unsqueeze(1) == torch.arange(self.c_n_class, device=DEVICE).unsqueeze(0))
        c_mask = c_mask.float()
        c_class_count = c_mask.sum(dim=0, keepdim=True) + 1e-8
        c_i_fea = torch.matmul(c_mask.T, c_s) / c_class_count.T  # 平均池化 [n_classes, dim]
        c_i_fea = c_i_fea.unsqueeze(1)  # [n_classes, 1, dim]
        c_i_sem = c_sem.unsqueeze(1)  # [n_classes, 1, dim]
        c_i_final = self.cross_att_con(c_i_sem, c_i_fea, c_i_fea)[0].squeeze(1)  # [n_classes, dim]
        c_updates = torch.matmul(c_mask, c_i_final)  # [n_constraints, dim]

        return v_updates, c_updates


# class Anchor2(torch.nn.Module):
#     def __init__(self, v_n, c_n, emb_size=64):
#         super().__init__()
#         self.emb_size = emb_size
#         self.v_n = v_n
#         self.c_n = c_n
#         self.layer_norm = nn.LayerNorm(self.emb_size)
#         self.cross_att_var = torch.nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)
#         self.cross_att_con = torch.nn.MultiheadAttention(self.emb_size, num_heads=4, batch_first=False)
#
#     def forward(self, v_s, c_s, v_sem, c_sem, v_class, c_class):
#         v_updates = torch.zeros_like(v_s)
#         c_updates = torch.zeros_like(c_s)
#
#         for v_i in range(self.v_n):
#             v_i_indices = torch.nonzero(v_class == v_i, as_tuple=False).squeeze(1)
#             if len(v_i_indices) == 0:
#                 continue
#             v_i_fea = v_s[v_i_indices].unsqueeze(1)
#             v_i_sem = v_sem[v_i].unsqueeze(0).unsqueeze(1)
#             v_i_final = self.cross_att_var(v_i_sem, v_i_fea, v_i_fea)[0].squeeze(1)
#             v_updates[v_i_indices] += v_i_final
#
#         for c_i in range(self.c_n):
#             c_i_indices = torch.nonzero(c_class == c_i, as_tuple=False).squeeze(1)
#             if len(c_i_indices) == 0:
#                 continue
#             c_i_fea = c_s[c_i_indices].unsqueeze(1)
#             c_i_sem = c_sem[c_i].unsqueeze(0).unsqueeze(1)
#             c_i_final = self.cross_att_con(c_i_sem, c_i_fea, c_i_fea)[0].squeeze(1)
#             c_updates[c_i_indices] += c_i_final
#
#         return v_updates, c_updates


def getPE(var_fea, p=True, d_model=12):
    lens = var_fea.shape[0]
    if p:
        d_model = 12  # max length 4095
        position = torch.arange(0, lens, 1)
        pe = torch.zeros(lens, d_model)
        for i in range(len(pe)):
            binary = str(bin(position[i]).replace('0b', ''))
            for j in range(len(binary)):
                pe[i][j] = int(binary[-(j + 1)])
        # position = torch.arange(0, lens).unsqueeze(1)  # 位置索引 (lens, 1)
        # div_term = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000.0) / d_model)  # 频率分量 (d_model/2, )
        # pe = torch.zeros(lens, d_model)  # 初始化位置编码 (lens, d_model)
        # pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列: sin
        # pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列: cos

        var_fea = torch.concat([var_fea, pe], dim=1)
    # else:
    #     random_features = torch.randn(lens, 1)
    #     var_fea = torch.concat([var_fea, random_features], dim=1)
    return var_fea
