import os

from sklearn.cluster import KMeans
from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
import torch_geometric
import gzip
import pickle
import numpy as np
import time
import pyscipopt as scip
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        return x * weights  # 加权特征


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

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

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
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GNNPolicy_class(torch.nn.Module):
    def __init__(self, TaskName=None):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6
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

        # self.cross_attention = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

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

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features, c_mask=None
            # , multimodal_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        constraint_features = self.se_con(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        variable_features = self.se_var(variable_features)

        # Two half convolutions
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )

        constraint_features = self.dropout(constraint_features)
        variable_features = self.dropout(variable_features)

        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )

        if c_mask is not None:
            mask_constraint_features = constraint_features[c_mask == 1]
        else:
            mask_constraint_features = constraint_features

        con_output = torch.sigmoid(self.con_mlp(mask_constraint_features).squeeze(-1) / self.temperature)
        var_output = torch.sigmoid(self.var_mlp(variable_features).squeeze(-1) / self.temperature)

        return con_output, var_output


class GraphDataset_class(torch_geometric.data.Dataset):

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

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

        A, v_map, v_nodes, c_nodes, b_vars = BG
        # sparse_cluster, labels = utils.get_label_by_kmeans(coupling_degrees)
        # critical_list_by_coupling = [1 if label == sparse_cluster else 0 for label in labels]

        critical_list = critical_list_by_sparse

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        constraint_features[torch.isnan(constraint_features)] = 1

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
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


class AnchorGNN(torch.nn.Module):
    def __init__(self, task):
        super().__init__()
        emb_size = 64
        var_fea, con_fea, edge, edge_feature = self.get_by_semantics(task)
        self.v_sem_fea = var_fea
        self.v_n_class = var_fea[0]

        self.c_sem_fea = con_fea
        self.c_n_class = con_fea[0]

        self.edge = edge
        self.edge_fea = edge_feature
        self.self_att = torch.nn.MultiheadAttention(emb_size, num_heads=4, batch_first=True)
        self.cross_att = torch.nn.MultiheadAttention(emb_size, num_heads=4, batch_first=True)

    def forward(self, v, c, v_class, c_class, mpnn=False):
        # v_class: [[indices],...,[indices]]   get batch
        # c_class:
        # v_fea 先 self-att( /MPNN + 残差) 后与v cross att: v_fea q, v k v
        # get final v_class_f, c_class_f
        # v_class_f^T W v ?  融合 or concat
        if not mpnn:
            fea = torch.concat([self.v_sem_fea, self.c_sem_fea], dim=0)
            fea_sem = self.self_att(fea, fea, fea)[0]
            v_sem = fea_sem[:self.v_n_class]
            c_sem = fea_sem[-self.c_n_class:]
            for v_i in self.v_n_class:
                v_i_fea = v[v_class[v_i]]
                v_i_sem = v_sem[v_i].unsqueeze(0)
                v_i_final = self.cross_att(v_i_sem, v_i_fea, v_i_fea)
                # v_final =

        else:
            pass