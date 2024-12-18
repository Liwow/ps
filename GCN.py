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


class GNNPolicy(torch.nn.Module):
    def __init__(self, TaskName=None, position=False):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6

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

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features
            # , multimodal_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output


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


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, position=False):
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

        sols = np.round(sols, 0)
        return BG, sols, objs, varNames

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars, _, _ = BG

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

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)
        graph.objVals = torch.FloatTensor(objs)
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


class GNNPolicy_position(torch.nn.Module):
    def __init__(self, TaskName=None, position=False):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 18

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

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output


# GNN merged a text by cross_attention
class GNNPolicy_multimodal(torch.nn.Module):
    def __init__(self, TaskName=None):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6
        path = "../../local_models/t5-base"
        self.text = utils.text_setting(TaskName)
        self.cross_attention = torch.nn.MultiheadAttention(emb_size, num_heads=4, batch_first=True)
        self.tokenizer = T5Tokenizer.from_pretrained(path, legacy=False)
        self.text_encoder = T5EncoderModel.from_pretrained(path).to("cuda")
        self.semantic_proj = torch.nn.Linear(768, 64)

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

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # text-embedding
        text_tokenizer = self.tokenizer(self.text, return_tensors="pt").to("cuda")
        text_embedding = self.text_encoder(**text_tokenizer).last_hidden_state
        text_embedding = self.semantic_proj(text_embedding)
        text_embedding = text_embedding.view(-1, text_embedding.size(2))

        # merge
        variable_features = self.cross_attention(variable_features, text_embedding, text_embedding)[0]

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output


# GNN with random features or text features in v and c
class GNNPolicy_features(torch.nn.Module):
    def __init__(self, TaskName=None):
        super().__init__()
        self.text_size = 64
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6
        self.is_random = True
        path = "../../local_models/t5-base"
        # todo text数据集获取方法
        self.text = utils.text_setting(TaskName)
        self.cross_attention = torch.nn.MultiheadAttention(emb_size, num_heads=4, batch_first=True)
        self.tokenizer = T5Tokenizer.from_pretrained(path, legacy=False)
        self.text_encoder = T5EncoderModel.from_pretrained(path).to("cuda")
        self.semantic_proj = torch.nn.Linear(768, 64)

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

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features
    ):
        num_con = constraint_features.shape[0]
        num_var = variable_features.shape[0]
        if self.is_random:
            # todo 需要解决增加的维度远大于原特征维度的问题
            random_features_con = torch.randn(num_con, self.text_size)
            random_features_var = torch.randn(num_var, self.text_size)
            constraint_features = torch.cat((constraint_features, random_features_con), dim=1)
            variable_features = torch.cat((variable_features, random_features_var), dim=1)
        else:
            # todo 批次的增加text features
            text_features_con = torch.randn(num_con, self.text_size)
            text_features_var = torch.randn(num_var, self.text_size)
            constraint_features = torch.cat((constraint_features, text_features_con), dim=1)
            variable_features = torch.cat((variable_features, text_features_var), dim=1)
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output


# GNN predict constraints and variable
class GNNPolicy_constraint(torch.nn.Module):
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

        self.conv_c_to_v3 = BipartiteGraphConvolution()

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

        variable_features = self.conv_c_to_v3(
            constraint_features, edge_indices, edge_features, variable_features
        )
        if c_mask is not None:
            mask_constraint_features = constraint_features[c_mask == 1]
        else:
            mask_constraint_features = constraint_features

        con_output = torch.sigmoid(self.con_mlp(mask_constraint_features).squeeze(-1) / self.temperature)
        var_output = torch.sigmoid(self.var_mlp(variable_features).squeeze(-1) / self.temperature)

        return con_output, var_output


class GNNPolicy_loss(torch.nn.Module):
    def __init__(self, TaskName=None):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6
        self.temperature = 0.6
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

        self.conv_c_to_v3 = BipartiteGraphConvolution()

        self.encoderlayer = nn.TransformerEncoderLayer(d_model=64, nhead=4,
                                                       dim_feedforward=128, dropout=0.1,
                                                       activation='gelu', batch_first=True)

        self.con_mlp = torch.nn.Sequential(
            nn.TransformerEncoder(self.encoderlayer, num_layers=2),
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
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )

        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )

        variable_features = self.conv_c_to_v3(
            constraint_features, edge_indices, edge_features, variable_features
        )
        if c_mask is not None:
            mask_constraint_features = constraint_features[c_mask == 1]
        else:
            mask_constraint_features = constraint_features

        con_output = torch.sigmoid(self.con_mlp(mask_constraint_features).squeeze(-1) / self.temperature)
        var_output = torch.sigmoid(self.var_mlp(variable_features).squeeze(-1) / self.temperature)

        return con_output, var_output


class GraphDataset_position(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, position=False):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData
        varNames = solData['var_names']

        sols = solData['sols'][:50]  # [0:300]
        objs = solData['objs'][:50]  # [0:300]

        sols = np.round(sols, 0)
        return BG, sols, objs, varNames

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars, _, _ = BG

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        lens = variable_features.shape[0]
        feature_widh = 12  # max length 4095
        position = torch.arange(0, lens, 1)

        position_feature = torch.zeros(lens, feature_widh)
        for i in range(len(position_feature)):
            binary = str(bin(position[i]).replace('0b', ''))

            for j in range(len(binary)):
                position_feature[i][j] = int(binary[-(j + 1)])

        v = torch.concat([variable_features, position_feature], dim=1)

        variable_features = v

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
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


class GraphDataset_constraint(torch_geometric.data.Dataset):

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


class GraphDataset_loss(torch_geometric.data.Dataset):

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

    def get_critical(self, path, method="fix", n=15):
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
        critical_list_by_sparse, var_num_list = self.get_critical(self.sample_files[index], method="fix")

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
        t_labels = []
        for slack in slacks:
            mask = [1 if con[2] in ['<', '>'] else 0 for con in slack]
            labels = [1 if con[1] == 0 and con[2] in ['<', '>'] else 0 for con in slack]
            mask = torch.tensor(mask, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int)
            cc_labels = torch.bitwise_and(labels[mask == 1], torch.tensor(critical_list, dtype=torch.int))
            c_mask.append(mask)
            t_labels.append(labels[mask == 1].float())
            c_labels.append(cc_labels.float())
        graph.c_labels = torch.cat(c_labels).reshape(-1)
        graph.t_labels = torch.cat(t_labels).reshape(-1)
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


def postion_get(variable_features):
    lens = variable_features.shape[0]
    feature_widh = 12  # max length 4095
    position = torch.arange(0, lens, 1)

    position_feature = torch.zeros(lens, feature_widh)
    for i in range(len(position_feature)):
        binary = str(bin(position[i]).replace('0b', ''))

        for j in range(len(binary)):
            position_feature[i][j] = int(binary[-(j + 1)])

    variable_features = torch.FloatTensor(variable_features.cpu())
    v = torch.concat([variable_features, position_feature], dim=1).to(DEVICE)
    return v


def Loss_CV(mip, sol_per, con_per):
    lamda = 1
    sol = sol_per
    A, b, _ = mip.getCoff()

    result = torch.sparse.mm(A, sol.unsqueeze(1))
    Ax_minus_b = result.squeeze(1) - b

    tight_constraints_loss = torch.mul(con_per[con_per > 0.8], torch.pow(Ax_minus_b, 2))
    tight_loss = tight_constraints_loss.sum()

    loss = lamda * tight_loss

    return loss
