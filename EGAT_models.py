import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from EGAT_layers import SpGraphAttentionLayer


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        '''
        Function Description:
        Initializes the model by defining the size of the feature space, and sets up layers for encoding decision variables, edge features, and constraint features. 
        It includes two semi-convolutional attention layers and a final output layer.
        - nfeat: Initial feature dimension.
        - nhid: Dimension of the hidden layers.
        - nclass: Number of classes; for 0-1 integer programming, this would be 2.
        - dropout: Dropout rate.
        - alpha: Coefficient for leakyReLU.
        - nheads: Number of heads in the multi-head attention mechanism.
        Hint: Use the pre-written SpGraphAttentionLayer for the attention layers.
        '''
        super(SpGAT, self).__init__()
        self.dropout = dropout
        embed_size = 64
        self.input_module = torch.nn.Sequential(
            torch.nn.Linear(nfeat, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
        )
        self.attentions_u_to_v = [SpGraphAttentionLayer(embed_size,
                                                        nhid,
                                                        dropout=dropout,
                                                        alpha=alpha,
                                                        concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_u_to_v):
            self.add_module('attention_u_to_v_{}'.format(i), attention)
        self.attentions_v_to_u = [SpGraphAttentionLayer(embed_size,
                                                        nhid,
                                                        dropout=dropout,
                                                        alpha=alpha,
                                                        concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_v_to_u):
            self.add_module('attention_v_to_u_{}'.format(i), attention)

        self.out_att_u_to_v = SpGraphAttentionLayer(nhid * nheads,
                                                    embed_size,
                                                    dropout=dropout,
                                                    alpha=alpha,
                                                    concat=False)
        self.out_att_v_to_u = SpGraphAttentionLayer(nhid * nheads,
                                                    embed_size,
                                                    dropout=dropout,
                                                    alpha=alpha,
                                                    concat=False)
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(embed_size, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, 1, bias=False),
            # torch.nn.Linear(embed_size, nclass, bias=False),
            #torch.nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, edgeA, edgeB, edge_feat):
        '''
        Function Description:
        Executes the forward pass using the provided constraint, edge, and variable features, processing them through an EGAT to produce the output.

        Parameters:
        - x: Features of the variable and constraint nodes.
        - edgeA, edgeB: Information about the edges.
        - edge_feat: Features associated with the edges.

        Return: The result after the forward propagation.
        '''
        #print(x)
        x = self.input_module(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        #print(x)
        new_edge = torch.cat([att(x, edgeA, edge_feat)[1] for att in self.attentions_u_to_v], dim=1)
        x = torch.cat([att(x, edgeA, edge_feat)[0] for att in self.attentions_u_to_v], dim=1)
        x = self.out_att_u_to_v(x, edgeA, edge_feat)
        new_edge = torch.mean(new_edge, dim=1).reshape(new_edge.size()[0], 1)
        #x = self.softmax(x)
        new_edge_ = torch.cat([att(x, edgeB, new_edge)[1] for att in self.attentions_v_to_u], dim=1)
        x = torch.cat([att(x, edgeB, new_edge)[0] for att in self.attentions_v_to_u], dim=1)
        x = self.out_att_v_to_u(x, edgeB, new_edge)
        new_edge_ = torch.mean(new_edge_, dim=1).reshape(new_edge_.size()[0], 1)

        x = self.output_module(x).sigmoid()


        return x, new_edge_


class EgatDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

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
        variable_features = getPE(variable_features, self.position)
        constraint_features = getPE(constraint_features, self.position)
        edge_features = A._values().unsqueeze(1)

        constraint_features[torch.isnan(constraint_features)] = 1

        edgeA = []
        edgeB = []
        edge_num = len(edge_indices[0])
        n = variable_features.shape[0]
        for i in range(edge_num):
            edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
            edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])
        edgeA = torch.as_tensor(edgeA)
        edgeB = torch.as_tensor(edgeB)

        m = constraint_features.shape[0]
        var_size = variable_features.shape[1]
        con_size = constraint_features.shape[1]

        constraint_features = constraint_features.tolist()
        for i in range(m):
            for j in range(var_size - con_size):
                constraint_features[i].append(0)
        constraint_features = torch.as_tensor(constraint_features).float()
        features = torch.cat([variable_features, constraint_features], dim=0)

        graph = BipartiteNodeData(
            torch.FloatTensor(variable_features),
            torch.FloatTensor(constraint_features),
            torch.FloatTensor(features),
            torch.LongTensor(edgeA),
            torch.LongTensor(edgeB),
            torch.FloatTensor(edge_features),
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
            variable_features,
            constraint_features,
            features,
            edge_A,
            edge_B,
            edge_features,

    ):
        super().__init__()
        self.variable_features = variable_features
        self.constraint_features = constraint_features
        self.features = features
        self.edge_A = edge_A
        self.edge_attr = edge_features
        self.edge_B = edge_B

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_A":
            n = self.constraint_features.size(0) + self.variable_features.size(0)
            return torch.tensor(
                [[n, n]]
            )
        elif key == "edge_B":
            n = self.constraint_features.size(0) + self.variable_features.size(0)
            return torch.tensor(
                [[n, n]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


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
    else:
        random_features = torch.randn(lens, 1)
        var_fea = torch.concat([var_fea, random_features], dim=1)
    return var_fea
