import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch_geometric
import gzip
import pickle
import numpy as np
from GCN import BipartiteNodeData, getPE, BipartiteGraphConvolution
import time
import pyscipopt as scip
import utils


class GraphDataset(torch_geometric.data.Dataset):
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

        variable_features = getPE(v_nodes, self.position)
        edge_features = A._values().unsqueeze(1)

        constraint_features[torch.isnan(constraint_features)] = 1

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features.float()),
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


class GNNPolicy_edl(torch.nn.Module):
    def __init__(self, TaskName=None, position=False, lambda1=1.0, k=2, act_type='softplus'):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 7 if not position else 18
        self.lambda1 = lambda1  # W/K
        self.dropout = 0.5
        self.K = k
        self.act_type = act_type

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
            torch.nn.Linear(emb_size, self.K, bias=False),
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

        output = self.output_module(variable_features)
        if self.act_type == "softplus":
            evi = torch.nn.Softplus()
            alpha = evi(output) + self.lambda1
        elif self.act_type == 'relu':
            alpha = torch.relu(output) + self.lambda1
        elif self.act_type == 'exp':
            alpha = torch.exp(output) + self.lambda1
        else:
            raise NotImplementedError

        return alpha

    def get_p(self, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)  # [varnum, 1]
        pre_sols = 1 - alpha[:, 0] / S.squeeze()
        uncertainty = self.K * self.lambda1 / S.squeeze()

        return pre_sols, uncertainty

    def edl_loss(self, alpha, sols, weights, loss_type='ce'):
        S = torch.sum(alpha, dim=1, keepdim=True)  # [varnum, 1]
        sols_one_hot = F.one_hot(sols.long(), num_classes=self.K).float()  # [50, varnum, 2]
        alpha_expanded = alpha.unsqueeze(0)  # [1, varnum, 2]
        S_expanded = S.unsqueeze(0)  # [1, varnum, 1]
        if loss_type == 'ce':
            likelihood = sols_one_hot * (torch.digamma(alpha_expanded) - torch.digamma(S_expanded))
            sample_loss = -torch.sum(likelihood, dim=2)  # [50, varnum]
            sample_loss = sample_loss * weights[:, None]  # [50, varnum]
            weighted_loss = sample_loss.sum()
        elif loss_type == 'mse':
            gap = sols_one_hot - alpha_expanded / S_expanded
            loss_mse = gap.pow(2).sum(-1)
            loss_var_ = (alpha_expanded * (S_expanded - alpha_expanded) / (
                        S_expanded * S_expanded * (S_expanded + 1))).sum(
                -1)
            sum_loss = loss_mse + loss_var_
            sample_loss = sum_loss * weights[:, None]
            weighted_loss = sample_loss.sum()
        else:
            raise ValueError('loss_type should be either ce or mse')
        kl_loss = compute_kl_loss(alpha)

        pre_sols = 1 - alpha[:, 0] / S.squeeze()
        uncertainty = self.K * self.lambda1 / S.squeeze()
        # total_loss = weighted_loss.sum()  # 总损失
        return weighted_loss, kl_loss, pre_sols, uncertainty

    def fisher_loss(self, alpha, sols, weights):
        loss_mse, loss_var, loss_det_fisher = compute_fisher_loss(sols, alpha)
        sum_loss = loss_mse + loss_var + 0.08 * loss_det_fisher
        sample_loss = sum_loss * weights[:, None]
        weighted_loss = sample_loss.sum()

        kl_loss = compute_kl_loss(alpha)

        S = torch.sum(alpha, dim=1, keepdim=True)
        pre_sols = 1 - alpha[:, 0] / S.squeeze()
        uncertainty = self.K * self.lambda1 / S.squeeze()

        return weighted_loss, kl_loss, pre_sols, uncertainty


def compute_kl_loss(alphas, labels=None, target_concentration=1.0, concentration=1.0, reverse=True):
    # TODO: Need to make sure this actually works right...
    # todo: so that concentration is either fixed, or on a per-example setup
    # Create array of target (desired) concentration parameters

    target_alphas = torch.ones_like(alphas) * concentration
    if labels is not None:
        target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)

    if reverse:
        loss = dirichlet_kl_divergence(alphas, target_alphas)
    else:
        loss = dirichlet_kl_divergence(target_alphas, alphas)

    return loss


def dirichlet_kl_divergence(alphas, target_alphas):
    epsilon = torch.tensor(1e-8)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                          torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = torch.squeeze(alp0_term + alphas_term).mean()

    return loss


def compute_fisher_loss(sols, evi_alp_):
    # batch_dim, n_samps, num_classes = evi_alp_.shape
    evi_alp_ = evi_alp_.unsqueeze(0)
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
    labels_1hot_ = F.one_hot(sols.long(), num_classes=2).float()  # [50, varnum, 2]
    gamma1_alp = torch.polygamma(1, evi_alp_)
    gamma1_alp0 = torch.polygamma(1, evi_alp0_)

    gap = labels_1hot_ - evi_alp_ / evi_alp0_
    loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1)
    loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(-1)

    loss_det_fisher_ = - (torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1)))
    loss_det_fisher_ = torch.where(torch.isfinite(loss_det_fisher_), loss_det_fisher_,
                                   torch.zeros_like(loss_det_fisher_))

    return loss_mse_, loss_var_, loss_det_fisher_
