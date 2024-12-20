import torch_geometric
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from openTSNE import TSNE
import utils
import helper
from umap import UMAP
import gc
import random
import os
import numpy as np
import torch
from helper import get_a_new2, get_bigraph, get_pattern
from GCN_class import getPE
from gp_tools import primal_integral_callback, get_gp_best_objective

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
multimodal = False
TaskName = "CA"
model_name = f'{TaskName}.pth'
if TaskName == "CA_multi":
    multimodal = True
# load pretrained model
if TaskName == "IP":
    # Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy, postion_get
    from GCN import GraphDataset_position as GraphDataset
    from GCN_class import GraphDataset_class
elif multimodal:
    from GCN import GNNPolicy_multimodal as GNNPolicy
else:
    from GCN import GNNPolicy as GNNPolicy
    from GCN import GraphDataset
    from GCN_class import GraphDataset_class
    # from GCN_class import GNNPolicy_class as GNNPolicy

TaskName = 'CA'
position = False
gp_solve = False
pathstr = f'./models/{model_name}'
policy = GNNPolicy(TaskName, position=position).to(DEVICE)
state = torch.load(pathstr, map_location=DEVICE)
policy.load_state_dict(state)
policy.eval()

# load instance
# sample_names = sorted(os.listdir(f'./instance/train/{TaskName}'))[2:]
# test_ins_name = sample_names[0]
# ins_name_to_read = f'./instance/train/{TaskName}/{test_ins_name}'
# v_class_name, c_class_name = get_pattern("./task_config.json", TaskName)
# _, _, _, _, _, v_class, c_class, _ = get_bigraph(ins_name_to_read, v_class_name, c_class_name)
# A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_name_to_read)
# constraint_features = c_nodes.cpu()
# constraint_features[torch.isnan(constraint_features)] = 1  # remove nan value
# variable_features = getPE(v_nodes, postion)
# if TaskName == "IP":
#     variable_features = postion_get(variable_features)
# edge_indices = A._indices()
# edge_features = A._values().unsqueeze(1).float()
# v_class = utils.convert_class_to_labels(v_class, variable_features.shape[0])
# c_class = utils.convert_class_to_labels(c_class, constraint_features.shape[0])

DIR_BG = f'./dataset/{TaskName}/BG'
DIR_SOL = f'./dataset/{TaskName}/solution'
sample_names = os.listdir(DIR_BG)
sample_files = [(os.path.join(DIR_BG, name), os.path.join(DIR_SOL, name).replace('bg', 'sol')) for name in sample_names]
train_files, valid_files = utils.split_sample_by_blocks(sample_files, 0.9, block_size=100)
valid_data = GraphDataset(valid_files, position=position)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=1, shuffle=False)

batch = next(iter(valid_loader))

valid_data_class = GraphDataset_class(valid_files, position=position)
batch_class = next(iter(valid_data_class))
v_class = batch_class.v_class
c_class = batch_class.c_class

# prediction
_, v_logits, c_logits = policy(
    batch.constraint_features.to(DEVICE),
    batch.edge_index.to(DEVICE),
    batch.edge_attr.to(DEVICE),
    batch.variable_features.to(DEVICE),
)
v_logits = v_logits.detach().cpu().numpy()
c_logits = c_logits.detach().cpu().numpy()


def reduce_dimensions(data, method='tsne', n_components=2):
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, n_jobs=-1, random_state=42)
        return reducer.fit(data)
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'umap':
        reducer = UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError("Unsupported reduction method")
    return reducer.fit_transform(data)


# 降维
v_embeddings = reduce_dimensions(v_logits, method='tsne')

c_embeddings = reduce_dimensions(c_logits, method='tsne')


def plot_embeddings(embeddings, labels, title, ax):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        ax.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f"Class {label}", alpha=0.7)
    ax.set_title(title)
    ax.legend()


def plot_embeddings_combined(v_embeddings, c_embeddings, v_class, c_class, title):
    embeddings = np.vstack([v_embeddings, c_embeddings])
    labels = np.hstack([v_class, c_class])  # 假设 v_class 和 c_class 是对应的类别标签
    types = np.array(['v'] * len(v_class) + ['c'] * len(c_class))  # 区分 v 和 c 类型

    unique_labels = np.unique(labels)
    unique_types = np.unique(types)
    fig, ax = plt.subplots(figsize=(10, 8))

    for label in unique_labels:
        for node_type in unique_types:
            idx = (labels == label) & (types == node_type)
            marker = 'o' if node_type == 'v' else '*'  # v 节点用圆圈，c 节点用方块
            ax.scatter(
                embeddings[idx, 0], embeddings[idx, 1],
                label=f"Type {node_type}, Class {label}",
                alpha=0.7, marker=marker
            )

    ax.set_title(title)
    ax.legend()
    plt.show()


# 绘图
figs, axs = plt.subplots(1, 2, figsize=(14, 6))
plot_embeddings(v_embeddings, np.array(v_class), "V Nodes", axs[0])
plot_embeddings(c_embeddings, np.array(c_class), "C Nodes", axs[1])
plt.tight_layout()
plt.show()

plot_embeddings_combined(v_embeddings, c_embeddings, v_class, c_class, "Combined V and C Nodes")
