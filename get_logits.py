
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from openTSNE import TSNE
from umap import UMAP
import numpy as np


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


def plot_embeddings(embeddings, labels, title, ax):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 1:
            continue
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


def plot_logits(v_logits, c_logits, v_class, c_class):
    v_logits = v_logits.detach().cpu().numpy()
    c_logits = c_logits.detach().cpu().numpy()
    v_embeddings = reduce_dimensions(v_logits, method='tsne')
    c_embeddings = reduce_dimensions(c_logits, method='tsne')
    # 绘图
    figs, axs = plt.subplots(1, 2, figsize=(14, 6))
    plot_embeddings(v_embeddings, np.array(v_class), "V Nodes", axs[0])
    plot_embeddings(c_embeddings, np.array(c_class), "C Nodes", axs[1])
    plt.tight_layout()
    plt.show()

    plot_embeddings_combined(v_embeddings, c_embeddings, v_class, c_class, "Combined V and C Nodes")


if __name__ == "__main__":
    pass



