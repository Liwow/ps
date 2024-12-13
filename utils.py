import random
import re
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.nn import DataParallel


def split_sample_by_blocks(sample_files, train_rate, block_size):
    sample_files = sorted(sample_files, key=lambda x: int(re.search(r'\d+', str(x)).group()))
    sample_files = sample_files[:]
    random.seed(0)
    train_files = []
    valid_files = []

    num_blocks = (len(sample_files) + block_size - 1) // block_size

    for i in range(num_blocks):
        # Get the current block of files
        start_idx = i * block_size
        # Ensure end_idx doesn't exceed length of sample_files
        end_idx = min((i + 1) * block_size, len(sample_files))

        block_files = sample_files[start_idx:end_idx]

        random.shuffle(block_files)
        split_index = int(train_rate * len(block_files))
        train_files.extend(block_files[:split_index])
        valid_files.extend(block_files[split_index:])

    return train_files, valid_files


def text_setting(task):
    text = r"This is a description of an instance of a mixed-integer programming problem: "
    if task == "CA" or task == "CA_m" or task == "CA_multi":
        text += r"This is a combinatorial auction problem which can be formulated as an integer linear programming problem. It involves 300 items and 1500 bidding groups. " \
                "The main objective is to maximize the total revenue from bidders while ensuring that each item is assigned to at most one bidder. " \
                "The variable is defined as follows：x_j = 1 \text{ indicates that bidder} j \text{ 's combination S_j is selected}，x_j = 0 \text{ means it is not selected.}；" \
                "The objective function is: \text{Maximize} \quad \sum_{j} v_j x_j, where v_j represents the bid value of bidder for the item combination S_j. The goal of the objective function is to maximize the total revenue from selected bids, aiming for the highest auction revenue. " \
                "The constraint is:\sum_{j: i \in S_j} x_j \leq 1 \quad \forall i \in \text{Items}。which ensures that each item is assigned to at most one bidder, preventing duplicate allocation of any item."
    return text


def get_label_by_kmeans(list):
    X = np.array(list).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()
    sparse_cluster = np.argmin(cluster_centers)  # 稀疏类的索引
    return sparse_cluster, labels


def get_pair(critical_list, labels, critical_num):
    # critical_list: 0/1 是否重要 （全都不包括等式约束）
    # labels： 0/1 是否紧
    # critical_num： 重要值
    if len(labels) != len(critical_list):
        return
    scores = []
    critical_num = normalize_to_range(critical_num)
    for i in range(len(labels)):
        score = 0
        if labels[i]:
            score += 5
        elif critical_list[i]:
            score += critical_num[i]
        scores.append(score)

    return scores


def focal_loss(pre_cons, labels, weight, alpha=0.75, gamma=0.8):
    pos_loss = - 2 * alpha * ((1 - pre_cons + 1e-8) ** gamma) * (pre_cons + 1e-8).log() * (labels == 1).float()
    neg_loss = - 2 * (1 - alpha) * (pre_cons ** gamma) * (1 - pre_cons + 1e-8).log() * (labels == 0).float()

    masked_con_loss = (pos_loss + neg_loss) * weight[:, None]

    return masked_con_loss


def normalize_to_range(data, new_min=0, new_max=2):
    if not data:
        raise ValueError("The input list is empty.")

    old_min = min(data)
    old_max = max(data)

    if old_min == old_max:
        # 如果所有元素都相同，则将它们都映射为 new_min（避免除以零）
        return [new_min] * len(data)

    # 归一化处理
    normalized_data = [
        new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)
        for x in data
    ]
    return normalized_data


def convert_class_to_labels(class_, n):
    labels = [-1] * n  # 初始化所有约束类别为 -1（无类别）
    for class_idx, indices in enumerate(class_):
        for idx in indices:
            labels[idx] = class_idx  # 设置对应索引的类别
    return labels
