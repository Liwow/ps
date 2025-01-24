import random
import re
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.nn import DataParallel
import math


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


def compare(pre_sol, sols, task, is_local=False, u=None):
    # m 1 n 0
    n, m, delta = test_hyperparam(task)
    u0 = 0.5
    sorted_indices = torch.argsort(pre_sol, descending=True)  # 降序排序

    pre_sol_rounded = torch.round(pre_sol)

    if is_local:
        top_m_indices = sorted_indices[:m] if m > 0 else []
        bottom_n_indices = sorted_indices[-n:] if n > 0 else []
    else:
        l = len(sorted_indices)
        m = int(l * 0.5)
        n = l - m
        top_m_indices = sorted_indices[:m] if m > 0 else []
        bottom_n_indices = sorted_indices[-n:] if n > 0 else []
    if u is not None:
        top_m_indices = [idx for idx in top_m_indices if u[idx] < u0]
        bottom_n_indices = [idx for idx in bottom_n_indices if u[idx] < u0]
        # if len(top_m_indices) < m or len(bottom_n_indices) < n:
        #     print("u works")
    best_radio = 0
    best_m_ratio = 0
    best_n_ratio = 0
    for i in range(20):
        top_m_correct = (pre_sol_rounded[top_m_indices] == sols[i][top_m_indices]).sum().item()
        bottom_n_correct = (pre_sol_rounded[bottom_n_indices] == sols[i][bottom_n_indices]).sum().item()
        top_m_ratio = top_m_correct / m if m > 0 else 0
        bottom_n_ratio = bottom_n_correct / n if n > 0 else 0
        radio = (top_m_correct + bottom_n_correct) / (m + n)
        if radio > best_radio:
            best_radio = radio
            best_m_ratio = top_m_ratio
            best_n_ratio = bottom_n_ratio
            acc = int(m + n - top_m_correct - bottom_n_correct <= delta)
    # print(f"m 1 ratio: {top_m_ratio}, n 0 ratio: {bottom_n_ratio}, Total best ratio: {best radio}")

    return best_radio


def test_hyperparam(task):
    '''
    set the hyperparams
    k_0, k_1, delta
    '''
    if task == "IP":
        return 400, 5, 10
    elif task == "IS":
        return 300, 300, 20
    elif task == "WA":  # 0, 500, 10
        return 0, 500, 10
    elif task == "CA":  # 600 0 1
        return 600, 0, 1
    elif task == "CA_big":  # 1000 0 1
        return 1200, 0, 2
    elif task == "beasley":
        return 50, 17, 10
    elif task == "ns":
        return 120, 18, 20
    elif task == "binkar":
        return 54, 24, 10
    elif task == "neos":
        return 20129, 569, 700  # 20741 609
    elif task == "mas":
        return 136, 14, 10
    elif task == "markshare":
        return 14, 12, 9
    elif task == "case118":  # 10000,1164,18878
        return 2000, 0, 100
    elif task == "case300":  # 13137,1767,22470
        return 2000, 0, 100
    elif task == "case1951rte":  # 80000, 2000, 15w
        return 20000, 50, 300
    elif task == "case6515rte":
        return 50000, 1200, 1200
    elif task == "case2868rte":
        return 30000, 50, 500
    elif task == "case2869pegase":
        return 22000, 50, 600
