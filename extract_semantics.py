import gurobipy as gp
import re
from collections import defaultdict


def extract(file_path):
    # 读取文件

    current_item = 0
    with open(file_path, 'r') as file:
        lp_content = file.readlines()

    # 解析目标函数
    objective_section = lp_content[lp_content.index('Maximize\n') + 1: lp_content.index('Subject to\n')]
    bid_prices = {}
    price_pattern = re.compile(r'\+([\d\.]+)\s*x(\d+)')
    for line in objective_section:
        matches = price_pattern.findall(line)
        for price, bid in matches:
            bid_prices[int(bid)] = float(price)

    # 解析约束
    constraints_section = lp_content[lp_content.index('Subject to\n') + 1: next(
        i for i, line in enumerate(lp_content) if "End" in line)]
    bid_items = defaultdict(list)
    constraint_pattern = re.compile(r'\+1\s*x(\d+)')
    for constraint_line in constraints_section:
        # 查找约束标识符，跟踪物品（如：c0, c1,... 对应于物品）
        if constraint_line.startswith(' c'):
            current_item = int(re.search(r'c(\d+)', constraint_line).group(1))
        matches = constraint_pattern.findall(constraint_line)
        for bid in matches:
            bid_items[int(bid)].append(current_item)

    # 计算物品和竞标的数量
    num_items = max(current_item + 1 for bids in bid_items.values() for current_item in bids)  # 假设0起始索引
    num_bids = len(bid_prices)

    # 提取竞标信息
    bid_info = {bid: {'items': items, 'price': bid_prices[bid]} for bid, items in bid_items.items()}
    return num_items, num_bids, bid_info


file = "instance/train/CA_m/CA_m_0.lp"
num_items, num_bids, bid_info = extract(file)

print(f"Number of items: {num_items}")
print(f"Number of bids: {num_bids}")

# 打印第49个竞标的信息
# bid_id = 49
# if bid_id in bid_info:
#     print(f"\nBid {bid_id} information:")
#     print(f"Items in bid: {bid_info[bid_id]['items']}")
#     print(f"Bid price: {bid_info[bid_id]['price']}")
# else:
#     print(f"Bid {bid_id} not found in the data.")

for bid, info in bid_info.items():
    print(f"Bid {bid}: Items: {info['items']}, Price: {info['price']}")
