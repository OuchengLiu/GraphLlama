import math
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset
import json
import torch_sparse
import pyg_lib
from tqdm import tqdm
import concurrent.futures


K = 5
DATASET_NAME = 'ogbn-products'  # 'ogbn-arxiv'
pt_file_path = '/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/x_embs.pt'
# 并发数量：依赖于ThreadPoolExecutor的默认行为，通常是处理器核心数的5倍。这样做的好处是它可以自动适应运行代码的机器的硬件资源，但如果需要更精细的控制并发级别，可以通过在创建ThreadPoolExecutor时指定max_workers参数来实现。


# 加载数据集
def load_dataset(pt_file_path, set):
    embeddings = torch.load(pt_file_path)
    dataset = PygNodePropPredDataset(name=DATASET)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    set_idx = split_idx[set] # ['train'] ['valid'] ['test']
    return embeddings, set_idx, data


# 计算余弦相似度
def compute_similarity(embeddings, node, neighbors):
    # 假设 embeddings 是二维的：[num_nodes, num_features]
    node_embedding = embeddings[node]  # 添加一个维度以确保是二维的
    neighbor_embeddings = embeddings[neighbors]

    # 标准化嵌入向量
    node_embedding = node_embedding / node_embedding.norm(dim=1, keepdim=True)
    neighbor_embeddings = neighbor_embeddings / neighbor_embeddings.norm(dim=1, keepdim=True)

    # 计算相似度得分
    sim_scores = torch.mm(node_embedding, neighbor_embeddings.t()).squeeze(0)
    return sim_scores


# 获取2-hop邻域
def get_2hop_neighbors_parallel(data, node):
    # 该函数处理单个节点，获取其2-hop邻居
    loader = NeighborLoader(data, input_nodes=[node], num_neighbors=[-1, -1], batch_size=1, shuffle=False)
    for batch in loader:
        if batch.edge_index.size(1) == 0:
            return node, []  # 没有邻居的情况
        neighbors = batch.n_id.tolist()
        if node in neighbors:
            neighbors.remove(node)  # 移除当前节点自身
        return node, neighbors


def find_top_k_neighbors_for_node(embeddings, node, neighbors, k):
    if len(neighbors) == 0:
        return node, [node] * k  # 如果没有邻居，用自己填充
    sim_scores = compute_similarity(embeddings, node, neighbors)
    top_k_values, top_k_indices = torch.topk(sim_scores, k=min(k, len(neighbors)), largest=True)
    top_k_neighbors = [neighbors[i] for i in top_k_indices.tolist()]
    return node, top_k_neighbors


def find_top_k_neighbors_parallel(embeddings, two_hop_neighbors, k):
    # 创建一个字典来存储每个节点的top-k邻居
    top_k_neighbors = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 为每个节点提交找到top-k邻居的任务
        futures = [executor.submit(find_top_k_neighbors_for_node, embeddings, node, neighbors, k) 
                   for node, neighbors in two_hop_neighbors.items()]
        for future in concurrent.futures.as_completed(futures):
            node, top_k_neighbors_for_node = future.result()
            top_k_neighbors[node] = top_k_neighbors_for_node
    
    # 返回按照原始节点顺序组织的top-k邻居列表
    ordered_top_k_neighbors = {node: top_k_neighbors[node] for node in two_hop_neighbors}
    return ordered_top_k_neighbors


# 转换函数：将字典键转换为字符串，将Tensor转换为列表
def prepare_for_json(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if isinstance(key, torch.Tensor):
                # 如果键是Tensor，转换为它的数值
                new_key = key.item() if key.numel() == 1 else key.tolist()
            else:
                new_key = str(key)
            new_dict[new_key] = prepare_for_json(value)
        return new_dict
    elif isinstance(data, list):
        return [prepare_for_json(element) for element in data]
    elif isinstance(data, torch.Tensor):
        return data.tolist()
    else:
        return data


# 保存为JSON文件的函数
def save_to_json(data, file_name):
    data = prepare_for_json(data)
    print("save begin")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def save_to_pt(data, file_name):
    torch.save(data, file_name)


def extract_parallel(set_name):
    embeddings, set_idx, data = load_dataset(pt_file_path, set_name)
    data.edge_index = to_undirected(data.edge_index)
    
    # 使用ThreadPoolExecutor并行获取2-hop邻居
    two_hop_neighbors = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_2hop_neighbors_parallel, data, node.item()) for node in set_idx]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing nodes"):
            node, neighbors = future.result()
            two_hop_neighbors[node] = neighbors

    save_to_json(two_hop_neighbors , f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_neighbors.json')
    save_to_pt(two_hop_neighbors, f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_neighbors.pt')
    
    # two_hop_neighbors = torch.load(f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_neighbors.pt')
    k = K  # Top K相似节点
    top_k_neighbors = find_top_k_neighbors_parallel(embeddings, two_hop_neighbors, k)
    save_to_json(top_k_neighbors, f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_top_k_neighbors.json')
    save_to_pt(top_k_neighbors, f'/Data/{DATASET_NAME}/all-roberta-large-v1/main/cached_embs/{set}_top_k_neighbors.pt')


def main():
    # 对train, valid, test分别调用extract_parallel
    extract_parallel('train')
    extract_parallel('valid')
    extract_parallel('test')


if __name__ == "__main__":
    main()
