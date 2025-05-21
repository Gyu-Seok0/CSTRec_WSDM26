import os
import sys
import pickle
import random
from datetime import datetime
import gc

import numpy as np
import pandas as pd
import torch
import nvidia_smi

def to_np(x):
    return x.detach().cpu().numpy()

def print_command_with_args(args, call_running_start = False):
    print("\n[Running start]") if call_running_start else print("\n[Running end]")
    print(f"[Time] {datetime.now()}")
    print(f"[CMD] {'python -u ' + ' '.join(sys.argv)}")
    print(f"[Args] {vars(args)}\n")

def save_pickle(path, data):
    file_path = path + ".pickle"
    with open(file_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print(f"[SAVE] {file_path}")

def load_pickle(path):
    file_path = path + ".pickle"
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        
    print(f"[LOAD] {file_path}")
    return data

def load_data(data_path, num_blocks = 5):
    
    blocks = []
    for idx in range(num_blocks):
        blocks.append(load_pickle(os.path.join(data_path, f"block_{idx}")))
    
    new_user_item_ids = load_pickle(os.path.join(data_path, "new_user_item_ids")) # {block_id: {user_id:list(), item_id:list()}, ... }
    dormant_user_ids = load_pickle(os.path.join(data_path, "user_ids_only_first_last_block")) # list()
    
    return blocks, new_user_item_ids, dormant_user_ids

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def print_cl_result(block_id, prev_block_eval_result, cl_eval_result):
    # H@K, M@K, N@K for each block
    # for eval_block_id in range(block_id + 1):
    #     print(f"\t\t[Eval_block_id:{eval_block_id}]")
    #     print(f"\t\t{prev_block_eval_result[f'block_{eval_block_id}']}\n")
    
    for key, item in prev_block_eval_result.items():
        print(f"\t\t[Eval_{key}]")
        print(f"\t\t{item}\n")
    
    # CL metric: RA, LA, H_mean
    print("\t\t[CL Result]")
    for cl_metric in ["RA", "LA", "H_mean"]:
        print(f"\t\t{cl_metric}: {cl_eval_result[cl_metric]}")
        
        
def cl_result_to_csv(cl_total_test_result, eval_K_list, file_path = None):
    
    rs_metrics = [f"{M}@{K}" for K in eval_K_list for M in ["H", "M", "N"]]
    cl_metrics = ['RA', 'LA', 'H_mean']

    df = pd.DataFrame(index = rs_metrics)

    for block_id, block_cl_result in enumerate(cl_total_test_result):
        for cl_metric, rs_metrics_value in block_cl_result.items(): # key
            for rs_metric, rs_metric_value in rs_metrics_value.items():
                df.at[rs_metric, f'After Block {block_id} {cl_metric}'] = rs_metric_value

    num_block = len(cl_total_test_result)
    num_cl_metric = len(cl_metrics)

    col_1 = [f"After Block {i}" for i in range(num_block) for _ in range(num_cl_metric)]
    col_2 = ["RA", "LA", "H_mean"] * num_block
    #df.columns = [col_1, col_2]
    df.columns = pd.MultiIndex.from_arrays([col_1, col_2])

    df.to_csv(f"{file_path}.csv", header=True)
    print(f"[SAVE CL Result as DataFrame into CSV] {file_path}")

def get_block_info(block, seen_user_ids, seen_item_ids):
    
    cur_user_ids = set(block.user_id)
    cur_item_ids = set(block.item_id)
    
    new_user_ids = cur_user_ids.difference(seen_user_ids)
    new_item_ids = cur_item_ids.difference(seen_item_ids)
    
    num_interactions = block.shape[0]
    UI_SIZE = len(cur_user_ids) * len(cur_item_ids)
    sparsity = 1 - (num_interactions / UI_SIZE)
    
    # prev_user_id_set = prev_user_id_set.union(cur_user_id_set)
    # prev_item_id_set = prev_item_id_set.union(cur_item_id_set)

    print(f"user:{len(cur_user_ids)}(new:{len(new_user_ids)}), item:{len(cur_item_ids)}(new:{len(new_item_ids)}), interactions:{num_interactions}, avg.seq_length:{block.groupby('user_id').count().iloc[:, -1].mean():.4f}, sparsity:{sparsity:.4f}")
    
    return cur_user_ids, cur_item_ids, new_user_ids

def sparse_mat_diagonal_zero(sparse_mat):
    # Get the indices and values of the sparse matrix
    indices = sparse_mat._indices()
    values = sparse_mat._values()

    # Create a mask to filter out the diagonal values
    mask = indices[0] != indices[1]

    # Filter out the diagonal values
    new_indices = indices[:, mask]
    new_values = values[mask]

    # Create a new sparse matrix with zeroed diagonal
    return torch.sparse_coo_tensor(new_indices, new_values, sparse_mat.size()).to(sparse_mat.device)

def sparse_mat_rowwise_division(sparse_mat, sqrt = False):
    # Sum of values along the rows
    row_sum = torch.sparse.sum(sparse_mat, dim=1).to_dense()
    
    if sqrt:
        row_sum = torch.sqrt(row_sum)

    # Get the non-zero values and indices
    values = sparse_mat._values()
    indices = sparse_mat._indices()

    # Divide the values by the corresponding row sums
    row_indices = indices[0]
    row_divisor = row_sum[row_indices]

    # Perform the division
    new_values = values / row_divisor

    # Create a new sparse matrix with the divided values
    return torch.sparse_coo_tensor(indices, new_values, sparse_mat.size()).to(sparse_mat.device)

def dict2rating_sparse_mat(dataset:dict, row_size, col_size):
    row = []
    col = []
    for user_id, item_ids in dataset.items(): # dict()
        row += [user_id] * len(item_ids)
        col += item_ids
    value = [1] * len(row)
    return torch.sparse_coo_tensor((row, col), value, (row_size, col_size)).half() # U x I

def prepare_LWCKD_data(block_id, prev_dataset, dataset, seen_max_user_id_list, seen_max_item_id_list, num_topk_neighbor, device = 'cpu'):
    
    cur_R = dict2rating_sparse_mat(dataset.User, 
                                   row_size = seen_max_user_id_list[block_id] + 1,
                                   col_size = seen_max_item_id_list[block_id] + 1).to(device)
    
    prev_R = dict2rating_sparse_mat(prev_dataset.User, 
                                    row_size = seen_max_user_id_list[block_id - 1] + 1,
                                    col_size = seen_max_item_id_list[block_id - 1] + 1).to(device)
    
    prev_user_size, prev_item_size = prev_R.shape
    
    prev_UU = sparse_mat_diagonal_zero(torch.sparse.mm(prev_R, prev_R.T)) # U x U
    prev_UU_neighbor = torch.topk(prev_UU.to_dense(), k = num_topk_neighbor).indices.tolist() # U x topk
    prev_UU_dict = dict(zip(range(len(prev_UU_neighbor)), prev_UU_neighbor))
    prev_UU = dict2rating_sparse_mat(prev_UU_dict, row_size = prev_user_size, col_size = prev_user_size).to(device)

    prev_II = sparse_mat_diagonal_zero(torch.sparse.mm(prev_R.T, prev_R)) # I x I
    prev_II_neighbor = torch.topk(prev_II.to_dense(), k = num_topk_neighbor).indices.tolist() # I x topk
    prev_II_dict = dict(zip(range(len(prev_II_neighbor)), prev_II_neighbor))
    prev_II = dict2rating_sparse_mat(prev_II_dict, row_size = prev_item_size, col_size = prev_item_size).to(device)
    
    return cur_R, prev_R, prev_UU, prev_II

# https://gist.github.com/s-mawjee/ad0d8e0c7e07265cae097899fe48c023

def check_gpu():
    nvidia_smi.nvmlInit()
    _NUMBER_OF_GPU = nvidia_smi.nvmlDeviceGetCount()
    return _NUMBER_OF_GPU

def print_gpu_usage(_NUMBER_OF_GPU):
    for i in range(_NUMBER_OF_GPU):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        name = nvidia_smi.nvmlDeviceGetName(handle)  # GPU 이름 가져오기
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f'GPU-{i}: {name.decode("utf-8")}, GPU-Memory: {_bytes_to_megabytes(info.used)}/{_bytes_to_megabytes(info.total)} MB')

def _bytes_to_megabytes(bytes):
    return round((bytes/1024)/1024,2)

def pairwise_cos_distance(x1, x2): # 0 ~ 2
    return 1 - torch.mm(L2_normalization(x1), 
                        L2_normalization(x2).T)

def L2_normalization(x):
    norms = torch.norm(x, dim = -1, keepdim = True)
    norms = torch.clamp(norms, min = 1e-8)
    return (x / norms + 1e-8)

def print_model_details(model):
    print("=== Trainable Parameters ===")
    total_params = 0
    for name, p in model.named_parameters():
        bytes_ = p.numel() * p.element_size()
        total_params += bytes_
        shape_str = str(tuple(p.shape))
        print(f"{name:40s} | shape={shape_str:15s} | {bytes_/1e6:8.3f} MB")
    
    print("\n=== Registered Buffers ===")
    total_bufs = 0
    for name, b in model.named_buffers():
        bytes_ = b.numel() * b.element_size()
        total_bufs += bytes_
        shape_str = str(tuple(b.shape))
        print(f"{name:40s} | shape={shape_str:15s} | {bytes_/1e6:8.3f} MB")
    
    print(f"\nTotal params size: {total_params/1e6:.3f} MB")
    print(f"Total buffers size: {total_bufs/1e6:.3f} MB")
    print(f"Overall model size: {(total_params+total_bufs)/1e6:.3f} MB")