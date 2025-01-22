from tqdm import tqdm
from time import time
from copy import deepcopy

import numpy as np
import torch
from utils import to_np

# def evaluate(model, dataloder, seen_max_item_id, device, eval_K_list, mode, eval_mode, method):
    
#     start_infer_time = time()
#     total_pos_rank = []
    
#     model.eval()
#     with torch.no_grad():
#         dataloder.dataset.mode = mode
        
#         for batch in tqdm(dataloder):
            
#             # batch
#             eval_seq = to_np(batch['seq'])
#             target_items = to_np(batch['target_item'])
#             batch = {key: value.to(device) for key, value in batch.items()}
            
#             # predict
#             scores = to_np(model.predict(eval_mode, return_score = True, **batch))
            
#             if eval_mode == "Full":
#                 # seen items should have '-np.inf' score
#                 batch_size, window_size = eval_seq.shape
#                 row = np.repeat(np.arange(batch_size), window_size)
#                 col = eval_seq.reshape(-1)
#                 scores[(row, col)] = -np.inf
#                 scores[:, 0] = -np.inf # 0-index item for zero padding should have '-np.inf' score
#                 scores[:, seen_max_item_id + 1:] -np.inf # future items should have '-np.inf' score
                
#                 sorted_indices = np.argsort(scores, axis = 1)[:, ::-1]
#                 pos_rank = np.where(sorted_indices == target_items.reshape(-1, 1))[1] + 1 # 1-based rank
#                 pos_rank = pos_rank[target_items != 0] # if target_item_id == 0, it is excluded in evaluation.
            
#             elif eval_mode == "LOO":
#                 sorted_indices = np.argsort(scores, axis = 1)[:, ::-1]
#                 pos_rank = np.where(sorted_indices == 0)[1] + 1 # 1-based rank
#                 pos_rank = pos_rank[target_items[:, 0] != 0] # if target_item_id == 0, it is excluded in evaluation.
 
#             total_pos_rank = np.append(total_pos_rank, pos_rank)
    
#     eval_result = {}
#     mrr_base_table = (1/total_pos_rank)
#     ndcg_base_table = (1/np.log2(total_pos_rank + 1)) / (1/np.log2(1 + 1)) # (dcg) / (idcg)
    
#     for K in eval_K_list:
#         hit_table_K = total_pos_rank <= K
#         mrr_table_K = mrr_base_table * hit_table_K
#         ndcg_table_K = ndcg_base_table * hit_table_K
        
#         eval_result[f"H@{K}"] = round(hit_table_K.mean(), 4)
#         eval_result[f"M@{K}"] = round(mrr_table_K.mean(), 4)
#         eval_result[f"N@{K}"] = round(ndcg_table_K.mean(), 4)
    
#     end_infer_item = time()
#     eval_result["infer_time"] = f"{end_infer_item - start_infer_time:.4f} secs"
    
#     return eval_result

def evaluate(model, dataloader, seen_max_item_id, device, eval_K_list, mode, eval_mode, method, cur_block_id = None):
    start_infer_time = time()
    total_pos_rank = []

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        dataloader.dataset.mode = mode
        
        # if method == "CSR" and mode == "Test" and cur_block_id == 2:
        #     print("EVALUATE")
        #     import pdb;pdb.set_trace()
        
        
        # if cur_block_id == 2:
        #     import pdb; pdb.set_trace()

        for batch in tqdm(dataloader):
            # Transfer batch to device (GPU/CPU)
            batch = {key: value.to(device) for key, value in batch.items()}
            
            # Prediction (stay on GPU until necessary)
            scores = model.predict(eval_mode, return_score=True, **batch)

            if eval_mode == "Full":
                eval_seq = batch['seq']  # This stays on GPU
                target_items = batch['target_item']  # On GPU
                batch_size, window_size = eval_seq.shape
                
                # Mask out seen items and future items
                row = torch.arange(batch_size).repeat_interleave(window_size)
                col = eval_seq.view(-1)
                scores[row, col] = -np.inf  # Mask seen items
                scores[:, 0] = -np.inf  # Mask zero-padding items
                scores[:, seen_max_item_id + 1:] = -np.inf  # Mask future items

                # Use np.argpartition to get top K ranks
                K = max(eval_K_list)
                sorted_indices = torch.argsort(scores, descending=True, dim=1)
                pos_rank = (sorted_indices == target_items.view(-1, 1)).nonzero()[:, 1] + 1  # 1-based rank

                # Exclude target_item_id == 0 (padding) from evaluation
                pos_rank = pos_rank[target_items.view(-1) != 0]

            elif eval_mode == "LOO":
                eval_seq = batch['seq']  # Stays on GPU
                target_items = batch['target_item']  # On GPU
                
                # Use np.argpartition to get top K ranks
                K = max(eval_K_list)
                sorted_indices = torch.argsort(scores, descending=True, dim=1)
                pos_rank = (sorted_indices == target_items.view(-1, 1)).nonzero()[:, 1] + 1  # 1-based rank

                # Exclude target_item_id == 0 (padding) from evaluation
                pos_rank = pos_rank[target_items.view(-1) != 0]

            # Store results on CPU (for final metric calculation)
            total_pos_rank = np.append(total_pos_rank, pos_rank.cpu().numpy())

    # Metric Calculation
    eval_result = {}
    mrr_base_table = (1 / total_pos_rank)
    ndcg_base_table = (1 / np.log2(total_pos_rank + 1)) / (1 / np.log2(1 + 1))  # (dcg) / (idcg)

    for K in eval_K_list:
        hit_table_K = total_pos_rank <= K
        mrr_table_K = mrr_base_table * hit_table_K
        ndcg_table_K = ndcg_base_table * hit_table_K

        eval_result[f"H@{K}"] = round(hit_table_K.mean(), 4)
        eval_result[f"M@{K}"] = round(mrr_table_K.mean(), 4)
        eval_result[f"N@{K}"] = round(ndcg_table_K.mean(), 4)

    end_infer_time = time()
    eval_result["infer_time"] = f"{end_infer_time - start_infer_time:.4f} secs"

    return eval_result

def continual_evaluate(prev_LA_sum, cur_block_id, eval_dataloader_list, seen_max_item_id_list, model, device, args, mode):
    
    print(f"\n\t[{mode}]")
    
    LA = deepcopy(prev_LA_sum) # dict()
    RA = {key: 0. for key in LA.keys()}
    H_mean = {}

    cl_eval_result = {}
    prev_block_eval_result = {}
    
    # Eval
    for eval_block_id in range(cur_block_id + 1):
        
        eval_dataloader = eval_dataloader_list[eval_block_id]
        seen_max_item_id = seen_max_item_id_list[eval_block_id]
        
        eval_result = evaluate(model, eval_dataloader, seen_max_item_id, device, args.eval_K_list, mode, args.eval_mode, args.method, cur_block_id) # dict()        
        prev_block_eval_result[f"block_{eval_block_id}"] = eval_result
        
        if (args.with_base_block and eval_block_id == 0) or eval_block_id >= 1:
            for metric in RA.keys(): # H@K, M@K, N@K
                value = eval_result[metric]
                
                if eval_block_id < cur_block_id: # RA does not include the current block.
                    RA[metric] += value
            
                elif eval_block_id == cur_block_id: # LA should update the only current block with the result of previous block.
                    LA[metric] += value  
    
    # H-mean
    denominator = cur_block_id if args.with_base_block else cur_block_id -1
    denominator += 1e-8
    for metric in RA.keys(): # H@K, M@K, N@K
        RA[metric] /= denominator  # RA does not include the current block.
        LA[metric] /= (denominator + 1)
        H_mean[metric] = (2* RA[metric] * LA[metric]) / (RA[metric] + LA[metric] + 1e-8)
        
        RA[metric] = round(RA[metric], 4)
        LA[metric] = round(LA[metric], 4)
        H_mean[metric] = round(H_mean[metric], 4)
    
    # Save
    for cl_metric in ["RA", "LA", "H_mean"]:
        cl_eval_result[cl_metric] = eval(cl_metric)
    
    return cl_eval_result, prev_block_eval_result

def continual_evaluate_RA(prev_LA_sum, cur_block_id, eval_dataloader_list, seen_max_item_id_list, model, device, args, mode):
    
    print(f"\n\t[{mode}]")
    
    LA = deepcopy(prev_LA_sum) # dict()
    RA = {key: 0. for key in LA.keys()}
    H_mean = {}

    cl_eval_result = {}
    prev_block_eval_result = {}
    
    # Eval
    for eval_block_id in range(cur_block_id + 1):
        
        eval_dataloader = eval_dataloader_list[eval_block_id]
        seen_max_item_id = seen_max_item_id_list[eval_block_id]
        
        eval_result = evaluate(model, eval_dataloader, seen_max_item_id, device, args.eval_K_list, mode, args.eval_mode, args.method, cur_block_id) # dict()        
        prev_block_eval_result[f"block_{eval_block_id}"] = eval_result
        
        if (args.with_base_block and eval_block_id == 0) or eval_block_id >= 1:
            for metric in RA.keys(): # H@K, M@K, N@K
                value = eval_result[metric]
                
                if eval_block_id <= cur_block_id: # RA includes the current block.
                    RA[metric] += value
            
                if eval_block_id == cur_block_id: # LA should update the only current block with the result of previous block.
                    LA[metric] += value  
    
    # H-mean
    denominator = cur_block_id if args.with_base_block else cur_block_id -1
    denominator += 1e-8
    for metric in RA.keys(): # H@K, M@K, N@K
        RA[metric] /= (denominator + 1)
        LA[metric] /= (denominator + 1)
        H_mean[metric] = (2* RA[metric] * LA[metric]) / (RA[metric] + LA[metric] + 1e-8)
        
        RA[metric] = round(RA[metric], 4)
        LA[metric] = round(LA[metric], 4)
        H_mean[metric] = round(H_mean[metric], 4)
    
    # Save
    for cl_metric in ["RA", "LA", "H_mean"]:
        cl_eval_result[cl_metric] = eval(cl_metric)
    
    return cl_eval_result, prev_block_eval_result