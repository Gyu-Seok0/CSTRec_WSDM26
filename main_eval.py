import os
import time
import cpuinfo
import argparse
from time import time
from datetime import timedelta
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import *
from train_utils import *
from eval_utils import *
from config import get_config
from data_loader import SeqRecDataset, NewUserDataset, FullBatchDataset

from models.SASREC import SASREC
from models.CSR import CSR

def main(args):
    
    # random_seed & device
    set_random_seed(args.random_seed)
    device = "cpu" if args.device == "cpu" else torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[Device: {device} & Random seed: {args.random_seed}]")
    
    # model/optimizer/criterion
    backbone = SASREC(args.max_user, args.max_item, device, args)
    losses = {'total_loss' : 0.0, 'pos_loss' : 0.0, 'neg_loss' : 0.0}
    
    if args.method == "CSR":
        model = eval(args.method)(args.max_user, args.max_item, args.hidden_dims, args.num_CSR_layer, args.num_CSR_head, args.dropout_rate, args.window_size, args.CSR_std_lambda, args.num_PKA_neighbor, args.CSR_temperature, 
                     args.use_current, args.use_historical, args.num_C, args.C_length, args.num_H, args.H_length, args.matching_loss_lambda, device)
        model.toggle_csn(args.use_csn)
        
        losses['std_loss'] = 0.0
        losses['matching_loss'] = 0.0
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    
    model = model.to(device)
    criterion = BCELoss() if args.method == "Reloop2" else BCEWithLogitsLoss()
    
    print(f"[Model]\n {model}")
    size_model = 0
    for param in model.parameters():
        size_model += param.numel() * torch.finfo(param.data.dtype).bits
    print(f"\n[Model size]: {size_model} bits | {size_model / 8e6:.2f} MB") # 8 means 8bits = 1byte, 10^6 byte = 1 MB
    print(f"\n[Method] {args.method}")
    print(f"\n[Loss] {criterion}")
    
    # dataset info
    data_path = os.path.join(args.data_folder, args.data) # blocks, new_user_item_ids, dormant_user_ids = load_data(data_path, args.num_block)
    new_user_dataloader_list = [None] # to align the index, we add None for block 0.
    eval_dataloader_list = [] # list[dataloder, ...]
    seen_max_user_id_list = [] # list[int, ...]
    seen_max_item_id_list = [] # list[int, ...]
    seen_user_ids = set()
    seen_item_ids = set()
    each_block_user_ids = [] # list[set(), ...]
    each_block_item_ids = [] # list[set(), ...]
    
    # eval info
    prev_valid_LA_sum = dict()
    prev_test_LA_sum = dict()
    cl_total_test_result = list() # list[dict(), ...]
    block_results = list()
    
    for K in args.eval_K_list:
        for metric in ["H", "M", "N"]:
            prev_valid_LA_sum[f"{metric}@{K}"] = 0.
            prev_test_LA_sum[f"{metric}@{K}"] = 0.

    for block_id in range(args.num_block):
        print(f"\n[Train_block_id:{block_id}]")
        
        ''' Dataset '''
        block = load_pickle(os.path.join(data_path, f"block_{block_id}"))
        prev_block = deepcopy(train_block) if block_id > 1 else None
        train_block = pd.concat([prev_block, block]) if args.method == "Full_batch" else block
        
        cur_user_ids, cur_item_ids, new_user_ids = get_block_info(train_block, seen_user_ids, seen_item_ids) # set(), set()
        seen_user_ids = seen_user_ids.union(cur_user_ids) # set()
        seen_item_ids = seen_item_ids.union(cur_item_ids) # set()
        seen_max_user_id_list.append(max(seen_user_ids)) # list[int, ...]
        seen_max_item_id_list.append(max(seen_item_ids)) # list[int, ...]
        each_block_user_ids.append(cur_user_ids) # list[set(), ...]
        each_block_item_ids.append(cur_item_ids) # list[set(), ...]
        
        prev_dataset = deepcopy(dataset) if block_id > 1 else None  
        dataset = SeqRecDataset(block, args.window_size, args.target_size, args.num_train_neg, args.num_eval_neg, args.mode, args.eval_mode, args.data, args.random_seed)
        dataloder = DataLoader(dataset, args.batch_size, shuffle = True, drop_last = False, pin_memory = True, num_workers = 1)
        eval_dataloader = DataLoader(dataset, args.batch_size, shuffle = False, drop_last = False, pin_memory = True, num_workers = 1)
        eval_dataloader_list.append(deepcopy(eval_dataloader))
        del eval_dataloader
    
        if block_id > 1 and args.FB == True:
            print("Using Full_batch_dataset!")
            dataset = FullBatchDataset(prev_dataset, dataset) # Integrate the previous and current dataset.
            dataloder = DataLoader(dataset, args.batch_size, shuffle = True, drop_last = False)
        print(f"# of data_instances:{dataset.__len__()}")
        
        if block_id >= 1:
            new_user_dataset = NewUserDataset(dataset, new_user_ids)
            new_user_dataloader = DataLoader(new_user_dataset, args.batch_size, shuffle = False, drop_last = False)
            new_user_dataloader_list.append(deepcopy(new_user_dataloader))
            del new_user_dataloader
            
            print(f"\n\t[Test for New Users Before Learning until block_id {block_id - 1}]")
            eval_result = evaluate(model, new_user_dataloader_list[-1], seen_max_item_id_list[-1], device, args.eval_K_list, mode = "Test", eval_mode = args.eval_mode, method = args.method)
            print(f"\t{eval_result}\n")
        
        if args.fast_check and block_id == 0:
            continue
                
        ''' Training & Validation '''
        if block_id >= args.update_start_block_id and args.method == "CSR" and args.update:
                model.update_history(eval_dataloader_list[-2])

        model = model.to(device)
        optimizer = Adam(model.parameters(), lr = args.lr, weight_decay = 0.0 if block_id == 0 else args.reg)
        best_model_param, train_results = train(dataloder, model, device, criterion, optimizer, losses, prev_valid_LA_sum, block_id, eval_dataloader_list, new_user_dataloader_list, seen_max_item_id_list, new_user_ids, args, target_metric = "cur")
        block_results += train_results
        
        # After training...
        model.load_state_dict(best_model_param)
        model = model.to(device)
                
        print("\n\t[The best valid result]")
        if args.eval_RA:
            best_cl_valid_result, best_prev_block_valid_result = continual_evaluate_RA(prev_valid_LA_sum, block_id, eval_dataloader_list, seen_max_item_id_list, model, device, args, mode = "Valid")
        else:
            best_cl_valid_result, best_prev_block_valid_result = continual_evaluate(prev_valid_LA_sum, block_id, eval_dataloader_list, seen_max_item_id_list, model, device, args, mode = "Valid")

        print_cl_result(block_id, best_prev_block_valid_result, best_cl_valid_result)
        
        ''' Test for Continual Learning '''
        if args.eval_RA:
            cl_test_result, prev_block_test_result = continual_evaluate_RA(prev_test_LA_sum, block_id, eval_dataloader_list, seen_max_item_id_list, model, device, args, mode = "Test")
        else:
            cl_test_result, prev_block_test_result = continual_evaluate(prev_test_LA_sum, block_id, eval_dataloader_list, seen_max_item_id_list, model, device, args, mode = "Test")

        print_cl_result(block_id, prev_block_test_result, cl_test_result)
        
        cl_total_test_result.append(cl_test_result)
        
        try:
            cl_result_to_csv(cl_total_test_result, args.eval_K_list, file_path = args.cl_result_save_path)
            log_path = "/".join(args.cl_result_save_path.split('/')[:-1])
            log_path = log_path.replace('cl_result', 'training_log')
            Path(log_path).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(block_results).to_csv(f"{log_path}/df_rs_{str(args.random_seed)}.csv", index=False)
            print(f"[SAVE training_log] {log_path}")
        except Exception as e:   
            print('[Error]', e)
        
        if block_id >= 1:
            print("\n\t[Test for New Users After Learning]")
            eval_result = evaluate(model, new_user_dataloader_list[-1], seen_max_item_id_list[-1], device, args.eval_K_list, mode = "Test", eval_mode = args.eval_mode, method = args.method)
            print(f"\t\t{eval_result}\n")
        
        # Save
        if args.model_save_path is not None:
            dir = Path(args.model_save_path)
            dir.mkdir(parents=True, exist_ok=True)
            model_save_path = os.path.join(dir, f"block_{block_id}.pth")
            
            torch.save(best_model_param, model_save_path)            
            print(f"[SAVE] {model_save_path}")
        
        # Update LA sum for continual eval
        if (args.with_base_block and block_id == 0) or block_id >= 1:
            for metric in prev_valid_LA_sum.keys():
                prev_valid_LA_sum[metric] += best_prev_block_valid_result[f"block_{block_id}"][metric]
                prev_test_LA_sum[metric] += prev_block_test_result[f"block_{block_id}"][metric]
        
if __name__ == "__main__":
    args = get_config()
    print('[Checking for CPU]')
    info = cpuinfo.get_cpu_info()
    print(f"brain:{info['brand_raw']}, cnt:{os.getenv('SLURM_CPUS_PER_TASK')}")    

    print('\n[Checking for GPU]')
    _NUMBER_OF_GPU = check_gpu()
    if _NUMBER_OF_GPU > 0:
        print_gpu_usage(_NUMBER_OF_GPU)
    else:
        print("No GPU found.")
    
    run_start = time()
    print_command_with_args(args, call_running_start = True)
    main(args)
    print_command_with_args(args, call_running_start = False)
    run_end = time()
    print(f"[Total running time] {timedelta(seconds = run_end - run_start)} secs")
