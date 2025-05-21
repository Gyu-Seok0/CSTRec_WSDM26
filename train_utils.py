from time import time
from copy import deepcopy
from tqdm import tqdm

import torch

from utils import print_cl_result, get_gpu_memory
from eval_utils import continual_evaluate, evaluate

def train_epoch(dataloder, model, device, epoch, criterion, optimizer, epoch_losses, args):
    
    start_time = time()
    
    dataloder.dataset.mode = "Train"
    if epoch == 1 or epoch % args.neg_cycle == 0:
        print("\t[Negative Sampling]")
        dataloder.dataset.neg_sampling_train()
        
    model.train()
    for batch in tqdm(dataloder): # batch = {user_ids : tensor(), train_seqs : tensor(), pos_seqs : tensor(), neg_seqs : tensor()}
        
        # forward
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_losses = model.loss(criterion, **batch)
        total_loss = batch_losses['total_loss']
        
        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        for loss in epoch_losses.keys():
            if loss in batch_losses.keys():
                epoch_losses[loss] += batch_losses[loss].item()
    
    end_time = time()
    training_time = end_time - start_time
    
    for loss in epoch_losses.keys():
        epoch_losses[loss] = round(epoch_losses[loss] / len(dataloder), 4)
    epoch_losses['training_time'] = f"{training_time:.4f} secs"

    return epoch_losses
    
def train(dataloder, model, device, criterion, optimizer, losses,
          prev_valid_LA_sum, block_id, eval_dataloader_list, new_user_dataloader_list, seen_max_item_id_list, new_user_ids, args, target_metric):
    
    patience = 0
    best_valid_score = 0.0
    best_model_param = None
    train_results = list()
    
    if args.method == "CSR":
        model.block_id = block_id
    
    ''' Training for Current Block '''
    for epoch in range(1, args.max_epoch + 1):
        print(f"\n\t[Epoch:{epoch}/{args.max_epoch}]")
        
        if args.method == "Reloop2":
            model.cur_epoch = epoch
        
        should_current_update = (
            args.method == "CSR" and
            block_id >= 1 and
            args.use_current and
            ((epoch-1) % args.update_current_cycle == 0 or epoch == 1)
        )
        if should_current_update:
            model.get_current_interests(eval_dataloader_list[-1])
        
        should_historical_update = (
            args.method == "CSR" and
            model.use_history and
            args.use_historical and
            ((epoch-1) % args.update_historical_cycle == 0 or epoch == 1)
        )
        if should_historical_update:
            model.get_historical_interests(eval_dataloader_list[-1])
        
        should_PKA = (
            block_id >= args.update_start_block_id and
            args.method == "CSR" and
            args.PKA and
            ((epoch-1) % args.PKA_cycle == 0 or epoch == 1)
        )
        
        if should_PKA:
            if args.method == "CSR":
                model.update_history_new_users(eval_dataloader_list[-1], new_user_ids)
            
        
        epoch_losses = deepcopy(losses)
        epoch_result = train_epoch(dataloder, model, device, epoch, criterion, optimizer, epoch_losses, args)
        print(f"\t{epoch_result}")
        
        epoch_result['block_id'] = block_id
        epoch_result['epoch'] = epoch
        train_results.append(deepcopy(epoch_result))
                
        ''' Validation for Continual Learing '''
        if epoch == 1 or epoch % args.val_cycle == 0:
            if args.cl_evaluate:
                cl_valid_result, prev_block_valid_result = continual_evaluate(prev_valid_LA_sum, block_id, eval_dataloader_list, seen_max_item_id_list, model, device, args, mode = "Valid")
                print_cl_result(block_id, prev_block_valid_result, cl_valid_result)
                target_score = cl_valid_result["H_mean"]["H@20"] if target_metric == "prev" else prev_block_valid_result[f"block_{block_id}"]["H@20"] # history + current vs current
            else:
                print(f"\n\t[Valid]")
                eval_result = evaluate(model, eval_dataloader_list[-1], seen_max_item_id_list[-1], device, args.eval_K_list, mode = "Valid", eval_mode = args.eval_mode, method = args.method)
                print(f"\t\t[Eval_block_id:{block_id}]")
                print(f"\t\t{eval_result}\n")
                target_score = eval_result["H@20"]
                
            if best_valid_score < target_score:
                print(f"\n\t\tThe best valid score is changed: prev={best_valid_score:.4f} -> cur={target_score:.4f}")
                best_valid_score = target_score
                best_model_param = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                patience = 0
            else:
                if epoch > args.warm_epoch:
                    patience += args.val_cycle
                    print(f"\n\t\t[Patience:{patience}/{args.early_stop}]")
                    
                    if patience >= args.early_stop:
                        print("\t[Early Stopping]")
                        break
        get_gpu_memory()
        if args.fast_check and epoch >= 5:
            print("\n[Breaking by fast check]\n")
            break
        
    return best_model_param, train_results
