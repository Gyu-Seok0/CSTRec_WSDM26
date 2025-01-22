from time import time
from copy import deepcopy
from tqdm import tqdm

import torch

from utils import print_cl_result, get_gpu_memory
from eval_utils import continual_evaluate, evaluate

def train_epoch(dataloder, model, device, epoch, criterion, optimizer, epoch_losses, args, block_id):
    
    start_time = time()
    
    dataloder.dataset.mode = "Train"
    if epoch == 1 or epoch % args.neg_cycle == 0:
        dataloder.dataset.neg_sampling_train()
        
    # if block_id == 2 and args.method == "CSR":
    #     print(f"Train block_id:{block_id} for debug")
    #     import pdb; pdb.set_trace()
        
    
    model.train()
    for batch in tqdm(dataloder): # batch = {user_ids : tensor(), train_seqs : tensor(), pos_seqs : tensor(), neg_seqs : tensor()}
        
        # forward
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_losses = model.loss(criterion, **batch)
        total_loss = batch_losses['total_loss']
        
        # backward
        optimizer.zero_grad()
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # add
        # import pdb; pdb.set_trace() 
        # for name, param in model.named_parameters():
        #     print(name, param, param.grad)
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
    
    if args.method in ["Proposed", "CSR", "CSR_delta"]:
        model.block_id = block_id
    
    # if args.method == "CSR":
    #     model.update_block_id(block_id)
    
    ''' Training for Current Block '''
    for epoch in range(1, args.max_epoch + 1):
        print(f"\n\t[Epoch:{epoch}/{args.max_epoch}]")
        
        if args.method == "Reloop2":
            model.cur_epoch = epoch
        
        should_plasticity_update = (
            args.method in ["CSR","CSR_delta"] and
            block_id >= 1 and
            args.use_plasticity and
            (epoch % args.update_plasticity_cycle == 0 or epoch == 1)
        )
        if should_plasticity_update:
            model.get_plasticity_prompts(eval_dataloader_list[-1])
        
        should_stability_update = (
            args.method in ["CSR", "CSR_delta"] and
            model.use_history and
            args.use_stability and
            (epoch % args.update_stability_cycle == 0 or epoch == 1)
        )
        if should_stability_update:
            model.get_stability_prompts(eval_dataloader_list[-1])
        
        should_adaptively_update = (
            block_id >= args.update_start_block_id and
            args.method in ["Proposed", "CSR", "CSR_delta"] and
            args.adaptive_update and
            (epoch % args.adaptive_update_cycle == 0 or epoch == 1)
        )
        
        if should_adaptively_update:
            if args.method == "Proposed":
                model.adaptive_update_for_Users(eval_dataloader_list[-1], new_user_ids, all_user = True)
            elif args.method in ["CSR", "CSR_delta"]:
                model.update_hisotry_new_users(eval_dataloader_list[-1], new_user_ids)
            
            # print("\n\t[Test for New Users while Learning]")
            # eval_result = evaluate(model, new_user_dataloader_list[-1], seen_max_item_id_list[-1], device, args.eval_K_list, mode = "Test", eval_mode = args.eval_mode, method = args.method)
            # print(f"\t{eval_result}\n")
        
        epoch_losses = deepcopy(losses)
        epoch_result = train_epoch(dataloder, model, device, epoch, criterion, optimizer, epoch_losses, args, block_id)
        print(f"\t{epoch_result}")
        
        if args.method == "LWC_KD_PIW":
            if args.LWC_KD_annealing and model.LWC_KD_lambda > 1e-8:
                model.LWC_KD_lambda *= torch.exp(torch.tensor(-epoch)/args.LWC_KD_anneal_T).to(device)
        
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
        
    return best_model_param
