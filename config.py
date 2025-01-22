import argparse

def get_config(method = None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', type = str, default = './datasets')
    parser.add_argument('--data', '--d', type = str, default = 'ml-1m')
    parser.add_argument('--num_block', type = int, default = 5, help = "The number of blocks (e.g., one base block(60%) & four incremental blocks(4 x 10%))")
    parser.add_argument('--window_size', type = int, default = 50, help = "The window zize of subsequences")
    parser.add_argument('--target_size', type = int, default = 1, help = "The target size for next-item prediction")
    parser.add_argument('--num_train_neg', '--ntn', type = int, default = 2, help = "The number of negative samples for each item in each seq (training)")
    parser.add_argument('--num_eval_neg', '--nen', type = int, default = 199, help = "The number of negative samples for each target item (valid / test) -> only for 'Leave-One-Out' setup")
    parser.add_argument('--batch_size', '--bs', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 0.0005, help = "The learning rate")
    parser.add_argument('--reg', type = float, default = 0., help = "The weight decay")
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--random_seed', '--rs', type = int, default = 0)
    parser.add_argument('--device', type = str, default = 0, help = "cpu or gpu_id")
    parser.add_argument('--hidden_dims', '--dims', type = int, default = 64)
    parser.add_argument('--dropout_rate', '--dr', type = float, default = 0.1)
    parser.add_argument('--num_layers', type = int, default = 2, help = "The number of layers in Transformer")
    parser.add_argument('--num_heads', type = int, default = 2, help = "The number of heads in each layer in Transformer")
    parser.add_argument('--val_cycle', type = int, default = 5, help = "Validation cycle in epochs")
    parser.add_argument('--neg_cycle', type = int, default = 1, help = "Negative sampling cycle in epochs")
    parser.add_argument('--eval_K_list', type = list, default = [5, 10, 20], help = "K list for evaluation")
    parser.add_argument('--early_stop', type = int, default = 10)
    parser.add_argument('--mode', type = str, default = 'Train', help = "The mode for dataloader (Train/Valid/Test)")
    parser.add_argument('--eval_mode', type = str, default = 'Full', help = "The mode for evaluation (Full or LOO (i.e., Leave-One-Out))")

    parser.add_argument('--cl_result_save_path', '--cs', type = str, default = None)
    parser.add_argument('--model_save_path', '--sp', type = str, default = None)
    parser.add_argument('--model_load_path', type = str, default = None)
    parser.add_argument('--method', '--m', type = str, default = None, help = "CL method such as Full_batch, Fine_tune, ... ")
    parser.add_argument('--warm_epoch', type = int, default = 10)
    parser.add_argument('--with_base_block', action = "store_true", help = "If it is true, we calculate the cl_results with the base block")
    parser.add_argument('--cl_evaluate', action = "store_true", help = "If it is true, we calculate the cl_results in validation process")
    parser.add_argument('--fast_check', '--fc', action="store_true", help = "It is used for debugging")

    # LWC-KD
    parser.add_argument('--num_LWC_KD_topk_neighbor', '--lwckd_top_n', type = int, default = 10, help = "The number of top-k neighbors in User-User or Item-Item relation")
    parser.add_argument('--num_LWC_KD_layer', '--lwckd_n_layer', type = int, default = 2, help = "The number of LWC_KD layers")
    parser.add_argument('--LWC_KD_temperature', '--lwckd_T', type = float, default = 1.)
    parser.add_argument('--LWC_KD_lambda', '--lwckd_la', type = float, default = 0.001)
    parser.add_argument('--update_start_block_id', type = int, default = 2, help = "The start block of LWC_KD")
    parser.add_argument('--LWC_KD_annealing', '--lwckd_anneal', action="store_true", help = "annealing")
    parser.add_argument('--LWC_KD_anneal_T', '--lwckd_anneal_T', type = float, default = 0.5, help = "The hyperparameter for scaling the value of LWC_KD_lambda through epochs")
    parser.add_argument('--item_side_only', action = argparse.BooleanOptionalAction, help = "If it is true, we use LWC_KD_PIW as item side")

    # PIW
    parser.add_argument('--num_cluster', '--piw_n_c', type = int, default = 10, help = "The start block of LWC_KD")
    parser.add_argument('--PIW_lambda', '--piw_la', type = float, default = 0.001)

    # Reloop2
    parser.add_argument('--memory_size', '--ms', type = int, default = 10)
    parser.add_argument('--sc_gamma', '--scg', type = float, default = 1.0, help = "The hyperparameter for scaling y_bar in the function of get_y_err (coarse-grained like 0.1, 0.2, ..)")
    parser.add_argument('--sc_lambda', '--scl', type = float, default = 0.5, help = "The hyperparameter for scaling y_err in the function of self_correct (fine-grained like 0.01, ... )")
    parser.add_argument('--sc_start_epoch', '--scs', type = int, default = 30, help = "The epoch of starting self-correction")
    
    # HPMN
    parser.add_argument('--num_HPMN_layer', '--nhl', type = int, default = 4)
    parser.add_argument('--num_HPMN_att_layer', '--nhal', type = int, default = 3)
    parser.add_argument('--HPMN_lambda', '--hl', type = float, default = 0.1, help = "The hyperparameter for scaling the covariance of memory slots")
    parser.add_argument('--use_HPMN_layer_norm', '--use_hl', action = argparse.BooleanOptionalAction)
    
    # LinRec
    parser.add_argument('--num_LinRec_layer', '--nll', type = int, default = 2)
    parser.add_argument('--num_LinRec_head', '--nlh', type = int, default = 2)
    
    # LimaRec
    parser.add_argument('--num_LimaRec_layer', '--nlil', type = int, default = 2)
    parser.add_argument('--num_LimaRec_head', '--nlih', type = int, default = 2)
    parser.add_argument('--LimaRec_lambda', '--lil', type = float, default = 0.)
    
    # Proposed
    parser.add_argument('--num_proposed_layer', '--npl', type = int, default = 2)
    parser.add_argument('--num_proposed_head', '--nph', type = int, default = 2)
    parser.add_argument('--std_lambda', '--vl', type = float, default = 0.)
    parser.add_argument('--prompt_lambda', '--pl', type = float, default = 0.001)
    
    parser.add_argument('--num_prompt', '--np', type = int, default = 10)
    parser.add_argument('--num_use_prompt', '--nup', type = int, default = 5)
    parser.add_argument('--use_stability', '--us', action = argparse.BooleanOptionalAction, help = "using stability")
    parser.add_argument('--use_plasticity', '--up', action = argparse.BooleanOptionalAction, help = "using plasticty")
    parser.add_argument('--feats_agg', '--fa', action = argparse.BooleanOptionalAction, help = "feature aggregation")
    # parser.add_argument('--prompt_start_epoch', '--pse', type = int, default = 1)
    parser.add_argument('--penalty_func', '--pf', type = str, default = "min_max", help = "normalization function for giving penalty to increase diversity of using prompts. options: softmax, min_max")
    
    parser.add_argument('--update', action = argparse.BooleanOptionalAction, help = "Update prev_r_z for previous users")
    parser.add_argument('--adaptive_update', '--ad_update', action = argparse.BooleanOptionalAction, help = "Adaptively update prev_rz value for new users")
    parser.add_argument('--adaptive_update_cycle', '--ad_update_cycle', type = int, default = 1, help = "The cycle of adaptive updating")
    parser.add_argument('--num_proposed_topk_neighbor', '--p_top_n', type = int, default = 10, help = "The number of top-k user neighbors for a new user to update its prev_r_z")
    parser.add_argument('--assign_func', '--af', type = str, default = "mean", help = "Prev_r_z assign function for new users. options: sim, mean")
    parser.add_argument('--proposed_temperature', '--p_T', type = float, default = 1.)
    parser.add_argument('--prompt_shared', '--ps', action = argparse.BooleanOptionalAction, help = "prompts shared")

    # IMSR
    parser.add_argument('--num_interest', type = int, default = 15)
    parser.add_argument('--num_routing_layer', type = int, default = 3)
    parser.add_argument('--multi_interest_func', '--mif', type = str, default = "self_attention", help = "The function of getting multi-interest (option: self_attention or routing")
    parser.add_argument('--IMSR_temperature', '--IMSR_t', type = float, default = 1.)
    parser.add_argument('--IMSR_kd_lambda', '--IMSR_kd', type = float, default = 0.001)
    parser.add_argument('--IMSR_logit_weight', '--IMSR_w', type = float, default = 0.001)


    # CSR
    parser.add_argument('--num_CSR_layer', type = int, default = 2)
    parser.add_argument('--num_CSR_head', type = int, default = 2)
    parser.add_argument('--CSR_std_lambda', type = float, default = 0.)
    parser.add_argument('--num_CSR_neighbor', '--CSR_nei', type = int, default = 20)
    parser.add_argument('--CSR_temperature', '--CSR_T', type = float, default = 1.)
    parser.add_argument('--num_P', type = int, default = 30)
    parser.add_argument('--num_S', type = int, default = 30)

    parser.add_argument('--P_length', '--P_l', type = int, default = 20)
    parser.add_argument('--S_length', '--S_l', type = int, default = 10)
    parser.add_argument('--matching_loss_lambda', '--mll', type = float, default = 0.001)
    parser.add_argument('--update_plasticity_cycle', '--upc', type = int, default = 1, help = "The cycle of updating plasticity prompts")
    parser.add_argument('--update_stability_cycle', '--usc', type = int, default = 1, help = "The cycle of updating stability prompts")
    parser.add_argument('--DPA', action = argparse.BooleanOptionalAction, help = "Deterministic Prompt Assignment")
    
    parser.add_argument('--FB', action = argparse.BooleanOptionalAction, help = "Full_batch")
    parser.add_argument('--eval_RA', action = argparse.BooleanOptionalAction, help = "Full_batch")

    args = parser.parse_args()
    
    if args.data == "ml-1m":
        args.max_user = 6028
        args.max_item = 2873
        args.window_size = 50
        args.hidden_dims = 64
        args.neg_cycle = 1            
        
    elif args.data == "gowalla":
        args.max_user = 34187
        args.max_item = 72461
        args.window_size = 25
        args.hidden_dims = 32
        args.neg_cycle = 5
    
    elif args.data == "yelp":
        args.max_user = 119691
        args.max_item = 56297
        args.window_size = 10
        args.hidden_dims = 16
        args.neg_cycle = 5
    
    if args.method == "Full_batch" or args.FB == True:
        args.window_size *= 2
    
    return args
