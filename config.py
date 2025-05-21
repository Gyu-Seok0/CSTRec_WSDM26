import argparse

def get_config(method = None):
    parser = argparse.ArgumentParser()
    
    # Learning
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
    parser.add_argument('--method', '--m', type = str, default = "CSR", help = "CL method such as Full_batch, Fine_tune, ... ")
    parser.add_argument('--warm_epoch', type = int, default = 10)
    parser.add_argument('--with_base_block', action = "store_true", help = "If it is true, we calculate the cl_results with the base block")
    parser.add_argument('--cl_evaluate', action = "store_true", help = "If it is true, we calculate the cl_results in validation process")
    parser.add_argument('--fast_check', '--fc', action="store_true", help = "It is used for debugging")
    parser.add_argument('--update_start_block_id', type = int, default = 2, help = "The start block of LWC_KD")

    # CSR
    parser.add_argument('--num_CSR_layer', type = int, default = 2)
    parser.add_argument('--num_CSR_head', type = int, default = 2)
    parser.add_argument('--CSR_std_lambda', type = float, default = 0.)
    parser.add_argument('--CSR_temperature', '--CSR_T', type = float, default = 1.)
    parser.add_argument('--num_C', type = int, default = 30)
    parser.add_argument('--num_H', type = int, default = 30)

    parser.add_argument('--C_length', '--C_l', type = int, default = 20)
    parser.add_argument('--H_length', '--H_l', type = int, default = 10)
    parser.add_argument('--matching_loss_lambda', '--mll', type = float, default = 0.001)
    parser.add_argument('--update_current_cycle', '--upc', type = int, default = 5, help = "The cycle of updating current interest pool")
    parser.add_argument('--update_historical_cycle', '--usc', type = int, default = 5, help = "The cycle of updating historical interest pool")
    
    parser.add_argument('--FB', action = argparse.BooleanOptionalAction, help = "Full_batch")
    parser.add_argument('--eval_RA', action = argparse.BooleanOptionalAction, help = "Full_batch")
    parser.add_argument('--use_csn', '--use_csn', action = argparse.BooleanOptionalAction)
    parser.add_argument('--use_historical', '--use_his', action = argparse.BooleanOptionalAction, help = "using historical")
    parser.add_argument('--use_current', '--use_cur', action = argparse.BooleanOptionalAction, help = "using plasticty")
    parser.add_argument('--update', action = argparse.BooleanOptionalAction, help = "Update prev_r_z for previous users")
    parser.add_argument('--PKA', action = argparse.BooleanOptionalAction, help = "Adaptively update prev_rz value for new users")
    parser.add_argument('--num_PKA_neighbor', '--num_PKA_nei', type = int, default = 20)
    parser.add_argument('--PKA_cycle', type = int, default = 5, help = "The cycle of adaptive updating")
   
    # eval_opt
    parser.add_argument('--opt', type = int, default = 0)

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
    
    if args.FB == True:
        args.window_size *= 2
    
    return args
