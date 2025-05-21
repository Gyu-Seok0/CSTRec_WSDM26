# #!/bin/bash

dataset_name=$1

case $dataset_name in
    "gowalla")
        cmd="python -u main_eval.py --d gowalla --use_csn --use_his --use_cur --PKA --update --lr 0.0001 --reg 5e-6 --dr 0.125 --num_C 50 --C_l 50 --num_H 30 --H_l 20 --num_PKA_nei 25 --mll 1e-4 --eval_RA --fc"
        ;;
    "ml-1m")
        cmd="python -u main_eval.py --d ml-1m --use_csn --use_his --use_cur --PKA --update --lr 0.0002 --reg 5e-6 --dr 0.05 --num_C 10 --C_l 20 --num_H 30 --H_l 10 --num_PKA_nei 25 --mll 1e-4 --eval_RA --fc"
        ;;
    "yelp")
        cmd="python -u main_eval.py --d yelp --use_csn --use_his --use_cur --PKA --update --lr 0.0002 --reg 5e-5 --dr 0.075 --num_C 30 --C_l 50 --num_H 30 --H_l 20 --num_PKA_nei 25 --mll 1e-4 --eval_RA --fc"
        ;;
    *)
        echo "Invalid dataset name! Please provide one of the following dataset names: yelp, amazon_game, citeulike_t."
        exit 1
        ;;
esac

echo $cmd
eval "$cmd"