#!/bin/bash
eval_data_dir=$1
toxic_aae=$2
nontoxic_aae=$3
toxic_wae=$4
nontoxic_wae=$5
other=$6

python3 project/predict_aave.py \
	--input_path ${eval_data_dir} \
    --toxic_aae_path ${toxic_aae} \
    --nontoxic_aae_path ${nontoxic_aae} \
    --toxic_wae_path ${toxic_wae} \
    --nontoxic_wae_path ${nontoxic_wae} \
    --other_path ${other}

