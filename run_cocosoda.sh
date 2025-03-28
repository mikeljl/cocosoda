#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=st-jjnunez-1-gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1

module spider miniconda3

# Initialize conda
source /arc/software/spack-2024/opt/spack/linux-rocky9-skylake_avx512/gcc-9.4.0/miniconda3-4.9.2-i27tct3b7z5gwpnz6b5y3dfviowrjv4o/etc/profile.d/conda.sh


# Step 3: Activate your environment

conda activate CoCoSoDa

lang=ruby
current_time=$(date "+%Y%m%d%H%M%S")
# current_time=tmp

code_length=256
nl_length=128

model_type=multi-loss-cocosoda #"base", "cocosoda" 
moco_k=1024 #queue size
moco_m=0.999
lr=2e-5
moco_t=0.07 #temperature, higher value means smoother distribution, distinguish harder negatives better

epoch=10
batch_size=64 #!!! change from 128 to 64

max_steps=100000 #number of steps to take in pre-training 
save_steps=1000 #save every 1000 steps
data_aug_type="replace_type"
couninue_pre_train_data_files='dataset/java/train.jsonl dataset/javascript/train.jsonl  dataset/python/train.jsonl  dataset/php/train.jsonl  dataset/go/train.jsonl dataset/ruby/train.jsonl'
CUDA_VISIBLE_DEVICES=0
base_model=unixcoder

function continue_pre_train () {
output_dir=./saved_models/cocosoda/
mkdir -p $output_dir
echo ${output_dir}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  python run.py  --eval_frequency  100 \
    --moco_m ${moco_m} --moco_t ${moco_t}  \
    --output_dir ${output_dir}  \
    --moco_k ${moco_k} \
    --model_type ${model_type} \
    --data_aug_type other \
    --config_name=microsoft/${base_model}-base  \
    --model_name_or_path=microsoft/${base_model}-base \
    --tokenizer_name=microsoft/${base_model}-base \
    --lang=$lang \
    --do_test \
    --time_score 1 \
    --do_multi_lang_continue_pre_train \
    --max_steps ${max_steps} \
    --save_steps ${save_steps} \
    --gradient_accumulation_steps 1 \
    --logging_steps 50 \
    --couninue_pre_train_data_files  ${couninue_pre_train_data_files} \
    --train_data_file=dataset/$lang/train.jsonl \
    --eval_data_file=dataset/$lang/valid.jsonl  \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --num_train_epochs ${epoch} \
    --code_length ${code_length} \
    --nl_length ${nl_length} \
    --train_batch_size ${batch_size} \
    --eval_batch_size 32 \
    --learning_rate ${lr} \
    --seed 123456 2>&1| tee ${output_dir}/save_tokenizer.log
}

#     --eval_batch_size 64 \ 


continue_pre_train 
