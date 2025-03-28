#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=st-jjnunez-1-gpu
#SBATCH --time=4-00:00:00
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
code_length=256
nl_length=128
model_type=base #"base", "cocosoda" ,"multi-loss-cocosoda"
moco_k=1024
moco_m=0.999
lr=2e-5
moco_t=0.07
max_steps=1000
aug_type_way=random_replace_type 
data_aug_type=random_mask
base_model=DeepSoftwareAnalytics/CoCoSoDa
CUDA_VISIBLE_DEVICES=0

zero-shot () {
output_dir=./saved_models/zero-shot/${lang}
mkdir -p $output_dir

 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run.py   --eval_frequency  100 \
    --do_zero_short \
    --moco_m ${moco_m} --moco_t ${moco_t}  \
    --model_type ${model_type} \
    --output_dir ${output_dir}  \
    --data_aug_type ${data_aug_type} \
    --moco_k ${moco_k} \
    --config_name=${base_model}  \
    --model_name_or_path=${base_model} \
    --tokenizer_name=${base_model} \
    --lang=$lang \
    --do_test \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --code_length ${code_length} \
    --nl_length ${nl_length} \
    --eval_batch_size 128 \
    --learning_rate ${lr} \
    --seed 123456 2>&1| tee ${output_dir}/running.log
}
zero-shot