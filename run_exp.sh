#!/bin/bash

base_model_path=path_to_your_base_model # e.g. a path to a 7B vicuna-v1.5 model.
output_dir=path_to_your_output_directory # e.g. model_checkpoints/, NOTE: need to be end with a "/"
infer_dir=path_to_directory_to_store_inference # e.g. infer/, NOTE: need to be end with a "/"
max_infer_sample=5000 # max samples to be inferred per sample. Default uses the test split of sft dataset for inference.
model=test_ulma  # prefix for output model name
pairwise_dataset=hh_rlhf_en  # hh_rlhf_en: is the original HH dataset. hh_rlhf_en_golden_rejected: is the HH_golden.
pointwise_dataset=${pairwise_dataset}_pointwise # hh_rlhf_en_pointwise: original. hh_rlhf_en_golden_rejected_pointwise: golden.
sft_dataset_rlhf=${pairwise_dataset}_sft # hh_rlhf_en_sft: original. hh_rlhf_en_golden_rejected_sft: golden.
step=1 # which step to start from (useful for stop & re-run):
          # 0: reserved for pretrain,
          # 1: ULMA & pointwise DPO,
          # 2: pairwise DPO,
          # 3: Unlikelihood,
          # 4. SFT,
          # 5: RM
          # 6. RLHF.
          # 7. Inference (for all models.)
num_gpus=8 # If run on single card, replace the fist line of each command by: CUDA_VISIBLE_DEVICES=0 python src/train_bash.py



# ULMA: log loss, Pointwise DPO: pointwise loss
if [ $step -le 1 ]; then
    sft_loss_type=("logloss" "pointwise_dpo")
    version=("${model}_ulma" "${model}_pointwise_dpo")

    for i in "${!sft_loss_type[@]}"
    do
        deepspeed --num_gpus ${num_gpus} --master_port=9902 src/train_bash.py \
            --stage ulma \
            --model_name_or_path ${base_model_path}  \
            --do_train True \
            --dataset "${pointwise_dataset}" \
            --template default \
            --finetuning_type lora \
            --lora_target q_proj,v_proj \
            --resume_lora_training False \
            --output_dir ${output_dir}${version[i]} \
            --overwrite_output_dir \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 2 \
            --lr_scheduler_type cosine \
            --logging_steps 10 \
            --save_steps 1000 \
            --learning_rate 1e-5 \
            --num_train_epochs 3.0 \
            --plot_loss \
            --fp16 \
            --ulma_sft_loss_type "${sft_loss_type[i]}" && continue
    done
fi

# Pairwise dpo
if [ $step -le 2 ]; then
    deepspeed --num_gpus ${num_gpus} --master_port=9902 src/train_bash.py \
        --stage dpo \
        --model_name_or_path ${base_model_path} \
        --do_train True \
        --dataset "${pairwise_dataset}" \
        --template default \
        --finetuning_type lora \
        --lora_target q_proj,v_proj \
        --resume_lora_training False \
        --output_dir ${output_dir}${model}_pairwise_dpo \
        --overwrite_output_dir \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 1000 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --plot_loss \
        --fp16 && continue
fi

# Unlikelihood
if [ $step -le 3 ]; then
  deepspeed --num_gpus ${num_gpus} --master_port=9902 src/train_bash.py \
      --stage unlikelihood \
      --model_name_or_path ${base_model_path} \
      --do_train \
      --dataset ${pointwise_dataset} \
      --template default \
      --finetuning_type lora \
      --lora_target q_proj,v_proj \
      --resume_lora_training False \
      --output_dir ${output_dir}${model}_unlikelihood \
      --overwrite_output_dir \
      --overwrite_cache \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --lr_scheduler_type cosine \
      --logging_steps 10 \
      --save_steps 1000 \
      --learning_rate 1e-5 \
      --num_train_epochs 3.0 \
      --plot_loss \
      --unlikelihood_sft_loss_type logloss \
      --fp16
fi

# SFT
if [ $step -le 4 ]; then
    deepspeed --num_gpus ${num_gpus} --master_port=9902 src/train_bash.py \
        --stage sft \
        --model_name_or_path ${base_model_path} \
        --do_train True \
        --dataset "${sft_dataset_rlhf}" \
        --template default \
        --finetuning_type lora \
        --lora_target q_proj,v_proj \
        --resume_lora_training False \
        --output_dir ${output_dir}${model}_sft \
        --overwrite_output_dir \
        --overwrite_cache \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 1000 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --plot_loss \
        --fp16 && continue
fi

# RM
if [ $step -le 5 ]; then
      deepspeed --num_gpus ${num_gpus} --master_port=9902 src/train_bash.py \
        --stage rm \
        --model_name_or_path ${base_model_path} \
        --do_train \
        --dataset "${pairwise_dataset}" \
        --template default \
        --finetuning_type lora \
        --lora_target q_proj,v_proj \
        --resume_lora_training False \
        --output_dir ${output_dir}${model}_rm \
        --overwrite_output_dir \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 1000 \
        --learning_rate 1e-6 \
        --num_train_epochs 3.0 \
        --plot_loss \
        --fp16 && continue
fi

# Somehow the deepspeed run into an error for ppo training. We fixed it using accelerate.
# PPO
if [ $step -le 6 ]; then
    accelerate launch \
        --config_file accelerate_default_config.yaml \
        --num_processes ${num_gpus} \
        --main_process_port 22331 \
        src/train_bash.py \
        --stage ppo \
        --model_name_or_path ${base_model_path} \
        --do_train True\
        --dataset "${sft_dataset_rlhf}"\
        --template default \
        --finetuning_type lora \
        --lora_target q_proj,v_proj \
        --resume_lora_training False \
        --reward_model ${output_dir}${model}_rm  \
        --output_dir ${output_dir}${model}_rlhf \
        --overwrite_output_dir \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 1000 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --plot_loss \
        --fp16 && continue
fi

# Inference
if [ $step -le 7 ]; then
    for suffix in (_ulma _pointwise_dpo _pairwise_dpo _unlikelihood _sft _rlhf)
    do
        deepspeed --num_gpus ${num_gpus} --master_port=9902 src/train_bash.py \
            --stage sft \
            --model_name_or_path ${base_model_path} \
            --do_predict \
            --dataset ${sft_dataset_rlhf} \
            --template default \
            --finetuning_type lora \
            --checkpoint_dir ${output_dir}${model}${suffix} \
            --output_dir ${infer_dir}${model}${suffix} \
            --per_device_eval_batch_size 8 \
            --max_samples ${max_infer_sample} \
            --predict_with_generate \
            --split test \
            --do_sample False && continue
    done
fi
