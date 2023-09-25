# Unified-Language-Model-Alignment 
This is the code for the paper **Unified Language Model Alignment with Demonstration and Point-wise Human Preference** (under review). It is developed based on [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) and our methods are tested on 8 A100 80G GPUs with Vicuna v1.5 7B model (see our paper detailed settings), but it should work for various settings.

## The Alignment Approaches
This is the table for all implemented methods, with the corresponding dataset to use.

| Approach               |    SFT dataset     |  Pairwise dataset  |  Pointwise dataset |
| ---------------------- | ------------------ | ------------------ | ------------------ |
| Supervised Fine-Tuning | :white_check_mark: |                    |                    |
| DPO (Pairwise)         |                    | :white_check_mark: |                    |
| Reward Modeling        |                    | :white_check_mark: |                    |
| PPO                    | :white_check_mark: |                    |                    |
| Pointwise DPO          |                    |                    | :white_check_mark: |
| Unlikelihood           |                    |                    | :white_check_mark: |
| ULMA                   |                    |                    | :white_check_mark: |

## Provided Datasets

All the methods are tested on Anthropic Helpful and Harmless (HH) datasets as well as the [HH golden](https://huggingface.co/datasets/Unified-Language-Model-Alignment/Anthropic_HH_Golden) dataset. The HH Golden is improved on the original harmless dataset by using  GPT4 to rewrite the original "chosen" answer. Compared with the original Harmless dataset, empirically this dataset improves the performance of SFT, RLHF, DPO or ULMA methods significantly on harmless metrics (see our paper for details).

Change **pairwise_dataset** in run_exp.sh from **hh_rlhf_en** to **hh_rlhf_en_golden_rejected** will switch between the HH and HH golden datasets (also affects the SFT and pointwise datasets). As mentioned in our paper, given a pairwise dataset, the corresponding SFT dataset is constructed using the chosen samples in the pairwise dataset, and the pointwise dataset are constructed using the chosen samples with score=1 and rejected samples with score=0.

## Getting Started

```bash
https://github.com/Unified-Language-Model-Alignment/src.git
cd src
pip install -r requirements.txt
sh run_exp.sh
```

The run_exp.sh will run the aformentioned 7 alignment algorithms sequentially, after which it will run inference using the test split of the SFT dataset. It is default tested on the original HH dataset. The users need to config the path to base model, and the output diretory for the finetuned models and the inference.

To customize configurations, users can changes the following settings in run_exp.sh.

```bash
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
```