# What Makes and Breaks Safety Fine-tuning? A Mechanistic Study

[![arXiv](https://img.shields.io/badge/arXiv-2405.20459-b31b1b.svg)](https://arxiv.org/abs/2407.10264)

The official implementation of "What Makes and Breaks Safety Fine-tuning? A Mechanistic Study". This work is accepted to NeurIPS 2024.

> [**What Makes and Breaks Safety Fine-tuning? A Mechanistic Study**](https://arxiv.org/abs/2407.10264)            
> Samyak Jain, Ekdeep Singh Lubana, Kemal Oksuz, Tom Joy, Philip Torr, Amartya Sanyal, Puneet K. Dokania

![alt text](activation_space_analysis.png)

## Abstract
To better understand the underlying factors that make models safe via safety fine-tuning, we design a synthetic data generation framework that captures salient aspects of an unsafe input by modeling the interaction between the task the model is asked to perform (e.g., “design”) versus the specific concepts the task is asked to be performed upon (e.g., a “cycle” vs. a “bomb”). Using this, we investigate three well-known safety fine-tuning methods—supervised safety fine-tuning, direct preference optimization, and unlearning—and provide significant evidence demonstrating that these methods minimally transform MLP weights to specifically align unsafe inputs into its weights’ null space. This yields a clustering of inputs based on whether the model deems them safe or not. Correspondingly, when an adversarial input (e.g., a jailbreak) is provided, its activations are closer to safer samples, leading to the model processing such an input as if it were safe.

## Usage
To install the required libraries run 

```
pip install -r requirements.txt
```

### Making the dataset

To make the datasets used for pre-training run

```
python ./data_generator/make_data_pretrain.py --sample_pcfg_number 1--min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data/pretrain_train_data_pcfg1.pkl' --start_random 0  --test_data_path './saved_data/pretrain_test_data_pcfg1.pkl'
```

```
python ./data_generator/make_data_safety_finetune_pcfg1.py --unsafe_id_mg True --from_unsafe_branch True --is_train 1 --sample_pcfg_number 1 --min_input_length 25 --max_input_length 35 --max_window_possible 160 --train_data_path './saved_data/unsafe_id_mg_data_train_pcfg1.pkl' --start_random 0  --test_data_path './saved_data/unsafe_id_mg_data_test_pcfg1.pkl'
```
Similarly generate the datasets for other three PCFGs.

### Pre-training and instruction fine-tuning


### Safety fine-tuning


### Evaluations



## How to Cite

Please cite the paper if you benefit from our paper or the repository:

```
@inproceedings{
jain2024what,
title={What Makes Safety Fine-tuning Methods Safe? A Mechanistic Study},
author={Anonymous},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=JEflV4nRlH}
}

```