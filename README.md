## Overview
This is an algorithm re-implemented in PyTorch, which aims to combine model-based reinforcement learning (MBRL) with normalizing flows for the first time to explore their applications and performance in deep reinforcement learning: 
[When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253). [A Coupled Flow Approach to Imitation Learning](https://arxiv.org/pdf/2305.00303)

This code is based on a [previous paper in the NeurIPS reproducibility challenge](https://openreview.net/forum?id=rkezvT9f6r) that reproduces the result with a tensorflow ensemble model but shows a significant drop in performance with a pytorch ensemble model. 
This code re-implements the ensemble dynamics model with pytorch and closes the gap. 






## Dependencies

MuJoCo 1.5 & MuJoCo 2.0

## Usage
> python main_mbpo.py --env_name ${env_name}'-v2' --num_epoch=${num_e} --use_algo 'discriminator'

if you want to quickly reverso context:
> sh run.sh

## Reference
* Official tensorflow implementation: https://github.com/JannerM/mbpo
* Code to the reproducibility challenge paper: https://github.com/jxu43/replication-mbpo
