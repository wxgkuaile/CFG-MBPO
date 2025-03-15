## Overview
Model-based reinforcement learning (MBRL) techniques are celebrated for their superior sample efficiency. However, inaccuracies in the model's multi-step predictions can hinder both the long-term performance and the overall sample efficiency. To address this challenge, we introduce a novel method called **Coupled Flows-Guided Model-Based Policy Optimization**. This approach leverages **coupled flows**—representations that capture both the true state-action distribution and the distribution generated by the learned dynamics model and policy. By optimizing these flows to minimize their divergence, we derive a loss function that serves two crucial purposes. 

First, this loss function is used to develop a discriminator, which identifies the most accurate simulated rollouts for effective policy learning. Second, it acts as a reward signal within the Markov Decision Process (MDP) framework governing the rollout process. This dual function allows the dynamics model to be iteratively refined, minimizing multi-step prediction errors through reinforcement learning techniques. Notably, we theoretically establish that the error in the cumulative expected return between the real environment and the dynamics model is bounded from above, ensuring robustness in policy optimization.

In summary, our method improves MBRL by using coupled flows to guide model learning and policy optimization, ultimately enhancing both sample efficiency and asymptotic performance.


## Overview of the repository structure

cfg_mbpo.py：program executes the main script.

predict_env.py: scripts for implementing dynamic model launch and trajectory planning

model.py: dynamic model (ensemble dynamics) construction and training module.

flow/: implements coupled flows and obtains coupled flow difference modules.

sac/: implements policy update and CFG-RL parts, mainly using SAC algorithm.

utils/: common tool functions and auxiliary modules, such as data preprocessing, logging, etc.

env/: partially truncated environment templates for convenient experiments.

requirements.txt: lists all required dependencies (such as Python version, PyTorch, TensorFlow, NumPy, etc.). Users only need to run pip install -r requirements.txt to install all dependencies.

## Dependencies

MuJoCo 1.5 & MuJoCo 2.0

## Usage

#1.Clone the repository

>git clone https://github.com/wxgkuaile/CFG-MBPO.git

>cd CFG-MBPO

#2.Install Dependencies

>pip install -r requirements.txt

#3.Run the training script

> python cfg_mbpo.py --env_name ${env_name}'-v2' --num_epoch=${num_e} --use_algo 'discriminator'

If you want to quickly execute multiple seeds in parallel, please deploy in the background：

> sh run.sh

## References
* When to Trust Your Model: Model-Based Policy Optimization (https://arxiv.org/pdf/1906.08253)
* A Coupled Flow Approach to Imitation Learning (https://arxiv.org/pdf/2305.00303)
* Plan To Predict: Learning an Uncertainty-Foreseeing Model for Model-Based Reinforcement Learning (https://arxiv.org/pdf/2301.08502)
