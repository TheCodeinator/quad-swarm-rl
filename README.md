# Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning

A codebase for training reinforcement learning policies for quadrotor swarms.
Includes:
* Flight dynamics simulator forked from https://github.com/amolchanov86/gym_art
and extended to support swarms of quadrotor drones
* Scripts and the necessary wrappers to facilitate training of control policies with Sample Factory
https://github.com/alex-petrenko/sample-factory

**Paper:** https://arxiv.org/abs/2109.07735

**Website:** https://sites.google.com/view/swarm-rl

## Installation

Initialize a Python environment, i.e. with `conda` (Python versions 3.6-3.8 are supported):

```
conda create -n swarm-rl python=3.8
conda activate swarm-rl
```

Clone and install this repo as an editable Pip package:
```
git clone https://github.com/Zhehui-Huang/quad-swarm-rl.git
cd quad-swarm-rl
pip install -e .
```

This should pull and install all the necessary dependencies including PyTorch.

## Running experiments

### Train

This will run the baseline experiment.
Change the number of workers appropriately to match the number of logical CPU cores on your machine, but it is advised that
the total number of simulated environments is close to that in the original command:

We provide a training script `train_local.sh`, so you can simply start training by command `bash train_local.sh`.

To monitor the experiments, go to the experiment folder, and run the following command:

```
tensorboard --logdir=./
```
### WandB support

If you want to monitor training with WandB, follow the steps below: 
- setup WandB locally by running `wandb login` in the terminal (https://docs.wandb.ai/quickstart#1.-set-up-wandb).
* add `--with_wandb=True` in the command.

Here is a total list of wandb settings: 
```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_project: WandB "Project" (default: sample_factory)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

### Test
To test the trained model, run the following command:

```
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME --quads_view_mode CAMERA_VIEWS
```
EXPERIMENT_NAME and PATH_TO_TRAIN_DIR can be found in the cfg.json file of your trained model

CAMERA_VIEWS can be any number of views from the following: `[topdown, global, chase, side, corner0, corner1, corner2, corner3, topdownfollow]`

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our CORL paper.

```
@inproceedings{batra21corl,
  author    = {Sumeet Batra and
               Zhehui Huang and
               Aleksei Petrenko and
               Tushar Kumar and
               Artem Molchanov and
               Gaurav S. Sukhatme},
  title     = {Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning},
  booktitle = {5th Conference on Robot Learning, CoRL 2021, 8-11 November 2021, London, England, {UK}},
  series    = {Proceedings of Machine Learning Research},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.07735}
}
```
