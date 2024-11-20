"""

Export trained model in ONNX (Open Neural Network eXchange) format

"""

import types
import json
import torch
import torch.nn as nn
import gymnasium as gym

from typing import List
from pathlib import Path
from attrdict import AttrDict
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.algo.utils.env_info import EnvInfo, extract_env_info
from sample_factory.algo.learning.learner import Learner
from sample_factory.utils.typing import Config
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.cfg.arguments import load_from_checkpoint
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.train import register_swarm_components


class Wrapper(nn.Module):
    actor_critic: ActorCritic
    cfg: Config
    env_info: EnvInfo

    def __init__(self, cfg: Config, env_info: EnvInfo, actor_critic: ActorCritic):
        super().__init__()
        self.cfg = cfg
        self.env_info = env_info
        self.actor_critic = actor_critic

    def forward(self, **obs):

        rnn_states = torch.zeros([1, self.cfg.rnn_size], dtype=torch.float32)
        action_mask = obs.pop("action_mask", None)
        normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs)
        policy_outputs = self.actor_critic(normalized_obs, rnn_states)
        actions = policy_outputs["actions"]
        rnn_states = policy_outputs["new_rnn_states"]
        #if self.cfg.eval_deterministic:
        #    action_distribution = self.actor_critic.action_distribution()
        #    actions = argmax_actions(action_distribution)
        # if actions.ndim == 1:
        #    actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(self.env_info, actions)

        return torch.from_numpy(actions).unsqueeze(1)


def generate_args(space: gym.spaces.Space, batch_size: int = 1):
    args = [unsqueeze_args(sample_space(space)) for _ in range(batch_size)]
    args = [a for a in args if isinstance(a, dict)]
    args = {k: torch.cat(tuple(a[k] for a in args), dim=0) for k in args[0].keys()} if len(args) > 0 else {}
    return args


def sample_space(space: gym.spaces.Space):
    if isinstance(space, gym.spaces.Discrete):
        return int(space.sample())
    elif isinstance(space, gym.spaces.Box):
        return torch.from_numpy(space.sample())
    elif isinstance(space, gym.spaces.Dict):
        return {k: sample_space(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(sample_space(s) for s in space.spaces)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def unsqueeze_args(args):
    if isinstance(args, int):
        return torch.tensor(args).unsqueeze(0)
    if isinstance(args, torch.Tensor):
        return args.unsqueeze(0)
    if isinstance(args, dict):
        return {k: unsqueeze_args(v) for k, v in args.items()}
    elif isinstance(args, tuple):
        return (unsqueeze_args(v) for v in args)
    else:
        raise NotImplementedError(f"Unsupported args type: {type(args)}")


def create_forward(original_forward, arg_names: List[str]):
    args_str = ", ".join(arg_names)
    func_code = f"""
def forward(self, {args_str}):
    bound_args = locals()
    bound_args.pop('self')
    return original_forward(**bound_args)
    """
    globals_vars = {"original_forward": original_forward}
    local_vars = {}
    exec(func_code, globals_vars, local_vars)
    return local_vars["forward"]


def patch_forward(model: nn.Module, input_names: List[str]):
    """
    Patch the forward method of the model to dynamically define the input arguments
    since *args and **kwargs are not supported in `torch.onnx.export`
    see also: https://github.com/pytorch/pytorch/issues/96981 and https://github.com/pytorch/pytorch/issues/110439
    """
    forward = create_forward(model.forward, input_names)
    model.forward = types.MethodType(forward, model)


def load_state_dict(cfg: Config, actor_critic: ActorCritic, device: torch.device) -> None:
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, 0), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    if checkpoint_dict:
        actor_critic.load_state_dict(checkpoint_dict["model"])
    else:
        raise RuntimeError("Could not load checkpoint")


def main():

    name = "train_multi_drone_real_256_256_full_encoder_128_mean_embed_smaller_env_lower_lr"

    model_dir = Path(f"../train_dir/{name}/")
    assert model_dir.exists(), f'Path {str(model_dir)} is not a valid path'
    # Load hyper-parameters
    cfg_path = model_dir.joinpath('config.json')
    with open(cfg_path, 'r') as f:
        args = json.load(f)
    args = AttrDict(args)
    cfg = load_from_checkpoint(args)

    register_swarm_components()
    env = make_quadrotor_env_multi(cfg)
    model = create_actor_critic(cfg, env.observation_space, env.action_space)
    model.eval()
    load_state_dict(cfg, model, torch.device("cpu"))

    wrapped_model = Wrapper(cfg, extract_env_info(env, cfg), model)

    m_arguments = generate_args(env.observation_space)

    input_names = list(m_arguments.keys())
    output_names = ["output_actions"]

    dynamic_axes = {key: {0: "batch_size"} for key in input_names + output_names}

    patch_forward(wrapped_model, input_names)

    torch.onnx.export(wrapped_model, (m_arguments,), f"{name}.onnx",
                      export_params=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

    return 0


if __name__ == '__main__':
    main()
