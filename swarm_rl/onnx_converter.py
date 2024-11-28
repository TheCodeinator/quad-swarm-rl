"""

Export trained model in ONNX (Open Neural Network eXchange) format

"""

import types
import json
import onnx
import torch
import torch.nn as nn
import onnxsim
import netron

from typing import List
from pathlib import Path
from attrdict import AttrDict
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.algo.utils.env_info import EnvInfo, extract_env_info
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.learning.learner import Learner
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.cfg.arguments import load_from_checkpoint
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.train import register_swarm_components


class RnnWrapper(nn.Module):
    policy: ActorCritic
    cfg: Config
    env_info: EnvInfo

    def __init__(self, cfg: Config, env_info: EnvInfo, actor_critic: ActorCritic):
        super().__init__()
        self.cfg = cfg
        self.env_info = env_info
        self.policy = actor_critic

    def forward(self, **obs):
        rnn = obs.pop("rnn_states")
        normalized_obs = prepare_and_normalize_obs(self.policy, obs)
        rnn = self.policy(normalized_obs, rnn, sample_actions=False)["new_rnn_states"]
        action_distribution = self.policy.action_distribution()
        actions = argmax_actions(action_distribution)
        return actions, rnn


class Wrapper(nn.Module):
    """
    Pass forward expected dummy rnn states for non rnn actor critics
    """
    policy: ActorCritic
    cfg: Config
    env_info: EnvInfo

    def __init__(self, cfg: Config, env_info: EnvInfo, actor_critic: ActorCritic):
        super().__init__()
        self.cfg = cfg
        self.env_info = env_info
        self.policy = actor_critic

    def forward(self, **obs):
        normalized_obs = prepare_and_normalize_obs(self.policy, obs)
        _ = self.policy(normalized_obs, sample_actions=False)
        action_distribution = self.policy.action_distribution()
        actions = argmax_actions(action_distribution)
        return actions


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

    # I could add a command parser here
    name = "neuralfly_no_rnn_sqz9"
    visualize = True

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

    torch.jit._state.disable()
    model = create_actor_critic(cfg, env.observation_space, env.action_space)
    model.eval()
    load_state_dict(cfg, model, torch.device("cpu"))
    torch.jit._state.enable()

    if cfg.use_rnn:
        model = RnnWrapper(cfg, extract_env_info(env, cfg), model)
    else:
        model = Wrapper(cfg, extract_env_info(env, cfg), model)

    input_names = ['obs']
    output_names = ["thrust_out"]

    m_arguments = {'obs': torch.rand(1, 48, dtype=torch.float32)}

    if cfg.use_rnn:
        input_names.append("rnn_states")
        output_names.append("rnn_out")
        m_arguments["rnn_states"] = torch.zeros([1, cfg.rnn_size], dtype=torch.float32)

    # if input width is not fixed
    # dynamic_axes = {key: {0: "batch_size"} for key in input_names + output_names}

    patch_forward(model, input_names)

    Path("../artifacts").mkdir(exist_ok=True)
    fn = f"../artifacts/{name}.onnx"

    torch.onnx.export(model, (m_arguments,), fn,
                      input_names=input_names,
                      output_names=output_names,
                      # dynamic_axes=dynamic_axes,
                      do_constant_folding=True,
                      opset_version=13,
                      keep_initializers_as_inputs=False,
                      )

    m_in = onnx.load(fn)

    m_out, check = onnxsim.simplify(m_in,
                                    5,
                                    skip_shape_inference=False,
                                    )

    if check:
        onnx.save(m_out, fn)
    else:
        raise AssertionError("Model optimization failed")

    if visualize:
        netron.start(fn, 10002, False)
        netron.wait()

    return 0


if __name__ == '__main__':
    main()
