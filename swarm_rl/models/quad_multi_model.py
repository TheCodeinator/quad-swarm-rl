from typing import List

import torch
import numpy as np
from sample_factory.model.core import ModelCore

from torch import nn

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.algo.utils.action_distributions import ContinuousActionDistribution
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.model.action_parameterization import ActionsParameterization
from sample_factory.model.model_utils import fc_layer, nonlinearity, create_mlp

from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE, QUADS_OBSTACLE_OBS_TYPE

from swarm_rl.models.attention_layer import MultiHeadAttention


class QuadNeighborhoodEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs):
        super().__init__()
        self.cfg = cfg
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs


class QuadNeighborhoodEncoderDeepsets(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.embedding_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg)
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
        neighbor_embeds = self.embedding_mlp(obs_neighbors)
        neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, self.neighbor_hidden_size)
        mean_embed = torch.mean(neighbor_embeds, dim=1)
        return mean_embed


class QuadNeighborhoodEncoderAttention(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        # outputs e_i from the paper
        self.embedding_mlp = nn.Sequential(
            fc_layer(self_obs_dim + neighbor_obs_dim, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg)
        )

        #  outputs h_i from the paper
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

        # outputs scalar score alpha_i for each neighbor i
        self.attention_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size * 2, neighbor_hidden_size),
            # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, 1),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation

        self_obs_repeat = self_obs.repeat(self.num_use_neighbor_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)
        neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(self.num_use_neighbor_obs, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.neighbor_hidden_size)
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1)

        return final_neighborhood_embedding


class QuadNeighborhoodEncoderMlp(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        self.neighbor_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim * num_use_neighbor_obs, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        final_neighborhood_embedding = self.neighbor_mlp(obs_neighbors)
        return final_neighborhood_embedding


class QuadMultiHeadAttentionEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        # Internal params
        if cfg.quads_obs_repr in QUADS_OBS_REPR:
            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        else:
            raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')

        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        self.use_obstacles = cfg.quads_use_obstacles

        if cfg.quads_neighbor_visible_num == -1:
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        else:
            self.num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_use_neighbor_obs

        # Embedding Layer
        fc_encoder_layer = cfg.rnn_size
        self.self_embed_layer = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self.neighbor_embed_layer = nn.Sequential(
            fc_layer(self.all_neighbor_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
        self.obstacle_embed_layer = nn.Sequential(
            fc_layer(self.obstacle_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )

        # Attention Layer
        self.attention_layer = MultiHeadAttention(4, cfg.rnn_size, cfg.rnn_size, cfg.rnn_size)

        # MLP Layer
        self.encoder_output_size = 2 * cfg.rnn_size
        self.feed_forward = nn.Sequential(fc_layer(3 * cfg.rnn_size, self.encoder_output_size),
                                          nonlinearity(cfg))

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        batch_size = obs.shape[0]
        obs_self = obs[:, :self.self_obs_dim]
        obs_neighbor = obs[:, self.self_obs_dim: self.self_obs_dim + self.all_neighbor_obs_dim]
        obs_obstacle = obs[:, self.self_obs_dim + self.all_neighbor_obs_dim:]

        self_embed = self.self_embed_layer(obs_self)
        neighbor_embed = self.neighbor_embed_layer(obs_neighbor)
        obstacle_embed = self.obstacle_embed_layer(obs_obstacle)
        neighbor_embed = neighbor_embed.view(batch_size, 1, -1)
        obstacle_embed = obstacle_embed.view(batch_size, 1, -1)
        attn_embed = torch.cat((neighbor_embed, obstacle_embed), dim=1)

        attn_embed, attn_score = self.attention_layer(attn_embed, attn_embed, attn_embed)
        attn_embed = attn_embed.view(batch_size, -1)

        embeddings = torch.cat((self_embed, attn_embed), dim=1)
        out = self.feed_forward(embeddings)

        return out

    def get_out_size(self):
        return self.encoder_output_size


class QuadMultiEncoder(Encoder):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        self.use_obstacles = cfg.quads_use_obstacles

        # Neighbor
        neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        if cfg.quads_neighbor_obs_type == 'none':
            num_use_neighbor_obs = 0
        else:
            if cfg.quads_neighbor_visible_num == -1:
                num_use_neighbor_obs = cfg.quads_num_agents - 1
            else:
                num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.all_neighbor_obs_size = neighbor_obs_dim * num_use_neighbor_obs

        # # Neighbor Encoder
        self.neighbor_encoder = None

        if num_use_neighbor_obs > 0:
            neighbor_encoder_type = cfg.quads_neighbor_encoder_type
            if neighbor_encoder_type == 'mean_embed':
                self.neighbor_encoder = QuadNeighborhoodEncoderDeepsets(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'attention':
                self.neighbor_encoder = QuadNeighborhoodEncoderAttention(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'mlp':
                self.neighbor_encoder = QuadNeighborhoodEncoderMlp(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'no_encoder':
                # Blind agent
                self.neighbor_encoder = None
            else:
                raise NotImplementedError

        neighbor_encoder_out_size = neighbor_hidden_size

        fc_encoder_layer = cfg.rnn_size
        # Encode Self Obs
        self.self_encoder = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self_encoder_out_size = calc_num_elements(self.self_encoder, (self.self_obs_dim,))

        # Encode Obstacle Obs
        obstacle_encoder_out_size = 0
        if self.use_obstacles:
            obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
            obstacle_hidden_size = cfg.quads_obst_hidden_size
            self.obstacle_encoder = nn.Sequential(
                fc_layer(obstacle_obs_dim, obstacle_hidden_size),
                nonlinearity(cfg),
                fc_layer(obstacle_hidden_size, obstacle_hidden_size),
                nonlinearity(cfg),
            )
            obstacle_encoder_out_size = calc_num_elements(self.obstacle_encoder, (obstacle_obs_dim,))

        total_encoder_out_size = self_encoder_out_size + neighbor_encoder_out_size + obstacle_encoder_out_size

        # This is followed by another fully connected layer in the action parameterization, so we add a nonlinearity
        # here
        self.feed_forward = nn.Sequential(
            fc_layer(total_encoder_out_size, 2 * cfg.rnn_size),
            nonlinearity(cfg),
        )
        self.encoder_out_size = calc_num_elements(self.feed_forward, (total_encoder_out_size,))

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        obs_self = obs[:, :self.self_obs_dim]
        batch_size = obs_self.shape[0]
        squeeze_dim = 0 if batch_size == 1 else 1
        self_embeddings = self.self_encoder(obs_self)
        cat_input_self = self_embeddings.squeeze(0).clone()
        neighborhood_embedding = self.neighbor_encoder(obs_self, obs, self.all_neighbor_obs_size, batch_size)
        cat_input_nb = neighborhood_embedding.squeeze(0).clone()
        embeddings = torch.cat((cat_input_self, cat_input_nb), dim=squeeze_dim)

        if self.use_obstacles:
            obs_obstacles = obs[:, self.self_obs_dim + self.all_neighbor_obs_size:]
            obstacle_embeds = self.obstacle_encoder(obs_obstacles).squeeze(0).clone()
            embeddings = torch.cat((embeddings, obstacle_embeds), dim=squeeze_dim)

        embeddings = embeddings.reshape(batch_size, self.encoder_out_size)

        return self.feed_forward(embeddings)

    def get_out_size(self) -> int:
        return self.encoder_out_size


class QuadMultiCore(ModelCore):

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    @staticmethod
    def forward(head_outputs, rnn_states):
        return head_outputs, rnn_states


class QuadMultiDecoder(Decoder):

    def __init__(self, cfg, decoder_input_size):
        super().__init__(cfg)
        decoder_layers: List[int] = cfg.decoder_mlp_layers
        activation = nonlinearity(cfg)
        self.mlp = create_mlp(decoder_layers, decoder_input_size, activation)
        self.decoder_out_size = calc_num_elements(self.mlp, (decoder_input_size,))

    def forward(self, core_output):
        return self.mlp(core_output)

    def get_out_size(self) -> int:
        return self.decoder_out_size


class QuadActionParametrization(ActionsParameterization):

    def __init__(self, cfg, core_out_size, action_space):
        super().__init__(cfg, action_space)

        num_action_outputs = np.prod(action_space.shape) * 2
        self.distribution_linear = nn.Linear(core_out_size, num_action_outputs)

    def forward(self, actor_core_output):
        """Just forward the FC layer and generate the distribution object."""
        action_distribution_params = self.distribution_linear(actor_core_output)
        action_distribution = ContinuousActionDistribution(params=action_distribution_params)
        return action_distribution_params, action_distribution


class QuadMultiActorCritic(ActorCriticSharedWeights):

    def __init__(self,
                 model_factory,
                 obs_space,
                 action_space,
                 cfg):
        super().__init__(model_factory, obs_space, action_space, cfg)

        self.action_parameterization = QuadActionParametrization(cfg, self.decoder.get_out_size(), action_space)

    def forward(self, normalized_obs_dict, rnn_states=None, values_only=False, sample_actions=True) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, _ = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions)
        result["new_rnn_states"] = rnn_states
        return result


class QuadMultiActorCriticRNN(ActorCriticSharedWeights):
    def __init__(self,
                 model_factory,
                 obs_space,
                 action_space,
                 cfg):
        super().__init__(model_factory, obs_space, action_space, cfg)

        self.action_parameterization = QuadActionParametrization(cfg, self.decoder.get_out_size(), action_space)

    def forward(self, normalized_obs_dict, rnn_states=None, values_only=False, sample_actions=True) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions)
        result["new_rnn_states"] = new_rnn_states
        return result


def make_quadmulti_encoder(cfg, obs_space) -> Encoder:
    return QuadMultiEncoder(cfg, obs_space)


def make_quadmulti_core(cfg, core_input_size: int) -> ModelCore:
    return QuadMultiCore(cfg, core_input_size)


def make_quadmulti_decoder(cfg, core_size) -> Decoder:
    return QuadMultiDecoder(cfg, core_size)


def make_quadmulti_ac(cfg, obs_space, action_space) -> ActorCriticSharedWeights:
    model_factory = global_model_factory()
    if cfg.use_rnn:
        return QuadMultiActorCriticRNN(model_factory, obs_space, action_space, cfg)
    else:
        model_factory.register_model_core_factory(make_quadmulti_core)
        return QuadMultiActorCritic(model_factory, obs_space, action_space, cfg)


def register_models():
    global_model_factory().register_actor_critic_factory(make_quadmulti_ac)
    global_model_factory().register_encoder_factory(make_quadmulti_encoder)
    global_model_factory().register_decoder_factory(make_quadmulti_decoder)
