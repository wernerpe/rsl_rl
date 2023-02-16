# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import OneHotCategorical, TransformedDistribution, AffineTransform
from torch.nn.modules import rnn
import torch.nn.functional as F

from rsl_rl.utils import TruncatedNormal, SquashedNormal

# from rsl_rl.modules.attention.encoders import EncoderAttention4


class Actor(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        ###PUT CODE HERE
        
        pass

class StrucutredCritic(nn.Module):
    def __init__(self,):
        ###PUT CODE HERE
        #look for CMAAC class for available params such as input output dimensios etc
        pass


class BilevelActorCriticAttention(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_add_obs,
                        num_actions,
                        # num_agents,
                        device,
                        encoder,
                        train_encoder=False,
                        act_min=None,
                        act_max=None,
                        act_ini=None,
                        discrete=False,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        critic_output_dim=1,
                        std_per_obs=True,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(BilevelActorCriticAttention, self).__init__()

        activation = get_activation(activation)

        encoder_type = kwargs['encoder_type']
        encoder_hidden_dims = kwargs['encoder_hidden_dims']

        # teamsize = kwargs['teamsize']
        # numteams = kwargs['numteams']

        num_ego_obs = kwargs['num_ego_obs']
        num_ado_obs = kwargs['num_ado_obs']

        self.n_critics = kwargs['numcritics']
        self.is_attentive = kwargs['attentive']

        # num_agent_max = num_agents
        num_ego_obs = num_ego_obs
        num_ado_obs = num_ado_obs
        if encoder_type=='identity':
            mlp_input_dim_a = encoder_hidden_dims[-1]
            mlp_input_dim_c = encoder_hidden_dims[-1]
        elif encoder_type=='attention4':
            mlp_input_dim_a = num_ego_obs + encoder_hidden_dims[-1]
            mlp_input_dim_c = num_ego_obs + encoder_hidden_dims[-1]
        else:
            mlp_input_dim_a = num_ego_obs + 1*num_ado_obs
            mlp_input_dim_c = num_ego_obs + 1*num_ado_obs

        # mlp_input_dim_a = num_ego_obs + 3*num_ado_obs  # FIXME: testing
        # mlp_input_dim_c = num_ego_obs + 3*num_ado_obs  # FIXME: testing

        mlp_input_dim_a += num_add_obs
        mlp_input_dim_c += num_add_obs

        enc_split_dim = num_actor_obs

        self.num_actions = num_actions
        self.mlp_output_dim_a = num_actions

        self.std_per_obs = std_per_obs
        self.std_ini = init_noise_std
        self.std_min = 3.e-2  # 1.e-2

        # # Encoder
        # self.encoder = encoder

        self.use_discrete_policy = discrete
        self.use_output_mapping = (act_min and act_max and act_ini) is not None
        if self.use_discrete_policy:
            assert self.use_output_mapping

            # Discrete actor
            self.num_bins = 5  # 5
            self._trafo_scale = torch.tensor(np.linspace(start=act_min, stop=act_max, num=self.num_bins, axis=-1), dtype=torch.float, device=device)
            self._trafo_delta = self._trafo_scale[:, 1] - self._trafo_scale[:, 0]
            self._trafo_loc = 0.0 * torch.tensor(act_min, dtype=torch.float, device=device)
            self.mlp_output_dim_a = num_actions * self.num_bins
        else:
            if self.use_output_mapping:
                self._mean_target_pos_min = nn.Parameter(torch.tensor(act_min[:2]), requires_grad=False)
                self._mean_target_pos_max = nn.Parameter(torch.tensor(act_max[:2]), requires_grad=False)
                self._mean_target_pos_off = nn.Parameter(torch.tensor([3.5, 0.0]), requires_grad=False)

                self._mean_target_std_min = nn.Parameter(torch.tensor(act_min[2:]), requires_grad=False)
                self._mean_target_std_max = nn.Parameter(torch.tensor(act_max[2:]), requires_grad=False)
                self._mean_target_std_ini = nn.Parameter(torch.tensor(act_ini[2:]), requires_grad=False)
                self._softplus = nn.Softplus()

            if self.std_per_obs:
                self.mlp_output_dim_a *= 2
        
        # Policy
        self.actor = ActorAttention(
            input_dim=mlp_input_dim_a, 
            hidden_dims=actor_hidden_dims, 
            split_dim=enc_split_dim,
            output_dim=self.mlp_output_dim_a, 
            activation=activation,
            encoder=encoder,
            train_encoder=False,
        )
        # Value function
        self.critics = nn.ModuleList([CriticAttention(
          input_dim=mlp_input_dim_c, 
          hidden_dims=critic_hidden_dims, 
          split_dim=enc_split_dim,
          output_dims=critic_output_dim,
          activation=activation,
          encoder=encoder,
          train_encoder=train_encoder,
        ) for _ in range(self.n_critics)])

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critics[0]}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        if self.use_discrete_policy:
            # One-hot Categorical --> weight each option by prob
            return (self._trafo_scale * self.distribution.probs).sum(dim=-1)
        else:
            # Gaussian
            return self.distribution.mean

    @property
    def action_std(self):
        if self.use_discrete_policy:
            # One-hot Categorical --> weight each option by prob
            return torch.sqrt(((self._trafo_scale - self.action_mean.unsqueeze(dim=-1))**2 * self.distribution.probs).sum(dim=-1))
        else:
            # Gaussian
            return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
        # return self.distribution.entropy.sum(dim=-1)

    def transform_mean_prediction(self, mean_raw):

        if self.use_output_mapping:
            mean_target_pos = self._mean_target_pos_min + (self._mean_target_pos_max - self._mean_target_pos_min) * 0.5 * (torch.tanh(mean_raw[:, 0:2]) + 1.0)
            # mean_target_pos = mean_target_pos + self._mean_target_pos_off

            # mean_target_std = self._softplus(mean_raw[:, 2:4])
            # mean_target_std = mean_target_std * self._mean_target_std_ini / self._softplus(torch.zeros_like(mean_target_std))
            # mean_target_std = mean_target_std + self._mean_target_std_min
            mean_target_std = self._mean_target_std_min + (self._mean_target_std_max - self._mean_target_std_min) * 0.5 * (torch.tanh(mean_raw[:, 2:4]) + 1.0)

            mean = torch.concat((mean_target_pos, mean_target_std), dim=-1)
        else:
            mean = 1.0 * torch.tanh(mean_raw / 1.0)  # 1.3, / 1.0
            # mean = mean_raw
        return mean

    def update_distribution(self, observations):
        
        if self.use_discrete_policy:
            # One-hot Categorical
            logits_merged = self.actor(observations)
            logits = logits_merged.view((*logits_merged.shape[:-1], self.num_actions, self.num_bins))
            logits = 2.0 * torch.tanh(logits / 2.0)  # 5.0,  10.0

            # # Add epsilon-greedy probabilities
            # epsilon = 0.0  # 0.1
            # damping = 1e-3
            # probs = torch.exp(logits) / (1.0 + torch.exp(logits))
            # probs = probs / probs.sum(dim=-1).unsqueeze(dim=-1)
            # probs = (1.0 - epsilon) * probs + epsilon * torch.ones_like(probs) / probs.shape[-1]
            # logits = torch.log(probs / (1.0 - probs + damping))

            self.distribution = OneHotCategorical(logits=logits)
        else:
            # Gaussian
            if self.std_per_obs:
                output_merged = self.actor(observations)
                output = output_merged.view((*output_merged.shape[:-1], self.num_actions, 2))
                mean_raw = output[..., 0]
                std_raw = output[..., 1]
                # V1
                std_ini = np.log(np.exp(self.std_ini) - 1)
                std = F.softplus(std_raw + std_ini) + self.std_min
                # V2
                # logstd_min = -5.0  # -4.0  # -3.0  # -4
                # logstd_max = 2.0  # +0.0  # +1.0  # 0.2
                # logstd = logstd_min + 0.5*(logstd_max-logstd_min)*(torch.tanh(std_raw)+1.0)
                # std = torch.exp(logstd)  # +0.5)
            else:
                mean_raw = self.actor(observations)
                std = mean_raw*0. + self.std
            mean = self.transform_mean_prediction(mean_raw)

            self.distribution = Normal(mean, std)

            # self.distribution = SquashedNormal(mean_raw, self.std)

    def convert_onehot_to_action(self, onehot):
        return (self._trafo_scale * onehot).sum(dim=-1) + self._trafo_loc

    def convert_action_to_onehot(self, action):
        return nn.functional.one_hot(((action - self._trafo_scale[:, 0]) / self._trafo_delta).long(), num_classes=self.num_bins)

    def act(self, observations, **kwargs):

        if self.use_discrete_policy:
            # One-hot Categorical
            self.update_distribution(observations)
            return self.convert_onehot_to_action(self.distribution.sample())
        else:
            # Gaussian
            self.update_distribution(observations)
            return self.distribution.sample()

    def get_actions_log_prob(self, actions):

        if self.use_discrete_policy:
            # One-hot Categorical
            actions_onehot = self.convert_action_to_onehot(actions)
            return self.distribution.log_prob(actions_onehot).sum(dim=-1)
        else:
            # Gaussian
            return self.distribution.log_prob(actions).sum(dim=-1)

    def update_dist_and_get_actions_log_prob(self, observations, actions):
        self.update_distribution(observations)
        return self.get_actions_log_prob(actions)

    def update_distribution_and_get_actions_log_prob_mu_sigma_entropy(self, observations, actions):
        self.update_distribution(observations)
        return self.get_actions_log_prob(actions), self.action_mean, self.action_std, self.entropy

    def act_inference(self, observations):
  
        if self.use_discrete_policy:
            # One-hot Categorical
            logits_merged = self.actor(observations)
            logits = logits_merged.view((*logits_merged.shape[:-1], self.num_actions, self.num_bins))
            actions_onehot = 1.0 * (logits==logits.max(dim=-1)[0].unsqueeze(-1))
            return self.convert_onehot_to_action(actions_onehot)
        else:
            # Gaussian
            actions_mean_raw = self.actor(observations)
            actions_mean = self.transform_mean_prediction(actions_mean_raw)
            return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # value = self.critic(critic_observations)
        # return value
        values = [critic(critic_observations) for critic in self.critics]
        return torch.stack(values, dim=1)
        # return torch.concat(values, dim=1)  # FIXME: check if right


class ActorAttention(nn.Module):
  
    def __init__(self, input_dim, hidden_dims, split_dim, output_dim, activation, encoder, train_encoder=False):

        super(ActorAttention, self).__init__()

        self.split_dim = split_dim

        self._encoder = encoder
        self._train_encoder = train_encoder

        actor_layers = []
        actor_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        actor_layers.append(nn.LayerNorm(hidden_dims[0]))
        actor_layers.append(nn.Tanh())
        # actor_layers.append(nn.ELU())
        # actor_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)
        self._network = nn.Sequential(*actor_layers)

    def forward(self, observations):
        # split obs, don't encode target
        obs1 = observations[..., :self.split_dim]
        obs2 = observations[..., self.split_dim:]
        latent = self._encoder(obs1)
        if not self._train_encoder:
            latent = latent.detach()
        obs = torch.concat((latent, obs2), dim=-1)
        # obs = observations  # FIXME: testing
        return self._network(obs)


class CriticAttention(nn.Module):
  
    def __init__(self, input_dim, hidden_dims, split_dim, activation, encoder, output_dims=1, train_encoder=False):

        super(CriticAttention, self).__init__()

        self.split_dim = split_dim

        self._encoder = encoder
        self._train_encoder = train_encoder
  
        critic_layers = []
        critic_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        critic_layers.append(nn.LayerNorm(hidden_dims[0]))
        critic_layers.append(nn.Tanh())
        # critic_layers.append(nn.ELU())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                critic_layers.append(nn.Linear(hidden_dims[l], output_dims))
            else:
                critic_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                critic_layers.append(activation)
        self._network = nn.Sequential(*critic_layers)

    def forward(self, observations):
        # split obs, don't encode target
        obs1 = observations[..., :self.split_dim]
        obs2 = observations[..., self.split_dim:]
        latent = self._encoder(obs1)
        if not self._train_encoder:
            latent = latent.detach()
        obs = torch.concat((latent, obs2), dim=-1)
        # obs = observations  # FIXME: testing
        return self._network(obs)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


