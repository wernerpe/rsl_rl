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


class BilevelDecCriticAttention(nn.Module):
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
                        act_all=None,
                        discrete=False,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        critic_output_dim=1,
                        # std_per_obs=True,
                        **kwargs):
        if kwargs:
            print("DecCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(BilevelDecCriticAttention, self).__init__()

        activation = get_activation(activation)

        encoder_type = kwargs['encoder_type']
        encoder_attend_dims = kwargs['encoder_attend_dims']

        teamsize = kwargs['teamsize']
        # numteams = kwargs['numteams']

        num_ego_obs = kwargs['num_ego_obs']
        num_ado_obs = kwargs['num_ado_obs']

        self.n_critics = kwargs['numcritics']
        self.is_attentive = kwargs['attentive']

        # num_agent_max = num_agents
        num_ego_obs = num_ego_obs
        num_ado_obs = num_ado_obs
        if encoder_type=='identity':
            mlp_input_dim_a = encoder_attend_dims[-1]
            mlp_input_dim_c = encoder_attend_dims[-1]
        elif encoder_type=='attention4':
            mlp_input_dim_a = num_ego_obs + encoder_attend_dims[-1] // 4  # //heads for new version
            mlp_input_dim_c = num_ego_obs + encoder_attend_dims[-1] // 4  # //heads for new version
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

        self.std_per_obs = kwargs['std_per_obs']
        self.std_ini = init_noise_std
        self.std_min = 3.e-2  # 1.e-2

        ### Discrete actor
        # # OLD
        # self.num_bins = 5  # 5
        # self._trafo_scale = torch.tensor(np.linspace(start=act_min, stop=act_max, num=self.num_bins, axis=-1), dtype=torch.float, device=device)
        # self._trafo_delta = self._trafo_scale[:, 1] - self._trafo_scale[:, 0]
        # self._trafo_loc = 0.0 * torch.tensor(act_min, dtype=torch.float, device=device)

        # NEW
        self.num_bins = len(act_all[0])
        self._trafo_scale = torch.tensor(act_all, dtype=torch.float, device=device)
        self._trafo_loc = 0.0 * torch.tensor(act_min, dtype=torch.float, device=device)

        self.mlp_output_dim_a = num_actions * self.num_bins
       
        # Value function
        self.critic = CriticAttention(
          input_dim=mlp_input_dim_c, 
          hidden_dims=critic_hidden_dims, 
          split_dim=enc_split_dim,
          num_actions=self.num_actions,
          num_bins=self.num_bins,
          activation=activation,
          encoder=encoder,
          train_encoder=train_encoder,
          teamsize = teamsize
        )

        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.epsilon = 0.2  # 0.1
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
        # One-hot Categorical --> weight each option by prob
        return (self._trafo_scale * self.distribution.probs).sum(dim=-1)

    @property
    def action_std(self):
        # One-hot Categorical --> weight each option by prob
        return torch.sqrt(((self._trafo_scale - self.action_mean.unsqueeze(dim=-1))**2 * self.distribution.probs).sum(dim=-1))

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        self.update_distribution_with_epsilon(observations, epsilon=self.epsilon)

    def convert_onehot_to_action(self, onehot):
        return (self._trafo_scale * onehot).sum(dim=-1) + self._trafo_loc

    def convert_action_to_onehot(self, action):
        # return nn.functional.one_hot(((action - self._trafo_scale[:, 0]) / self._trafo_delta).long(), num_classes=self.num_bins)
        return 1.0 * (action.unsqueeze(-1)==self._trafo_scale.unsqueeze(0).unsqueeze(0))

    def act(self, observations, **kwargs):
        return self.act_with_epsilon(observations, epsilon=self.epsilon)

    def act_inference(self, observations):
       return self.act_with_epsilon(observations, epsilon=0.0)

    def act_with_epsilon(self, observations, epsilon):
        self.update_distribution_with_epsilon(observations, epsilon)
        actions_onehot = self.distribution.sample()
        return self.convert_onehot_to_action(actions_onehot)

    def get_actions_log_prob(self, actions):
        actions_onehot = self.convert_action_to_onehot(actions)
        return self.distribution.log_prob(actions_onehot).sum(dim=-1)

    def update_dist_and_get_actions_log_prob(self, observations, actions):
        self.update_distribution(observations)
        return self.get_actions_log_prob(actions)

    def update_distribution_and_get_actions_log_prob_mu_sigma_entropy(self, observations, actions):
        self.update_distribution(observations)
        return self.get_actions_log_prob(actions), self.action_mean, self.action_std, self.entropy

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def update_distribution_with_epsilon(self, observations, epsilon):
        q_values = self.evaluate(observations)

        ### epsilon-greedy
        max_value = torch.max(q_values, dim=-1, keepdim=True)[0]
        greedy_probs = 1.0*(max_value == q_values) #torch.equal(q_values, max_value)
        greedy_probs /= torch.sum(greedy_probs, dim=-1, keepdim=True)  # why?

        # num_dim = q_values.shape[-1]
        # dither_probs = 1.0 / (num_dim) * torch.ones_like(q_values)
        # probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

        ### softmax
        act_greedy = 1.0 * (epsilon == 0.0)
        probs = (1.0-act_greedy) * torch.softmax(q_values, dim=-1) + act_greedy * greedy_probs

        self.distribution = torch.distributions.OneHotCategorical(probs=probs)


class CriticAttention(nn.Module):
  
    def __init__(self, input_dim, hidden_dims, split_dim, activation, encoder, num_actions, num_bins, teamsize, train_encoder=False, distributional=False):

        super(CriticAttention, self).__init__()

        self.split_dim = split_dim

        self._encoder = encoder
        self._train_encoder = train_encoder
        self._num_actions = num_actions
        self._num_bins = num_bins
        self._teamsize = teamsize

        critic_layers = []
        critic_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        critic_layers.append(nn.LayerNorm(hidden_dims[0]))
        critic_layers.append(nn.Tanh())
        # critic_layers.append(nn.ELU())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                critic_layers.append(nn.Linear(hidden_dims[l], num_actions * num_bins))
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
        q_values = self._network(obs)
        return q_values.view((-1, self._teamsize, self._num_actions, self._num_bins))


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


