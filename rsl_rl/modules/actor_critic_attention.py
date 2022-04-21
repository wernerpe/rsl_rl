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
from torch.nn.modules import rnn

from rsl_rl.modules.attention.encoders import EncoderAttention1, EncoderAttention2


class StructuredActorCriticAttention(nn.Module):
    def __init__(self,):
        #put code
        pass



class ActorCriticAttention(nn.Module):
    is_recurrent = False
    def __init__(self,  num_ego_obs,
                        num_ado_obs,
                        num_actions,
                        num_agents,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticAttention.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticAttention, self).__init__()

        activation = get_activation(activation)

        num_agent_max = num_agents
        num_ego_obs = num_ego_obs
        num_ado_obs = num_ado_obs
        mlp_input_dim_a = num_ego_obs + num_ado_obs
        mlp_input_dim_c = num_ego_obs + num_ado_obs

        encoder_type = kwargs['encoder_type']
        encoder_hidden_dims = kwargs['encoder_hidden_dims']

        # Encoder
        self.encoder = get_encoder(
          encoder_type=encoder_type,
          num_ego_obs=num_ego_obs, 
          num_ado_obs=num_ado_obs, 
          hidden_dims=encoder_hidden_dims, 
          num_agents=num_agent_max,
          activation=activation
        )

        # Policy
        self.actor = ActorAttention(
          input_dim=mlp_input_dim_a, 
          hidden_dims=actor_hidden_dims, 
          output_dim=num_actions, 
          activation=activation,
          encoder=self.encoder,
        )

        # Value function
        self.critic = CriticAttention(
          input_dim=mlp_input_dim_c, 
          hidden_dims=critic_hidden_dims, 
          activation=activation,
          encoder=self.encoder,
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

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
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


class ActorAttention(nn.Module):
  
    def __init__(self, input_dim, hidden_dims, output_dim, activation, encoder):

        super(ActorAttention, self).__init__()

        self._encoder = encoder

        actor_layers = []
        actor_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        actor_layers.append(nn.LayerNorm(hidden_dims[0]))
        actor_layers.append(nn.Tanh())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)
        self._network = nn.Sequential(*actor_layers)

    def forward(self, observations):
        latent = self._encoder(observations)
        return self._network(latent)


class CriticAttention(nn.Module):
  
    def __init__(self, input_dim, hidden_dims, activation, encoder):

        super(CriticAttention, self).__init__()

        self._encoder = encoder
  
        critic_layers = []
        critic_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        critic_layers.append(nn.LayerNorm(hidden_dims[0]))
        critic_layers.append(nn.Tanh())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                critic_layers.append(nn.Linear(hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                critic_layers.append(activation)
        self._network = nn.Sequential(*critic_layers)

    def forward(self, observations):
        latent = self._encoder(observations)
        return self._network(latent)


def get_encoder(encoder_type, num_ego_obs, num_ado_obs, hidden_dims, num_agents, activation):
    if encoder_type=="attention1":
        return EncoderAttention1(
          num_ego_obs=num_ego_obs, 
          num_ado_obs=num_ado_obs, 
          hidden_dims=hidden_dims, 
          output_dim=1, 
          num_agents=num_agents,
          activation=activation
        )
    elif encoder_type=="attention2":
        return EncoderAttention2(
          num_ego_obs=num_ego_obs, 
          num_ado_obs=num_ado_obs, 
          hidden_dims=hidden_dims, 
          output_dim=1, 
          num_agents=num_agents,
          activation=activation
        )


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


