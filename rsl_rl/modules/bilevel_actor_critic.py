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


class BilevelActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        act_min,
                        act_max,
                        act_ini,
                        device,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(BilevelActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        self.mlp_output_dim_a = num_actions

        self._mean_target_pos_min = nn.Parameter(torch.tensor(act_min[:2]), requires_grad=False)
        self._mean_target_pos_max = nn.Parameter(torch.tensor(act_max[:2]), requires_grad=False)
        self._mean_target_pos_off = nn.Parameter(torch.tensor([3.5, 0.0]), requires_grad=False)

        self._mean_target_std_min = nn.Parameter(torch.tensor(act_min[2:]), requires_grad=False)
        self._mean_target_std_max = nn.Parameter(torch.tensor(act_max[2:]), requires_grad=False)
        self._mean_target_std_ini = nn.Parameter(torch.tensor(act_ini[2:]), requires_grad=False)
        self._softplus = nn.Softplus()

        # Discrete actor
        self.num_bins = 5
        self._trafo_scale = torch.tensor(np.linspace(start=act_min, stop=act_max, num=self.num_bins, axis=-1), dtype=torch.float, device=device)
        self._trafo_delta = self._trafo_scale[:, 1] - self._trafo_scale[:, 0]
        self._trafo_loc = 0.0 * torch.tensor(act_min, dtype=torch.float, device=device)
        self.mlp_output_dim_a = num_actions * self.num_bins
        self.num_actions = num_actions
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(nn.LayerNorm(actor_hidden_dims[0]))
        actor_layers.append(nn.Tanh())
        # actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], self.mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(nn.LayerNorm(critic_hidden_dims[0]))
        critic_layers.append(nn.Tanh())
        # critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

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
        # # Gaussian
        # return self.distribution.mean

        # One-hot Categorical --> weight each option by prob
        return (self._trafo_scale * self.distribution.probs).sum(dim=-1)

    @property
    def action_std(self):
        # # Gaussian
        # return self.distribution.stddev

        # One-hot Categorical --> weight each option by prob
        return torch.sqrt(((self._trafo_scale - self.action_mean.unsqueeze(dim=-1))**2 * self.distribution.probs).sum(dim=-1))


    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def transform_mean_prediction(self, mean_raw):

        mean_target_pos = self._mean_target_pos_min + (self._mean_target_pos_max - self._mean_target_pos_min) * 0.5 * (torch.tanh(mean_raw[:, 0:2]) + 1.0)
        # mean_target_pos = mean_target_pos + self._mean_target_pos_off

        # mean_target_std = self._softplus(mean_raw[:, 2:4])
        # mean_target_std = mean_target_std * self._mean_target_std_ini / self._softplus(torch.zeros_like(mean_target_std))
        # mean_target_std = mean_target_std + self._mean_target_std_min
        mean_target_std = self._mean_target_std_min + (self._mean_target_std_max - self._mean_target_std_min) * 0.5 * (torch.tanh(mean_raw[:, 2:4]) + 1.0)

        mean = torch.concat((mean_target_pos, mean_target_std), dim=-1)
        return mean

    def update_distribution(self, observations):
        # # Gaussian
        # mean_raw = self.actor(observations)
        # mean = self.transform_mean_prediction(mean_raw)
        # self.distribution = Normal(mean, mean*0. + self.std)

        # One-hot Categorical
        logits_merged = self.actor(observations)
        logits = logits_merged.view((*logits_merged.shape[:-1], self.num_actions, self.num_bins))

        # Add epsilon-greedy probabilities
        epsilon = 0.1
        damping = 1e-3
        probs = torch.exp(logits) / (1.0 + torch.exp(logits))
        probs = probs / probs.sum(dim=-1).unsqueeze(dim=-1)
        probs = (1.0 - epsilon) * probs + epsilon * torch.ones_like(probs) / probs.shape[-1]
        logits = torch.log(probs / (1.0 - probs + damping))

        self.distribution = OneHotCategorical(logits=logits)
        # self.distribution = TransformedDistribution(base_dist, [AffineTransform(scale=self._trafo_scale, loc=self._trafo_loc)])

    def convert_onehot_to_action(self, onehot):
        return (self._trafo_scale * onehot).sum(dim=-1) + self._trafo_loc

    def convert_action_to_onehot(self, action):
        return nn.functional.one_hot(((action - self._trafo_scale[:, 0]) / self._trafo_delta).long(), num_classes=self.num_bins)

    def act(self, observations, **kwargs):
        # # Gaussian
        # self.update_distribution(observations)
        # return self.distribution.sample()
    
        # One-hot Categorical
        self.update_distribution(observations)
        return self.convert_onehot_to_action(self.distribution.sample())

    def get_actions_log_prob(self, actions):
        # # Gaussian
        # return self.distribution.log_prob(actions).sum(dim=-1)

        # One-hot Categorical
        actions_onehot = self.convert_action_to_onehot(actions)
        return self.distribution.log_prob(actions_onehot).sum(dim=-1)

    def act_inference(self, observations):
        # # Gaussian
        # actions_mean_raw = self.actor(observations)
        # actions_mean = self.transform_mean_prediction(actions_mean_raw)
        # return actions_mean

        # One-hot Categorical
        actions_mean_raw = self.actor(observations)
        actions_mean = self.transform_mean_prediction(actions_mean_raw)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

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


