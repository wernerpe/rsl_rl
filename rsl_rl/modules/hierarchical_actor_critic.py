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


class HierarchicalActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_hl_obs,
                        num_critic_hl_obs,
                        num_actions_hl,
                        num_actor_ll_obs,
                        num_critic_ll_obs,
                        num_actions_ll,
                        actor_hl_hidden_dims=[256, 256, 256],
                        critic_hl_hidden_dims=[256, 256, 256],
                        activation_hl='elu',
                        init_noise_hl_std=1.0,
                        actor_ll_hidden_dims=[256, 256, 256],
                        critic_ll_hidden_dims=[256, 256, 256],
                        activation_ll='elu',
                        init_noise_ll_std=1.0,
                        **kwargs):
        if kwargs:
            print("HierarchicalActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HierarchicalActorCritic, self).__init__()

        # ### High-level actor (HL)
        activation_hl = get_activation(activation_hl)

        mlp_input_dim_a_hl = num_actor_hl_obs
        mlp_input_dim_c_hl = num_critic_hl_obs

        # Policy
        actor_hl_layers = []
        actor_hl_layers.append(nn.Linear(mlp_input_dim_a_hl, actor_hl_hidden_dims[0]))
        actor_hl_layers.append(nn.LayerNorm(actor_hl_hidden_dims[0]))
        actor_hl_layers.append(nn.Tanh())
        # actor_layers.append(activation)
        for l in range(len(actor_hl_hidden_dims)):
            if l == len(actor_hl_hidden_dims) - 1:
                actor_hl_layers.append(nn.Linear(actor_hl_hidden_dims[l], num_actions_hl))
            else:
                actor_hl_layers.append(nn.Linear(actor_hl_hidden_dims[l], actor_hl_hidden_dims[l + 1]))
                actor_hl_layers.append(activation_hl)
        self.actor_hl = nn.Sequential(*actor_hl_layers)

        # Value function
        critic_hl_layers = []
        critic_hl_layers.append(nn.Linear(mlp_input_dim_c_hl, critic_hl_hidden_dims[0]))
        critic_hl_layers.append(nn.LayerNorm(critic_hl_hidden_dims[0]))
        critic_hl_layers.append(nn.Tanh())
        # critic_layers.append(activation)
        for l in range(len(critic_hl_hidden_dims)):
            if l == len(critic_hl_hidden_dims) - 1:
                critic_hl_layers.append(nn.Linear(critic_hl_hidden_dims[l], 1))
            else:
                critic_hl_layers.append(nn.Linear(critic_hl_hidden_dims[l], critic_hl_hidden_dims[l + 1]))
                critic_hl_layers.append(activation_hl)
        self.critic_hl = nn.Sequential(*critic_hl_layers)

        print(f"Actor HL MLP: {self.actor_hl}")
        print(f"Critic HL MLP: {self.critic_hl}")

        # Action noise
        self.std_hl = nn.Parameter(init_noise_hl_std * torch.ones(num_actions_hl))
        self.distribution_hl = None


        # ### Low-level actor (LL)
        activation_ll = get_activation(activation_ll)

        mlp_input_dim_a_ll = num_actor_ll_obs
        mlp_input_dim_c_ll = num_critic_ll_obs

        # Policy
        actor_ll_layers = []
        actor_ll_layers.append(nn.Linear(mlp_input_dim_a_ll, actor_ll_hidden_dims[0]))
        actor_ll_layers.append(nn.LayerNorm(actor_ll_hidden_dims[0]))
        actor_ll_layers.append(nn.Tanh())
        # actor_layers.append(activation)
        for l in range(len(actor_ll_hidden_dims)):
            if l == len(actor_ll_hidden_dims) - 1:
                actor_ll_layers.append(nn.Linear(actor_ll_hidden_dims[l], num_actions_ll))
            else:
                actor_ll_layers.append(nn.Linear(actor_ll_hidden_dims[l], actor_ll_hidden_dims[l + 1]))
                actor_ll_layers.append(activation_ll)
        self.actor_ll = nn.Sequential(*actor_ll_layers)

        # Value function
        critic_ll_layers = []
        critic_ll_layers.append(nn.Linear(mlp_input_dim_c_ll, critic_ll_hidden_dims[0]))
        critic_ll_layers.append(nn.LayerNorm(critic_ll_hidden_dims[0]))
        critic_ll_layers.append(nn.Tanh())
        # critic_layers.append(activation)
        for l in range(len(critic_ll_hidden_dims)):
            if l == len(critic_ll_hidden_dims) - 1:
                critic_ll_layers.append(nn.Linear(critic_ll_hidden_dims[l], 1))
            else:
                critic_ll_layers.append(nn.Linear(critic_ll_hidden_dims[l], critic_ll_hidden_dims[l + 1]))
                critic_ll_layers.append(activation_ll)
        self.critic_ll = nn.Sequential(*critic_ll_layers)

        print(f"Actor LL MLP: {self.actor_ll}")
        print(f"Critic LL MLP: {self.critic_ll}")

        # Action noise
        self.std_ll = nn.Parameter(init_noise_ll_std * torch.ones(num_actions_ll))
        self.distribution_ll = None

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
        mean_hl = self.distribution_hl.mean
        mean_ll = self.distribution_ll.mean
        return torch.cat((mean_hl, mean_ll), dim=-1)

    @property
    def action_std(self):
        stddev_hl = self.distribution_hl.stddev
        stddev_ll = self.distribution_ll.stddev
        return torch.cat((stddev_hl, stddev_ll), dim=-1)

    @property
    def action_hl_mean(self):
        return self.distribution_hl.mean

    @property
    def action_hl_std(self):
        return self.distribution_hl.stddev

    @property
    def action_ll_mean(self):
        return self.distribution_ll.mean

    @property
    def action_ll_std(self):
        return self.distribution_ll.stddev
    
    @property
    def entropy(self):
        # FIXME: does this make any sense?
        entropy_hl = self.distribution_hl.entropy().sum(dim=-1)
        entropy_ll = self.distribution_ll.entropy().sum(dim=-1)
        return entropy_hl + entropy_ll

    @property
    def entropy_hl(self):
        return self.distribution_hl.entropy().sum(dim=-1)

    @property
    def entropy_ll(self):
        return self.distribution_ll.entropy().sum(dim=-1)

    def update_distribution_hl(self, observations_hl):
        mean_hl = self.actor_hl(observations_hl)
        self.distribution_hl = Normal(mean_hl, mean_hl*0. + self.std_hl)
      
    def update_distribution_ll(self, observations_ll):
        mean_ll = self.actor_ll(observations_ll)
        self.distribution_ll = Normal(mean_ll, mean_ll*0. + self.std_ll)

    def act_hl(self, observations_hl, **kwargs):
        self.update_distribution_hl(observations_hl)
        return self.distribution_hl.sample()

    def act_ll(self, observations_ll, **kwargs):
        self.update_distribution_ll(observations_ll)
        return self.distribution_ll.sample()
    
    def get_actions_hl_log_prob(self, actions_hl):
        return self.distribution_hl.log_prob(actions_hl).sum(dim=-1)

    def get_actions_ll_log_prob(self, actions_ll):
        return self.distribution_ll.log_prob(actions_ll).sum(dim=-1)

    def act_hl_inference(self, observations_hl):
        actions_hl_mean = self.actor_hl(observations_hl)
        return actions_hl_mean

    def act_ll_inference(self, observations_ll):
        actions_ll_mean = self.actor_ll(observations_ll)
        return actions_ll_mean

    def evaluate_hl(self, critic_observations_hl, **kwargs):
        value_hl = self.critic_hl(critic_observations_hl)
        return value_hl

    def evaluate_ll(self, critic_observations_ll, **kwargs):
        value_ll = self.critic_ll(critic_observations_ll)
        return value_ll

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


