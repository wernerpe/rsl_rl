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
from torch.distributions import Normal
import copy
from rsl_rl.modules import ActorCritic
import trueskill

#2 agent actor critic
class MAActorCritic():
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        
        self.ac1 = ActorCritic( num_actor_obs,
                                num_critic_obs,
                                num_actions,
                                actor_hidden_dims,
                                critic_hidden_dims,
                                activation,
                                init_noise_std, 
                                **kwargs)
        
        self.ac2 = copy.deepcopy(self.ac1)
        self.is_recurrent = False


        self.agentratings = []
        for idx in range(2):
            self.agentratings.append((trueskill.Rating(),))

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.ac1.distribution.mean

    @property
    def action_std(self):
        return self.ac1.distribution.stddev
    
    @property
    def entropy(self):
        return self.ac1.distribution.entropy().sum(dim=-1)
    
    @property
    def parameters(self):
        return self.ac1.parameters
        
    def train(self):
       self.ac1.train()
       return None
    
    def to(self, device):
        self.ac1.to(device)
        self.ac2.to(device)
        return self

    def act(self, observations, **kwargs):
        actions1 = self.ac1.act(observations[:, 0,:])
        actions2 = self.ac2.act(observations[:, 1,:])
        actions = torch.cat((actions1.unsqueeze(1), actions2.unsqueeze(1)), dim = 1)
        return actions
    
    def get_actions_log_prob(self, actions):
        return self.ac1.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions1 = self.ac1.act_inference(observations)
        actions2 = self.ac2.act_inference(observations)
        actions = torch.cat((actions1.unsqueeze(1), actions2.unsqueeze(1)), dim = 1)
        return actions
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.ac1.critic(critic_observations)
        return value

    def update_ac_ratings(self, dones, infos):
        #update performance metrics of current policies
        num_races = torch.sum(1.0*dones)
        if num_races:
            avgranking = torch.mean(infos['ranking'][dones, :], dim = 0)
            #only update rankings if result is significant
            if avgranking[0] > 0.7:
                avgranking = [1, 0]
            elif avgranking[0] < 0.3:
                avgranking = [1, 0]
            else:
                avgranking = [0, 0]
                return

            update_ratio = num_races/len(dones)
            weighting = {(0,0):update_ratio, (1,0):update_ratio} 
            self.agentratings = trueskill.rate(self.agentratings, avgranking, weights = weighting)

    def redraw_ac_networks(self):
        #update population of competing agents, here simply load 
        #old version of agent 1 into ac2 slot
        self.ac2 = copy.deepcopy(self.ac1)
        #potentially randomize here