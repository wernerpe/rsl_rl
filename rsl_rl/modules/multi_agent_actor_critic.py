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

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims= critic_hidden_dims
        self.activation = activation
        self.init_noise_std = init_noise_std
        self.kwargs = kwargs

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
            self.agentratings.append((trueskill.Rating(mu=0),))

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
    def std(self):
        return self.ac1.std

    @property
    def entropy(self):
        return self.ac1.distribution.entropy().sum(dim=-1)
    
    @property
    def parameters(self):
        return self.ac1.parameters
    
    def load_state_dict(self, path):
        self.ac1.load_state_dict(path)
        self.ac2.load_state_dict(path)

    def eval(self):
        self.ac1.eval()
        self.ac2.eval()

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
    
    def act_inference(self, observations):
        actions1 = self.ac1.act_inference(observations[:, 0,:])
        actions2 = self.ac2.act_inference(observations[:, 1,:])
        actions = torch.cat((actions1.unsqueeze(1), actions2.unsqueeze(1)), dim = 1)
        return actions
        
    def act_ac_train(self, observations, **kwargs):
        actions = self.ac1.act(observations)
        return actions
    
    def get_actions_log_prob(self, actions):
        return self.ac1.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions1 = self.ac1.act_inference(observations[:, 0,:])
        actions2 = self.ac2.act_inference(observations[:, 1,:])
        actions = torch.cat((actions1.unsqueeze(1), actions2.unsqueeze(1)), dim = 1)
        return actions
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.ac1.critic(critic_observations)
        return value

    def update_ac_ratings(self, dones, infos):
        #update performance metrics of current policies
        if 'ranking' in infos:         
            dones_idx = torch.unique(torch.where(dones)[0])
            avgranking = torch.mean(1.0*infos['ranking'], dim = 0)

            #only update rankings if result is significant
            if avgranking[0] > 0.55:
                avgranking = [1, 0]
            elif avgranking[0] < 0.45:
                avgranking = [0, 1]
            else:
                #result isnt strong enough
                return

            update_ratio = (len(dones_idx)/len(dones)*torch.mean(infos['percentage_max_episode_length'])).item()
            new_ratings = trueskill.rate(self.agentratings, avgranking)
            for old, new, it in zip(self.agentratings, new_ratings, range(len(self.agentratings))):
                mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
                sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
                self.agentratings[it] = (trueskill.Rating(mu, sigma),)

    def redraw_ac_networks(self):
        #update population of competing agents, here simply load 
        #old version of agent 1 into ac2 slot
        self.ac1 = self.ac1
        self.ac2 = ActorCritic( self.num_actor_obs,
                                self.num_critic_obs,
                                self.num_actions,
                                self.actor_hidden_dims,
                                self.critic_hidden_dims,
                                self.activation,
                                self.init_noise_std, 
                                **self.kwargs)

        self.ac2.load_state_dict(self.ac1.state_dict())     
        self.ac2.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))             
        new_rating = (trueskill.Rating(mu=self.agentratings[0][0].mu),)
        self.agentratings[1] = self.agentratings[0]
        self.agentratings[0] = new_rating
        #potentially randomize here