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
import copy
from rsl_rl.modules import BilevelActorCritic, BilevelActorCriticAttention
import trueskill


class MultiTeamBilevelActorCritic(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_add_obs,
                        # num_agents,
                        num_actions,
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
                        **kwargs):

        super(MultiTeamBilevelActorCritic, self).__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        # self.num_agents = num_agents
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims= critic_hidden_dims
        self.activation = activation
        self.init_noise_std = init_noise_std
        self.kwargs = kwargs
        self.is_attentive = kwargs['attentive']
        self.num_teams = kwargs['numteams']
        self.team_size = kwargs['teamsize']

        self.n_critics = kwargs['numcritics']
        
        self.is_recurrent = False
        self.teams = [torch.tensor([idx for idx in range(self.team_size*start, self.team_size*start+self.team_size)], dtype=torch.long) for start in range(self.num_teams)]
        self.teamacs = [TeamBilevelActorCritic(num_actor_obs,
                                       num_critic_obs,
                                       num_add_obs,
                                       num_actions,
                                       device=device,
                                       encoder=encoder,
                                       train_encoder=train_encoder,
                                       act_min=act_min,
                                       act_max=act_max,
                                       act_ini=act_ini,
                                       discrete=discrete,
                                       actor_hidden_dims=actor_hidden_dims,
                                       critic_hidden_dims=critic_hidden_dims,
                                       critic_output_dim=1,  # 2?
                                       activation='elu',
                                       init_noise_std=init_noise_std,
                                       team_size=self.team_size,
                                       **kwargs) for idx in range(self.num_teams)]

        self.agentratings = []
        for idx in range(self.num_teams):
            self.agentratings.append((trueskill.Rating(mu=0),))

        self.is_recurrent = False

        self.max_num_models = kwargs['max_num_old_models']
        self.draw_probs_unnorm = np.ones((self.max_num_models,))
        self.draw_probs_unnorm[0:-3] = 0.4/(self.max_num_models-3)
        self.draw_probs_unnorm[-3:] = 0.6/3

        self.past_models = [self.teamacs[0].state_dict()]
        self.past_ratings_mu = [0]
        self.past_ratings_sigma = [self.agentratings[0][0].sigma]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.teamacs[0].action_mean

    @property
    def action_std(self):
        return self.teamacs[0].action_std
    
    @property
    def std(self):
        return self.teamacs[0].std
    
    @property
    def ego_action_mean(self):
        return self.teamacs[0].ego_mean
    
    @property
    def ego_action_std(self):
        return self.teamacs[0].ego_std

    @property
    def entropy(self):
        return self.teamacs[0].entropy

    @property
    def action_probs(self):
        return self.teamacs[0].action_probs
    
    @property
    def parameters(self):
        return self.teamacs[0].parameters

    # @property
    def state_dict(self):
        return self.teamacs[0].state_dict()
    
    def load_state_dict(self, path):
        for teamac in self.teamacs:
            teamac.load_state_dict(path)

    def load_multi_state_dict(self, paths):
        for idx, path in enumerate(paths):
            self.teamacs[idx].load_state_dict(path)

    def eval(self):
        for ac in self.teamacs:
            ac.eval()

    def train(self):
       [teamac.train() for teamac in self.teamacs]
       return None
    
    def to(self, device):
        for ac in self.teamacs:
            ac.to(device)
        return self

    def act(self, observations, **kwargs):
        actions = [ac.act(observations[:, self.teams[idx], :]) for idx, ac in enumerate(self.teamacs)]
        actions = torch.cat(tuple(actions), dim = 1)
        return actions
    
    def act_inference(self, observations, **kwargs):
        actions = [ac.act_inference(observations[:, self.teams[idx], :]) for idx, ac in enumerate(self.teamacs)]
        actions = torch.cat(tuple(actions), dim = 1)
        return actions
        
    def act_ac_train(self, observations, **kwargs):
        actions = self.teamacs[0].act(observations)
        return actions
    
    def update_distribution_and_get_actions_log_prob_mu_sigma_entropy(self, obs, actions):
        return self.teamacs[0].update_distribution_and_get_actions_log_prob_mu_sigma_entropy(obs, actions)
    
    def evaluate_inference_factors(self, observations, **kwargs):
        values = [ac.evaluate(observations[:, self.teams[idx],:]) for idx, ac in self.teamacs]
        values = torch.cat(tuple(values), dim = 1)
        return values
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.teamacs[0].evaluate(critic_observations)
        # #sum factors of value prediction
        # # value[:,:, 1] = torch.sum(value[:,:,1], dim =1).view(-1, 1)
        # value[..., 1] = torch.mean(value[...,1], dim=-1).unsqueeze(dim=-1)
        return value

    def get_ratings(self,):
        return self.agentratings

    def set_ratings(self, ratings):
        self.agentratings = ratings

    def update_ac_ratings(self, infos):
        #update performance metrics of current policies
        if 'ranking' in infos:         
            avgranking = [np.min(infos['ranking'][0][ids].cpu().numpy()) for ids in self.teams] #torch.mean(1.0*infos['ranking'], dim = 0).cpu().numpy()
            update_ratio = infos['ranking'][1]
            new_ratings = trueskill.rate(self.agentratings, avgranking)
            for old, new, it in zip(self.agentratings, new_ratings, range(len(self.agentratings))):
                mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
                sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
                self.agentratings[it] = (trueskill.Rating(mu, sigma),)

    def new_rating(self, mu, sigma):
        return (trueskill.Rating(mu = mu, sigma = sigma ),) 

    def redraw_ac_networks_KL_divergence(self, obs_batch):
        current_models = self.teamacs.copy()
        kl_divs = []
        self.teamacs[0].ac.update_distribution(obs_batch)
        dist_ego = self.teamacs[0].ac.distribution
        for ado_dict in self.past_models:
            self.teamacs[1].load_state_dict(ado_dict)
            self.teamacs[1].ac.update_distribution(obs_batch)
            dist_ado = self.teamacs[1].ac.distribution
            kl_divs.append(torch.mean(torch.sum(torch.distributions.kl_divergence(dist_ado, dist_ego), dim = 1)).item())       
        self.teamacs = current_models
        print('[MAAC POPULATION UPDATE] KLs', kl_divs)

        if np.min(kl_divs)>0.05:
            self.redraw_ac_networks(save = True)
        else:
            self.redraw_ac_networks(save = False)

    def redraw_ac_networks(self, save):

        #update population of competing agents, here simply load 
        #old version of agent 1 into ac2 slot
        if save:
            self.past_models.append(copy.deepcopy(self.teamacs[0].state_dict()))
            self.past_ratings_mu.append(self.agentratings[0][0].mu)
            self.past_ratings_sigma.append(self.agentratings[0][0].sigma)
            if len(self.past_models)> self.max_num_models:
                idx_del = np.random.randint(0, self.max_num_models-2)
                del self.past_models[idx_del]
                del self.past_ratings_mu[idx_del]
                del self.past_ratings_sigma[idx_del]

        #select model to load
        #renormalize dist
        if len(self.past_models) !=self.max_num_models:
            prob = 1/np.sum(self.draw_probs_unnorm[-len(self.past_models):]) * self.draw_probs_unnorm[-len(self.past_models):]
        else:
            prob = 1/np.sum(self.draw_probs_unnorm + 1e-4) *(self.draw_probs_unnorm + 1e-4)
        
        idx = np.random.choice(len(self.past_models), self.num_teams-1, p = prob)
        for agent_id, past_model_id in enumerate(idx):
            op_id = agent_id + 1

            state_dict = self.past_models[past_model_id]
            mu = self.past_ratings_mu[past_model_id]
            sigma = self.past_ratings_sigma[past_model_id]
            self.teamacs[op_id].load_state_dict(state_dict)
            self.teamacs[op_id].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            rating = (trueskill.Rating(mu = mu, sigma = sigma ),) 
            self.agentratings[op_id] = rating

        # rating_train_pol = (trueskill.Rating(mu = self.agentratings[0][0].mu, sigma = self.agentratings[0][0].sigma),)
        # self.agentratings[0] = rating_train_pol 
        


#multi agent actor critic with centralized critic for a single team
class TeamBilevelActorCritic(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_add_obs,
                        num_actions,
                        device,
                        team_size,
                        encoder,
                        train_encoder=False,
                        act_min=None,
                        act_max=None,
                        act_ini=None,
                        discrete=False,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        critic_output_dim=1,
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        
        super(TeamBilevelActorCritic, self).__init__()
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims= critic_hidden_dims
        self.activation = activation
        self.init_noise_std = init_noise_std
        self.kwargs = kwargs
        self.is_attentive = kwargs['attentive']
        self.team_size = kwargs['teamsize']
        self.action_means = []
        self.action_stds = []
        self.action_entropies = []

        self.team_size = team_size

        self.ac = BilevelActorCriticAttention(num_actor_obs=num_actor_obs, 
                                    num_critic_obs=num_critic_obs,
                                    num_add_obs=num_add_obs, 
                                    num_actions=num_actions, 
                                    device=device,
                                    encoder=encoder,
                                    train_encoder=train_encoder,
                                    act_min=act_min,
                                    act_max=act_max,
                                    act_ini=act_ini,
                                    discrete=discrete,
                                    actor_hidden_dims=actor_hidden_dims,
                                    critic_hidden_dims=critic_hidden_dims,
                                    activation=activation,
                                    init_noise_std=init_noise_std,
                                    critic_output_dim=critic_output_dim,
                                    **kwargs)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.action_means

    @property
    def action_std(self):
        return self.action_stds
    
    @property
    def mean(self):
        return self.action_mean

    @property
    def std(self):
        return self.action_std

    @property
    def ego_mean(self):
        return self.action_mean[:, 0]  # FIXME: discrete option correct?

    @property
    def ego_std(self):
        return self.action_std[:, 0]   # FIXME: discrete option correct?

    @property
    def entropy(self):
        return self.action_entropies

    @property
    def action_probs(self):
        return self.ac.distribution.probs
    
    #not sure how to fix these
    @property
    def parameters(self):
       return self.ac.parameters

    @property
    def state_dict(self):
       return self.ac.state_dict
    
    def load_state_dict(self, path):
       self.ac.load_state_dict(path)

    def eval(self):
        self.ac.eval()
        
    def train(self):
        self.ac.train()
        return None
    
    def to(self, device):
        self.ac.to(device)
        return self

    def act(self, observations, **kwargs):
        # actions = []
        # self.action_means = []
        # self.action_stds = []
        # self.action_entropies = []

        actions = self.ac.act(observations[:, :, :])
        self.action_means = self.ac.action_mean  # FIXME: discrete option correct?
        self.action_stds = self.ac.action_std    # FIXME: discrete option correct?
        self.action_entropies = self.ac.entropy
            
        return actions
    
    def act_inference(self, observations, **kwargs):
        actions = self.ac.act_inference(observations[:, :, :]).detach()
        return actions
    
    def update_distribution_and_get_actions_log_prob(self, obs, actions):
        return self.ac.update_dist_and_get_actions_log_prob(obs[:, :, :], actions[:, :, :])

    def update_distribution_and_get_actions_log_prob_mu_sigma_entropy(self, obs, actions):
        log_prob, mu, sigma, entropy = self.ac.update_distribution_and_get_actions_log_prob_mu_sigma_entropy(obs[:, :,:], actions[:, :, :])
        return log_prob, mu, sigma, entropy
    
    def evaluate_inference(self, observations, **kwargs):
        # values = []
        # values = [self.ac.evaluate(observations[:, 0,:])]
        # op_values = [ac.evaluate(observations[:, idx+1,:]).unsqueeze(1) for idx, ac in enumerate(self.opponent_acs)]  # FIXME: this does not exist!
        # values += op_values 
        # values = torch.cat(tuple(values), dim = 1)
        # return values
        return self.evaluate(observations, **kwargs)  # TODO: check whether correct / what it's used for
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.ac.evaluate(critic_observations)
        return value