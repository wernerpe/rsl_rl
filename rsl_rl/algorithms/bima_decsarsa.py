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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import MultiTeamBilevelDecCritic
from rsl_rl.storage import BimaSARSAStorage

import copy


class BimaDecSARSA:
    actor_critic: MultiTeamBilevelDecCritic
    target_actor_critic: MultiTeamBilevelDecCritic
    def __init__(self,
                 actor_critic,
                 centralized_value=False,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.target_actor_critic = copy.deepcopy(actor_critic)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = BimaSARSAStorage.Transition()

        # self._actions_min = torch.concat((
        #   self.actor_critic._mean_target_pos_min,
        #   self.actor_critic._mean_target_std_min
        #   ), dim=-1)
        # self._actions_max = torch.concat((
        #   self.actor_critic._mean_target_pos_max,
        #   self.actor_critic._mean_target_std_max
        #   ), dim=-1)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.n_critics = self.actor_critic.n_critics
        assert self.n_critics==1

        self.centralized_value = centralized_value

        self._value_stdv_run_mean = 0.05
        self._value_stdv_run_stdv = 0.0

        self.loss_fc = torch.nn.HuberLoss()
        self.num_update_steps = 0
        self.target_update_interval = 20  # 50

        self.use_sdqn = False  # True  # True
        self.use_mdqn = False  # True
        self.entropy_temperature = 0.03
        self.munchausen_coefficient = 0.9
        self.clip_value_min = -1e3

        self.num_bins = actor_critic.teamacs[0].ac.num_bins

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, num_agents):
        self.storage = BimaSARSAStorage(num_envs, num_transitions_per_env, actor_obs_shape, action_shape, num_agents, self.n_critics, self.num_bins, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    # def clip_actions(self, actions):
    #   return torch.clamp(actions, min=self._actions_min, max=self._actions_max)

    def compute_value_epsgreedy(self, q_pred):
        # Expected SARSA
        # q_pred = self.target_actor_critic.evaluate(obs)[:, self.actor_critic.teams[0], :]
        q_max = q_pred.amax(dim=-1).mean(dim=-1, keepdim=True)
        q_mean = q_pred.mean(dim=-1).mean(dim=-1, keepdim=True)
        epsilon = self.actor_critic.teamacs[0].ac.epsilon
        return (1.0 - epsilon) * q_max + epsilon * q_mean

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Compute the actions
        all_agent_actions =  self.actor_critic.act(obs).detach()

        # Record observations and actions
        self.transition.observations = obs[:, self.actor_critic.teams[0], :]
        self.transition.actions = all_agent_actions[:, self.actor_critic.teams[0], :]

        self.transition.target_curr_values = self.target_actor_critic.evaluate(obs[:, self.actor_critic.teams[0], :])

        return all_agent_actions

    def update_target_network(self,):
        state_dict = self.actor_critic.state_dict()
        self.target_actor_critic.load_state_dict(state_dict)
        print('[DecQL] Updated target network')
    
    def process_env_step(self, rewards, next_observations, dones, infos):
        self.transition.rewards = rewards[:, self.actor_critic.teams[0], :].clone()
        self.transition.next_observations = next_observations[:, self.actor_critic.teams[0], :]
        self.transition.dones = dones
        if 'agent_active' in infos:
          self.transition.active_agents = 1.0 * infos['agent_active']
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:  # TODO: check how many unsqueezes
            if self.use_sdqn or self.use_mdqn:
                bootstrap_target_value = self.entropy_temperature * torch.logsumexp(self.transition.target_curr_values / self.entropy_temperature, axis=-1).mean(dim=-1, keepdim=True)
            else:
                bootstrap_target_value = self.compute_value_epsgreedy(self.transition.target_curr_values)
            self.transition.rewards += self.gamma * bootstrap_target_value * infos['time_outs'].unsqueeze(1).unsqueeze(1).to(self.device)

        is_not_done = (1.0 - 1.0 * self.transition.dones.unsqueeze(dim=-1).unsqueeze(dim=-1))
        self.transition.target_next_values = is_not_done * self.target_actor_critic.evaluate(next_observations[:, self.actor_critic.teams[0], :])

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        pass

    def update(self):
        mean_value_loss = 0
        mean_param_norm = 0
        mean_target = 0
        mean_values = 0


        if self.actor_critic.is_attentive:
            generator = self.storage.attention_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            raise NotImplementedError
        for obs_batch, act_batch, rew_batch, next_obs_batch, dones_batch, target_curr_val_batch, target_next_val_batch in generator:

            act_idx_batch = torch.argmax(self.actor_critic.teamacs[0].ac.convert_action_to_onehot(act_batch), dim=-1)

            if self.use_sdqn or self.use_mdqn:
                target_val_batch = self.entropy_temperature * torch.logsumexp(target_next_val_batch / self.entropy_temperature, axis=-1).mean(axis=-1, keepdim=True)
            else:
                target_val_batch = self.compute_value_epsgreedy(target_next_val_batch)
            if self.use_mdqn:
                munchhausen_term = self.entropy_temperature * torch.log_softmax(target_curr_val_batch / self.entropy_temperature, axis=-1)
                munchausen_term_a = torch.gather(munchhausen_term, -1, act_idx_batch.unsqueeze(dim=-1)).mean(dim=-2)
                munchausen_term_a = torch.clip(munchausen_term_a, min=self.clip_value_min, max=0.)
                rew_batch += self.munchausen_coefficient * munchausen_term_a

            target = (rew_batch + self.gamma * target_val_batch).detach()

            val_all_batch = self.actor_critic.evaluate(obs_batch[:, self.actor_critic.teams[0], :])
            val_act_batch = torch.gather(val_all_batch, -1, act_idx_batch.unsqueeze(dim=-1)).mean(dim=-2)

            td_error = target - val_act_batch

            value_loss = self.loss_fc(td_error.squeeze(), (0*obs_batch[..., 0]).squeeze())

            # value_loss = value_loss.mean()
            target_mean = target.mean()
            values_mean = val_act_batch.mean()

            loss = value_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            param_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item() 
            mean_param_norm += param_norm.mean().item()

            mean_target += target_mean.item()
            mean_values += values_mean.item()

            if (self.num_update_steps + 1) % self.target_update_interval == 0:
                self.update_target_network()
            self.num_update_steps += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches

        mean_value_loss /= num_updates
        mean_param_norm /= num_updates
        mean_target /= num_updates
        mean_values /= num_updates

        self.storage.clear()

        return mean_value_loss, 0.0, 0.0, {'mean_norm': mean_param_norm,
                                            'mean_target': mean_target,
                                            'mean_values': mean_values,
                                            }

    def update_population(self,):
        # self.actor_critic.redraw_ac_networks()
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        elif self.actor_critic.is_attentive:
            generator = self.storage.attention_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        batch = next(generator)
        obs_batch = batch[0]

        self.actor_critic.redraw_ac_networks_KL_divergence(obs_batch)  # FIXME: uncomment & fix

    def update_ratings(self, eval_ranks, eval_ep_duration, max_ep_len):
        eval_team_ranks = -100. * torch.ones_like(eval_ranks[..., :len(self.actor_critic.teams)])

        for idx, team in enumerate(self.actor_critic.teams):
            eval_team_ranks[:, idx] = torch.min(eval_ranks[:, team], dim = 1)[0]  # .reshape(-1,1)
        # eval_team_ranks = (eval_team_ranks==0).type(torch.float)  # FIXME: this only works for 2 teams

        ratings = self.actor_critic.get_ratings()
        for ranks, dur in zip(eval_team_ranks, eval_ep_duration):
            new_ratings = trueskill.rate(ratings, ranks.cpu().numpy())
            update_ratio = 1.0*dur.item()/max_ep_len
            for it, (old, new) in enumerate(zip(ratings, new_ratings)):
                mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
                sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
                ratings[it] = (trueskill.Rating(mu, sigma),)
        self.actor_critic.set_ratings(ratings)
        return 
