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

from rsl_rl.modules import MultiTeamBilevelActorCritic
from rsl_rl.storage import BimaRolloutStorage

class BimaPPO:
    actor_critic: MultiTeamBilevelActorCritic
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
                 max_grad_norm=2.0,
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
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = BimaRolloutStorage.Transition()

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

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, num_agents):
        self.storage = BimaRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, num_agents, self.n_critics, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    # def clip_actions(self, actions):
    #   return torch.clamp(actions, min=self._actions_min, max=self._actions_max)

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Compute the actions and values
        all_agent_actions =  self.actor_critic.act(obs).detach()
        self.transition.actions = all_agent_actions[:, self.actor_critic.teams[0], :]
        self.transition.values = self.actor_critic.evaluate(critic_obs[:, self.actor_critic.teams[0], :]).detach()
        self.values_separate = torch.concat([self.actor_critic.evaluate(critic_obs[:, agent_id, :].unsqueeze(1)).detach() for agent_id in self.actor_critic.teams[0]], dim=-2)

        #only record log prob of actions from net to train
        lp, m, s, e = self.actor_critic.update_distribution_and_get_actions_log_prob_mu_sigma_entropy(obs[:, self.actor_critic.teams[0], :], self.transition.actions)

        self.transition.actions_log_prob = lp.detach()
        self.transition.action_mean = m.detach()
        self.transition.action_sigma = s.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs[:, self.actor_critic.teams[0], :]
        self.transition.critic_observations = critic_obs[:, self.actor_critic.teams[0], :]
        return all_agent_actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards[:, self.actor_critic.teams[0], :].clone().unsqueeze(dim=1).repeat(1, self.n_critics, 1, 1)
        self.transition.dones = dones
        if 'agent_active' in infos:
          self.transition.active_agents = 1.0 * infos['agent_active']

        # self.transition = self.transition.squeeze_single_dims()  # FIXME: accomodate single agent
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:  # TODO: check how many unsqueezes
            self.transition.rewards += self.gamma * self.transition.values * infos['time_outs'].unsqueeze(1).unsqueeze(1).unsqueeze(1).to(self.device)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs[:, self.actor_critic.teams[0], :]).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_joint_ratio_values = 0
        mean_jr_den = 0
        mean_jr_num = 0
        mean_advantage_values = 0
        mean_mu0_batch = 0
        mean_param_norm = 0

        mean_enc_wproj_a = 0
        mean_enc_wproj_c = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        elif self.actor_critic.is_attentive:
            generator = self.storage.attention_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_individual_batch, advantages_batch, returns_individual_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, active_agents in generator:

            actions_log_prob_batch, mu_batch, sigma_batch, entropy_batch = self.actor_critic.update_distribution_and_get_actions_log_prob_mu_sigma_entropy(obs_batch, actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # mu_batch = mu_batch.flatten(0,1)
            # sigma_batch = sigma_batch.flatten(0,1)
            # entropy_batch = entropy_batch.flatten(0,1)

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate


            # Surrogate loss
            ratio = torch.squeeze(torch.exp(torch.sum(actions_log_prob_batch, dim = 1) - torch.sum(old_actions_log_prob_batch.squeeze(-1), dim = 1)))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Centralization via mean
            if self.centralized_value:
                value_batch = value_batch.squeeze(dim=-1).mean(dim=-1)
                target_values_individual_batch = target_values_individual_batch.mean(dim=-1)
                returns_individual_batch = returns_individual_batch.mean(dim=-1)
            else:
                value_batch = value_batch.squeeze(dim=-1)

            # Value function loss
            if self.use_clipped_value_loss:
                # Old
                # value_individual_clipped = target_values_individual_batch + (value_batch[..., 0] - target_values_individual_batch).clamp(-self.clip_param, self.clip_param)
                # value_losses_individual = (value_batch[..., 0] - returns_individual_batch).pow(2)
                # value_losses_individual_clipped = (value_individual_clipped - returns_individual_batch).pow(2)
                # value_loss_individual = torch.max(value_losses_individual, value_losses_individual_clipped).mean((0, -1))
                
                # New
                value_individual_clipped = target_values_individual_batch + (value_batch - target_values_individual_batch).clamp(-self.clip_param, self.clip_param)
                value_losses_individual = (value_batch - returns_individual_batch).pow(2)
                value_losses_individual_clipped = (value_individual_clipped - returns_individual_batch).pow(2)
                value_loss_individual = torch.max(value_losses_individual, value_losses_individual_clipped)


                # #since both entries the same pick team entry for agent 0 by convention
                # value_team_clipped = target_values_team_batch[..., 0] + (value_batch[..., 0, 1] - target_values_team_batch[..., 0]).clamp(-self.clip_param,
                #                                                                                 self.clip_param)
                # value_losses_team = (value_batch[..., 0, 1] - returns_team_batch[..., 0]).pow(2)
                # value_losses_team_clipped = (value_team_clipped - returns_team_batch[..., 0]).pow(2)
                # value_loss_team = torch.max(value_losses_team, value_losses_team_clipped).mean(dim=0)
            else:
                # Old
                # value_loss_individual = (returns_individual_batch - value_batch[:,:,0]).pow(2).mean()
                # New
                value_loss_individual = (returns_individual_batch - value_batch).pow(2)
                # #since both entries the same pick team entry for agent 0 by convention
                # value_loss_team = (returns_team_batch[:,0] - value_batch[:,0,1]).pow(2).mean()
            value_loss = value_loss_individual.mean()

            loss = surrogate_loss + value_loss - self.entropy_coef * entropy_batch.mean()

            # # Penalize large action means to counteract tanh saturation --> for Gaussian
            # loss += (-1.0) * (self.actor_critic.distribution.mean**2).mean() - 0.1 * (self.actor_critic.distribution.scale**2).mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            param_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item() 
            mean_surrogate_loss += surrogate_loss.item()
            mean_joint_ratio_values += ratio.mean().item()
            mean_advantage_values += advantages_batch.mean().item()
            mean_jr_num += torch.sum(actions_log_prob_batch, dim = 1).mean().item()
            mean_jr_den += torch.sum(old_actions_log_prob_batch, dim = 1).mean().item()
            mean_mu0_batch += mu_batch[:, 0].mean().item()
            mean_entropy_loss += entropy_batch.mean().item()
            mean_param_norm += param_norm.mean().item()

            mean_enc_wproj_a += self.actor_critic.teamacs[0].ac.actor._encoder.projection_net.weight.mean().item()
            mean_enc_wproj_c += self.actor_critic.teamacs[0].ac.critics[0]._encoder.projection_net.weight.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_joint_ratio_values /= num_updates
        mean_advantage_values /= num_updates
        mean_jr_num /= num_updates
        mean_jr_den /= num_updates
        mean_mu0_batch /= num_updates
        mean_entropy_loss /= num_updates
        mean_param_norm /= num_updates

        mean_enc_wproj_a /= num_updates
        mean_enc_wproj_c /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, {'mean_joint_ratio_val': mean_joint_ratio_values, 
                                                                          'mean_advantage_val': mean_advantage_values, 
                                                                          'mean_jr_num': mean_jr_num, 
                                                                          'mean_jr_den': mean_jr_den, 
                                                                          'mean_mu0': mean_mu0_batch,
                                                                          'mean_norm': mean_param_norm,
                                                                          'mean_encoder_wproj_actor': mean_enc_wproj_a,
                                                                          'mean_encoder_wproj_critic': mean_enc_wproj_c,
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

        self.actor_critic.redraw_ac_networks_KL_divergence(obs_batch)

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
