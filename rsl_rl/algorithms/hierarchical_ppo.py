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

from rsl_rl.modules import HierarchicalActorCritic
from rsl_rl.storage import HierarchicalRolloutStorage

class HierarchicalPPO:
    actor_critic: HierarchicalActorCritic
    def __init__(self,
                 actor_critic,
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
        self.learning_rate_hl = learning_rate
        self.learning_rate_ll = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer_hl = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate_hl)
        self.optimizer_ll = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate_ll)
        self.transition = HierarchicalRolloutStorage.Transition()

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

    def init_storage(self, num_envs, num_transitions_per_env, 
        actor_obs_hl_shape, critic_obs_hl_shape, action_hl_shape,
        actor_obs_ll_shape, critic_obs_ll_shape, action_ll_shape):
        self.storage = HierarchicalRolloutStorage(num_envs, num_transitions_per_env, 
            actor_obs_hl_shape, critic_obs_hl_shape, action_hl_shape,
            actor_obs_ll_shape, critic_obs_ll_shape, action_ll_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act_hl(self, obs_hl, critic_obs_hl):

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states_hl = self.actor_critic.get_hidden_states_hl()
        # Compute the actions and values
        self.transition.actions_hl = self.actor_critic.act_hl(obs_hl).detach()
        self.transition.values_hl = self.actor_critic.evaluate_hl(critic_obs_hl).detach()
        self.transition.actions_hl_log_prob = self.actor_critic.get_actions_hl_log_prob(self.transition.actions_hl).detach()
        self.transition.action_hl_mean = self.actor_critic.action_hl_mean.detach()
        self.transition.action_hl_sigma = self.actor_critic.action_hl_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations_hl = obs_hl
        self.transition.critic_observations_hl = critic_obs_hl
        return self.transition.actions_hl

    def act_ll(self, obs_ll, critic_obs_ll):

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states_ll = self.actor_critic.get_hidden_states_hl()
        # Compute the actions and values
        self.transition.actions_ll = self.actor_critic.act_ll(obs_ll).detach()
        self.transition.values_ll = self.actor_critic.evaluate_ll(critic_obs_ll).detach()
        self.transition.actions_ll_log_prob = self.actor_critic.get_actions_ll_log_prob(self.transition.actions_ll).detach()
        self.transition.action_ll_mean = self.actor_critic.action_ll_mean.detach()
        self.transition.action_ll_sigma = self.actor_critic.action_ll_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations_ll = obs_ll
        self.transition.critic_observations_ll = critic_obs_ll
        return self.transition.actions_ll
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs_hl, last_critic_obs_ll):
        last_values_hl= self.actor_critic.evaluate_hl(last_critic_obs_hl).detach()
        last_values_ll= self.actor_critic.evaluate_ll(last_critic_obs_ll).detach()
        self.storage.compute_returns(last_values_hl, last_values_ll, self.gamma, self.lam)

    def update(self):
        mean_value_hl_loss = 0
        mean_surrogate_hl_loss = 0
        mean_value_ll_loss = 0
        mean_surrogate_ll_loss = 0

        num_updates = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_hl_batch, critic_obs_hl_batch, actions_hl_batch, target_values_hl_batch, advantages_hl_batch, returns_hl_batch, old_actions_hl_log_prob_batch, \
            old_mu_hl_batch, old_sigma_hl_batch, hid_states_hl_batch, masks_hl_batch, \
            obs_ll_batch, critic_obs_ll_batch, actions_ll_batch, target_values_ll_batch, advantages_ll_batch, returns_ll_batch, old_actions_ll_log_prob_batch, \
            old_mu_ll_batch, old_sigma_ll_batch, hid_states_ll_batch, masks_ll_batch in generator:

                if num_updates % 40 < 20:
                    # High-level update
                    self.actor_critic.act_hl(obs_hl_batch, masks=masks_hl_batch, hidden_states=hid_states_hl_batch[0])
                    actions_hl_log_prob_batch = self.actor_critic.get_actions_hl_log_prob(actions_hl_batch)
                    value_hl_batch = self.actor_critic.evaluate_hl(critic_obs_hl_batch, masks=masks_hl_batch, hidden_states=hid_states_hl_batch[1])
                    mu_hl_batch = self.actor_critic.action_hl_mean
                    sigma_hl_batch = self.actor_critic.action_hl_std
                    entropy_hl_batch = self.actor_critic.entropy_hl

                    # KL
                    if self.desired_kl != None and self.schedule == 'adaptive':
                        with torch.inference_mode():
                            kl_hl = torch.sum(
                                torch.log(sigma_hl_batch / old_sigma_hl_batch + 1.e-5) + (torch.square(old_sigma_hl_batch) + torch.square(old_mu_hl_batch - mu_hl_batch)) / (2.0 * torch.square(sigma_hl_batch)) - 0.5, axis=-1)
                            kl_hl_mean = torch.mean(kl_hl)

                            if kl_hl_mean > self.desired_kl_hl * 2.0:
                                self.learning_rate_hl = max(1e-5, self.learning_rate_hl / 1.5)
                            elif kl_hl_mean < self.desired_kl_hl / 2.0 and kl_hl_mean > 0.0:
                                self.learning_rate_hl = min(1e-2, self.learning_rate_hl * 1.5)
                            
                            for param_group in self.optimizer_hl.param_groups:
                                param_group['lr_hl'] = self.learning_rate_hl


                    # Surrogate loss
                    ratio_hl = torch.exp(actions_hl_log_prob_batch - torch.squeeze(old_actions_hl_log_prob_batch))
                    surrogate_hl = -torch.squeeze(advantages_hl_batch) * ratio_hl
                    surrogate_hl_clipped = -torch.squeeze(advantages_hl_batch) * torch.clamp(ratio_hl, 1.0 - self.clip_param,
                                                                                    1.0 + self.clip_param)
                    surrogate_hl_loss = torch.max(surrogate_hl, surrogate_hl_clipped).mean()

                    # Value function loss
                    if self.use_clipped_value_loss:
                        value_hl_clipped = target_values_hl_batch + (value_hl_batch - target_values_hl_batch).clamp(-self.clip_param,
                                                                                                        self.clip_param)
                        value_hl_losses = (value_hl_batch - returns_hl_batch).pow(2)
                        value_hl_losses_clipped = (value_hl_clipped - returns_hl_batch).pow(2)
                        value_hl_loss = torch.max(value_hl_losses, value_hl_losses_clipped).mean()
                    else:
                        value_hl_loss = (returns_hl_batch - value_hl_batch).pow(2).mean()

                    loss_hl = surrogate_hl_loss + self.value_loss_coef * value_hl_loss - self.entropy_coef * entropy_hl_batch.mean()

                    # Gradient step
                    self.optimizer_hl.zero_grad()
                    loss_hl.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer_hl.step()

                    mean_value_hl_loss += value_hl_loss.item()
                    mean_surrogate_hl_loss += surrogate_hl_loss.item()

                else:
                    # Low-level update
                    self.actor_critic.act_ll(obs_ll_batch, masks=masks_ll_batch, hidden_states=hid_states_ll_batch[0])
                    actions_ll_log_prob_batch = self.actor_critic.get_actions_ll_log_prob(actions_ll_batch)
                    value_ll_batch = self.actor_critic.evaluate_ll(critic_obs_ll_batch, masks=masks_ll_batch, hidden_states=hid_states_ll_batch[1])
                    mu_ll_batch = self.actor_critic.action_ll_mean
                    sigma_ll_batch = self.actor_critic.action_ll_std
                    entropy_ll_batch = self.actor_critic.entropy_ll

                    # KL
                    if self.desired_kl != None and self.schedule == 'adaptive':
                        with torch.inference_mode():
                            kl_ll = torch.sum(
                                torch.log(sigma_ll_batch / old_sigma_ll_batch + 1.e-5) + (torch.square(old_sigma_ll_batch) + torch.square(old_mu_ll_batch - mu_ll_batch)) / (2.0 * torch.square(sigma_ll_batch)) - 0.5, axis=-1)
                            kl_ll_mean = torch.mean(kl_ll)

                            if kl_ll_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate_ll / 1.5)
                            elif kl_ll_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate_ll * 1.5)
                            
                            for param_group in self.optimizer_ll.param_groups:
                                param_group['lr_ll'] = self.learning_rate_ll


                    # Surrogate loss
                    ratio_ll = torch.exp(actions_ll_log_prob_batch - torch.squeeze(old_actions_ll_log_prob_batch))
                    surrogate_ll = -torch.squeeze(advantages_ll_batch) * ratio_ll
                    surrogate_ll_clipped = -torch.squeeze(advantages_ll_batch) * torch.clamp(ratio_ll, 1.0 - self.clip_param,
                                                                                    1.0 + self.clip_param)
                    surrogate_ll_loss = torch.max(surrogate_ll, surrogate_ll_clipped).mean()

                    # Value function loss
                    if self.use_clipped_value_loss:
                        value_ll_clipped = target_values_ll_batch + (value_ll_batch - target_values_ll_batch).clamp(-self.clip_param,
                                                                                                        self.clip_param)
                        value_ll_losses = (value_ll_batch - returns_ll_batch).pow(2)
                        value_ll_losses_clipped = (value_ll_clipped - returns_ll_batch).pow(2)
                        value_ll_loss = torch.max(value_ll_losses, value_ll_losses_clipped).mean()
                    else:
                        value_ll_loss = (returns_ll_batch - value_ll_batch).pow(2).mean()

                    loss_ll = surrogate_ll_loss + self.value_loss_coef * value_ll_loss - self.entropy_coef * entropy_ll_batch.mean()

                    # Gradient step
                    self.optimizer_ll.zero_grad()
                    loss_ll.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    mean_value_ll_loss += value_ll_loss.item()
                    mean_surrogate_ll_loss += surrogate_ll_loss.item()

        # FIXME: split up
        mean_value_loss = mean_value_hl_loss + mean_value_ll_loss
        mean_surrogate_loss = mean_surrogate_hl_loss + mean_surrogate_ll_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
