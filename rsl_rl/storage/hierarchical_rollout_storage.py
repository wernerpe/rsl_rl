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
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class HierarchicalRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations_hl = None
            self.critic_observations_hl = None
            self.actions_hl = None
            self.values_hl = None
            self.actions_hl_log_prob = None
            self.action_hl_mean = None
            self.action_hl_sigma = None
            self.hidden_states_hl = None

            self.observations_ll = None
            self.critic_observations_ll = None
            self.actions_ll = None
            self.values_ll = None
            self.actions_ll_log_prob = None
            self.action_ll_mean = None
            self.action_ll_sigma = None
            self.hidden_states_ll = None

            self.rewards = None
            self.dones = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_hl_shape, privileged_obs_hl_shape, actions_hl_shape, obs_ll_shape, privileged_obs_ll_shape, actions_ll_shape, device='cpu'):

        self.device = device

        self.obs_hl_shape = obs_hl_shape
        self.privileged_obs_hl_shape = privileged_obs_hl_shape
        self.actions_hl_shape = actions_hl_shape

        self.obs_ll_shape = obs_ll_shape
        self.privileged_obs_ll_shape = privileged_obs_ll_shape
        self.actions_ll_shape = actions_ll_shape

        # Core
        self.observations_hl = torch.zeros(num_transitions_per_env, num_envs, *obs_hl_shape, device=self.device)
        if privileged_obs_hl_shape[0] is not None:
            self.privileged_observations_hl = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_hl_shape, device=self.device)
        else:
            self.privileged_observations_hl = None
        self.actions_hl = torch.zeros(num_transitions_per_env, num_envs, *actions_hl_shape, device=self.device)

        self.observations_ll = torch.zeros(num_transitions_per_env, num_envs, *obs_ll_shape, device=self.device)
        if privileged_obs_ll_shape[0] is not None:
            self.privileged_observations_ll = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_ll_shape, device=self.device)
        else:
            self.privileged_observations_ll = None
        self.actions_ll = torch.zeros(num_transitions_per_env, num_envs, *actions_ll_shape, device=self.device)

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_hl_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values_hl = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns_hl = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages_hl = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu_hl = torch.zeros(num_transitions_per_env, num_envs, *actions_hl_shape, device=self.device)
        self.sigma_hl = torch.zeros(num_transitions_per_env, num_envs, *actions_hl_shape, device=self.device)

        self.actions_ll_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values_ll = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns_ll = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages_ll = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu_ll = torch.zeros(num_transitions_per_env, num_envs, *actions_ll_shape, device=self.device)
        self.sigma_ll = torch.zeros(num_transitions_per_env, num_envs, *actions_ll_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations_hl[self.step].copy_(transition.observations_hl)
        if self.privileged_observations_hl is not None: self.privileged_observations_hl[self.step].copy_(transition.critic_observations_hl)
        self.actions_hl[self.step].copy_(transition.actions_hl)
        self.values_hl[self.step].copy_(transition.values_hl)
        self.actions_hl_log_prob[self.step].copy_(transition.actions_hl_log_prob.view(-1, 1))
        self.mu_hl[self.step].copy_(transition.action_hl_mean)
        self.sigma_hl[self.step].copy_(transition.action_hl_sigma)

        self.observations_ll[self.step].copy_(transition.observations_ll)
        if self.privileged_observations_ll is not None: self.privileged_observations_ll[self.step].copy_(transition.critic_observations_ll)
        self.actions_ll[self.step].copy_(transition.actions_ll)
        self.values_ll[self.step].copy_(transition.values_ll)
        self.actions_ll_log_prob[self.step].copy_(transition.actions_ll_log_prob.view(-1, 1))
        self.mu_ll[self.step].copy_(transition.action_ll_mean)
        self.sigma_ll[self.step].copy_(transition.action_ll_sigma)

        self._save_hidden_states(transition.hidden_states)

        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def compute_returns(self, last_values_hl, last_values_ll, gamma, lam):
        advantage_hl = 0
        advantage_ll = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values_hl = last_values_hl
                next_values_ll = last_values_ll
            else:
                next_values_hl = self.values_hl[step + 1]
                next_values_ll = self.values_ll[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()

            delta_hl = self.rewards[step] + next_is_not_terminal * gamma * next_values_hl - self.values_hl[step]
            advantage_hl = delta_hl + next_is_not_terminal * gamma * lam * advantage_hl
            self.returns_hl[step] = advantage_hl + self.values_hl[step]

            delta_ll = self.rewards[step] + next_is_not_terminal * gamma * next_values_ll - self.values_ll[step]
            advantage_ll = delta_ll + next_is_not_terminal * gamma * lam * advantage_ll
            self.returns_ll[step] = advantage_ll + self.values_ll[step]

        # Compute and normalize the advantages
        self.advantages_hl = self.returns_hl - self.values_hl
        self.advantages_hl = (self.advantages_hl - self.advantages_hl.mean()) / (self.advantages_hl.std() + 1e-8)

        self.advantages_ll = self.returns_ll - self.values_ll
        self.advantages_ll = (self.advantages_ll - self.advantages_ll.mean()) / (self.advantages_ll.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations_hl = self.observations_hl.flatten(0, 1)
        if self.privileged_observations_hl is not None:
            critic_observations_hl = self.privileged_observations_hl.flatten(0, 1)
        else:
            critic_observations_hl = observations_hl

        actions_hl = self.actions_hl.flatten(0, 1)
        values_hl = self.values_hl.flatten(0, 1)
        returns_hl = self.returns_hl.flatten(0, 1)
        old_actions_hl_log_prob = self.actions_hl_log_prob.flatten(0, 1)
        advantages_hl = self.advantages_hl.flatten(0, 1)
        old_mu_hl = self.mu_hl.flatten(0, 1)
        old_sigma_hl = self.sigma_hl.flatten(0, 1)

        observations_ll = self.observations_ll.flatten(0, 1)
        if self.privileged_observations_ll is not None:
            critic_observations_ll = self.privileged_observations_ll.flatten(0, 1)
        else:
            critic_observations_ll = observations_ll

        actions_ll = self.actions_ll.flatten(0, 1)
        values_ll = self.values_ll.flatten(0, 1)
        returns_ll = self.returns_ll.flatten(0, 1)
        old_actions_ll_log_prob = self.actions_ll_log_prob.flatten(0, 1)
        advantages_ll = self.advantages_ll.flatten(0, 1)
        old_mu_ll = self.mu_ll.flatten(0, 1)
        old_sigma_ll = self.sigma_ll.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_hl_batch = observations_hl[batch_idx]
                critic_observations_hl_batch = critic_observations_hl[batch_idx]
                actions_hl_batch = actions_hl[batch_idx]
                target_values_hl_batch = values_hl[batch_idx]
                returns_hl_batch = returns_hl[batch_idx]
                old_actions_hl_log_prob_batch = old_actions_hl_log_prob[batch_idx]
                advantages_hl_batch = advantages_hl[batch_idx]
                old_mu_hl_batch = old_mu_hl[batch_idx]
                old_sigma_hl_batch = old_sigma_hl[batch_idx]

                obs_ll_batch = observations_ll[batch_idx]
                critic_observations_ll_batch = critic_observations_ll[batch_idx]
                actions_ll_batch = actions_ll[batch_idx]
                target_values_ll_batch = values_ll[batch_idx]
                returns_ll_batch = returns_ll[batch_idx]
                old_actions_ll_log_prob_batch = old_actions_ll_log_prob[batch_idx]
                advantages_ll_batch = advantages_ll[batch_idx]
                old_mu_ll_batch = old_mu_ll[batch_idx]
                old_sigma_ll_batch = old_sigma_ll[batch_idx]

                yield obs_hl_batch, critic_observations_hl_batch, actions_hl_batch, target_values_hl_batch, advantages_hl_batch, returns_hl_batch, \
                       old_actions_hl_log_prob_batch, old_mu_hl_batch, old_sigma_hl_batch, (None, None), None, \
                      obs_ll_batch, critic_observations_ll_batch, actions_ll_batch, target_values_ll_batch, advantages_ll_batch, returns_ll_batch, \
                       old_actions_ll_log_prob_batch, old_mu_ll_batch, old_sigma_ll_batch, (None, None), None

    # # for RNNs only
    # def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

    #     padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
    #     if self.privileged_observations is not None: 
    #         padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
    #     else: 
    #         padded_critic_obs_trajectories = padded_obs_trajectories

    #     mini_batch_size = self.num_envs // num_mini_batches
    #     for ep in range(num_epochs):
    #         first_traj = 0
    #         for i in range(num_mini_batches):
    #             start = i*mini_batch_size
    #             stop = (i+1)*mini_batch_size

    #             dones = self.dones.squeeze(-1)
    #             last_was_done = torch.zeros_like(dones, dtype=torch.bool)
    #             last_was_done[1:] = dones[:-1]
    #             last_was_done[0] = True
    #             trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
    #             last_traj = first_traj + trajectories_batch_size
                
    #             masks_batch = trajectory_masks[:, first_traj:last_traj]
    #             obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
    #             critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

    #             actions_batch = self.actions[:, start:stop]
    #             old_mu_batch = self.mu[:, start:stop]
    #             old_sigma_batch = self.sigma[:, start:stop]
    #             returns_batch = self.returns[:, start:stop]
    #             advantages_batch = self.advantages[:, start:stop]
    #             values_batch = self.values[:, start:stop]
    #             old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

    #             # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
    #             # then take only time steps after dones (flattens num envs and time dimensions),
    #             # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
    #             last_was_done = last_was_done.permute(1, 0)
    #             hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
    #                             for saved_hidden_states in self.saved_hidden_states_a ] 
    #             hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
    #                             for saved_hidden_states in self.saved_hidden_states_c ]
    #             # remove the tuple for GRU
    #             hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
    #             hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

    #             yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
    #                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch
                
    #             first_traj = last_traj