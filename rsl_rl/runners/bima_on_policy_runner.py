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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import BimaPPO, BimaDecSARSA
# from rsl_rl.modules import BilevelActorCriticAttention, get_encoder
from rsl_rl.attention import get_encoder
from rsl_rl.modules import MultiTeamBilevelActorCritic, TeamBilevelActorCritic, MultiTeamBilevelDecCritic
from rsl_rl.env import VecEnv

import yaml
import os

from torch.nn.utils import parameters_to_vector
import copy


class BimaOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 num_agents=2):
        self.train_cfg = train_cfg
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        
        self.num_actions_hl = self.env.num_actions_hl
        self.num_obs_add_ll = self.env.num_obs_add_ll
        self.dt_hl = self.env.dt_hl
        self.iters_ado_ppc = env.iters_ado_ppc

        self.iter_per_hl = self.cfg["iter_per_hl"]
        self.iter_per_ll = self.cfg["iter_per_ll"]

        self.pretrain_ll_iter = self.cfg["pretrain_ll_iter"]
        self.warmup_hl_iter = self.cfg["warmup_hl_iter"]
        self.warmup_hl_ini = self.cfg["warmup_hl_ini"]
        self.warmup_hl_iter = min(self.warmup_hl_iter, self.pretrain_ll_iter)

        act_min = self.env.action_min_hl[:self.num_actions_hl]
        act_max = self.env.action_max_hl[:self.num_actions_hl]
        act_ini = self.env.action_ini_hl[:self.num_actions_hl]
        act_all = self.env.action_all_hl[:self.num_actions_hl]

        teamsize = self.policy_cfg['teamsize']
        numteams = self.policy_cfg['numteams']
        self.num_agents = teamsize * numteams

        # num_ego_obs = self.policy_cfg['num_ego_obs']
        # num_ado_obs = self.policy_cfg['num_ado_obs']

        encoder_type = self.policy_cfg['encoder_type']
        if encoder_type=='identity':
            self.policy_cfg['encoder_embed_dims'] = [-1]
            self.policy_cfg['encoder_attend_dims'] = [self.env.num_obs]
        encoder_embed_dims = self.policy_cfg['encoder_embed_dims']
        encoder_attend_dims = self.policy_cfg['encoder_attend_dims']

        encoders = [get_encoder(
            num_ego_obs=self.policy_cfg['num_ego_obs'], 
            num_ado_obs=self.policy_cfg['num_ado_obs'], 
            embed_dims=encoder_embed_dims,
            attend_dims=encoder_attend_dims, 
            teamsize=teamsize,
            numteams=numteams,
            encoder_type=encoder_type,
        )]
        for _ in range(numteams-1):
            encoders.append(copy.deepcopy(encoders[0]))

        actor_critic_class_hl = eval(self.cfg["policy_class_hl_name"]) # BilevelActorCritic
        actor_critic_hl: MultiTeamBilevelActorCritic = actor_critic_class_hl(
                                                        num_actor_obs=self.env.num_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_add_obs=0,
                                                        num_actions=self.num_actions_hl,
                                                        encoders=encoders,
                                                        train_encoder=False,
                                                        act_min=act_min,
                                                        act_max=act_max,
                                                        act_ini=act_ini,
                                                        act_all=act_all,
                                                        device=device,
                                                        discrete=True,
                                                        **self.policy_cfg).to(self.device)
        actor_critic_class_ll = eval(self.cfg["policy_class_ll_name"]) # ActorCritic
        actor_critic_ll: MultiTeamBilevelActorCritic = actor_critic_class_ll(
                                                        num_actor_obs=self.env.num_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_add_obs=self.num_obs_add_ll,
                                                        num_actions=self.env.num_actions,
                                                        encoders=encoders,
                                                        train_encoder=True,
                                                        device=device,
                                                        **self.policy_cfg).to(self.device)
        alg_class_hl = eval(self.cfg["algorithm_class_hl_name"]) # BilevelPPO
        self.alg_hl: BimaPPO = alg_class_hl(actor_critic_hl, centralized_value=True, device=self.device, schedule="adaptive", clip_param=0.2, entropy_coef=0.001, gamma=0.90, **self.alg_cfg)
        alg_class_ll = eval(self.cfg["algorithm_class_ll_name"]) # BilevelPPO
        self.alg_ll: BimaPPO = alg_class_ll(actor_critic_ll, centralized_value=False, device=self.device, schedule="adaptive", clip_param=0.2, entropy_coef=0.002, gamma=0.99, **self.alg_cfg)

        self.num_steps_per_env_hl = self.cfg["num_steps_per_env_hl"]
        self.num_steps_per_env_ll = self.cfg["num_steps_per_env_ll"]

        self.num_steps_hl_per_ll_update = int(self.num_steps_per_env_ll / self.dt_hl)

        self.save_interval = self.cfg["save_interval"]

        self.population_update_interval = self.cfg["population_update_interval"]

        # init storage and model
        self.alg_hl.init_storage(self.env.num_envs, self.num_steps_per_env_hl, [self.env.num_obs], [self.env.num_privileged_obs], [self.num_actions_hl], teamsize)
        if self.env.num_privileged_obs is None:
            num_privileged_obs_ll = self.env.num_privileged_obs
        else:
            num_privileged_obs_ll = self.env.num_privileged_obs + self.num_obs_add_ll
        self.alg_ll.init_storage(self.env.num_envs, self.num_steps_per_env_ll, [self.env.num_obs + self.num_obs_add_ll], [num_privileged_obs_ll], [self.env.num_actions], teamsize)

        # Log
        self.log_dir = log_dir
        if log_dir is not None:
          os.makedirs(self.log_dir + '/hl_model', exist_ok=True)
          os.makedirs(self.log_dir + '/ll_model', exist_ok=True)
        self.track_cfg = train_cfg['track_cfgs']
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        # Re-init as reset & step increment total_step via post_physics_step
        self.env.total_step = 0
        self.env.episode_length_buf *= 0.0

    def store_cfgs(self,):
                with open(self.log_dir + '/cfg_train.yml', 'w') as outfile:
                    yaml.dump(self.train_cfg, outfile, default_flow_style=False)
                    
                with open(self.log_dir + '/cfg.yml', 'w') as outfile:
                    yaml.dump(self.env.cfg, outfile, default_flow_style=False)

    def reset_environment_for_training(self):
        with torch.inference_mode():
          _, _ = self.env.reset()
          self.env.total_step = 0
          self.env.episode_length_buf *= 0.0
          obs, privileged_obs = self.env.get_observations(), self.env.get_privileged_observations()
          critic_obs = privileged_obs if privileged_obs is not None else obs
          obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
          return obs, critic_obs
      
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):

        iter_per_ll = self.iter_per_ll  # 200
        iter_per_hl = self.iter_per_hl  # 20
        iter_switch = [0]
        while iter_switch[-1] < num_learning_iterations:
            iter_switch.append(iter_switch[-1] + iter_per_ll)
            iter_switch.append(iter_switch[-1] + iter_per_hl)

        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            if self.track_cfg:
                self.store_cfgs()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg_hl.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg_ll.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        behavior_infos = []
        observation_infos = []
        rewbuffer_hl = deque(maxlen=100)
        trewbuffer_hl = deque(maxlen=100)
        lenbuffer_hl = deque(maxlen=100)
        cur_mean_reward_sum_hl = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_mean_team_reward_sum_hl = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length_hl = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        rewbuffer_ll = deque(maxlen=100)
        trewbuffer_ll = deque(maxlen=100)
        lenbuffer_ll = deque(maxlen=100)
        cur_mean_reward_sum_ll = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_mean_team_reward_sum_ll = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length_ll = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        log_param_abs_delta = False
        param_vec_ll_actor_mean_diff, param_vec_ll_critic_mean_diff, param_vec_ll_encoder_mean_diff = None, None, None
        param_vec_hl_actor_mean_diff, param_vec_hl_critic_mean_diff, param_vec_hl_encoder_mean_diff = None, None, None
        if log_param_abs_delta:
            # Low-level
            prev_param_vec_ll_actor = parameters_to_vector(self.alg_ll.actor_critic.teamacs[0].ac.actor._network.parameters()).detach()
            prev_param_vec_ll_critic = parameters_to_vector(self.alg_ll.actor_critic.teamacs[0].ac.critics[0]._network.parameters()).detach()
            prev_param_vec_ll_encoder = parameters_to_vector(self.alg_ll.actor_critic.teamacs[0].ac.actor._encoder.parameters()).detach()
            # High-level
            prev_param_vec_hl_actor = parameters_to_vector(self.alg_hl.actor_critic.teamacs[0].ac.actor._network.parameters()).detach()
            prev_param_vec_hl_critic = parameters_to_vector(self.alg_hl.actor_critic.teamacs[0].ac.critics[0]._network.parameters()).detach()
            prev_param_vec_hl_encoder = parameters_to_vector(self.alg_hl.actor_critic.teamacs[0].ac.actor._encoder.parameters()).detach()


        # Start with low-level training
        train_hl = False
        train_ll = True

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):

            self.env.set_dropout_prob(it)
            self.env.set_video_log_ep(it)
            self.env.set_ppc_prob(it)

            # if it > self.iters_ado_ppc:
            #     self.env.set_steer_ado_ppc(False)

            if it in iter_switch and it > self.pretrain_ll_iter:
                train_ll = not train_ll
                train_hl = not train_hl
                obs, critic_obs = self.reset_environment_for_training()

            if train_ll:
                # ### Low-level Training ###
                self.alg_hl.actor_critic.eval()
                self.alg_ll.actor_critic.train()
                self.env.set_train_level(low_level=True)
                self.env.set_reset_agent_ids(reset_id_num=1)

                start_ll = time.time()

                goal_pos_scale = self.warmup_hl_ini + it / (1.0 + self.warmup_hl_iter) * (1.0 - self.warmup_hl_ini)
                goal_pos_scale = min(goal_pos_scale, 1.0)

                
                # Rollout
                with torch.inference_mode():
                    for i in range(self.num_steps_per_env_ll):
                        # Sample new HL target at every step and decide whether to update internally
                        actions_hl_raw = self.alg_hl.act(obs, critic_obs)
                        actions_hl_raw[..., :2] *= goal_pos_scale  # allow for warmup


                        # if it < self.warmup_hl_iter:
                        #     actions_hl_raw[..., 2:4] = 0.3 + 0.0*actions_hl_raw[..., 2:4]

                        # # TODO: remove this after testing --> select middle uncertainty bin
                        # actions_hl_raw[..., 0] = 4.00 + 0.0*actions_hl_raw[..., 0]
                        # actions_hl_raw[..., 1] = 0.00 + 0.0*actions_hl_raw[..., 1]
                        # actions_hl_raw[..., 2] = 0.30 + 0.0*actions_hl_raw[..., 2]
                        # actions_hl_raw[..., 3] = 0.20 + 0.0*actions_hl_raw[..., 3]


                        self.env.set_hl_action_probs(self.alg_hl.actor_critic.action_probs)
                        actions_hl = self.env.project_into_track_frame(actions_hl_raw)
                        
                        obs_ll = torch.concat((obs, actions_hl), dim=-1)
                        critic_obs_ll = torch.concat((critic_obs, actions_hl), dim=-1)
                        # obs_ll = obs
                        # critic_obs_ll = critic_obs
                        actions_ll = self.alg_ll.act(obs_ll, critic_obs_ll)
                        self.env.set_ll_action_stats(self.alg_ll.actor_critic.action_mean, self.alg_ll.actor_critic.action_std)
                        self.env.set_ll_values_preds(self.alg_ll.transition.values.squeeze(dim=1))

                        # self.env.viewer.update_attention(self.alg_ll.actor_critic.teamacs[0].ac.actor._encoder.attention_weights)

                        obs, privileged_obs, rewards, dones, infos = self.env.step(actions_ll)
                                                
                        critic_obs = privileged_obs if privileged_obs is not None else obs
                        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                        if rewards.shape[-1]==2:
                            rewards = rewards[:, :, 0].unsqueeze(dim=-1)
                        self.alg_ll.process_env_step(rewards, dones, infos)
                        
                        if self.log_dir is not None:

                            # Book keeping
                            if 'episode' in infos:
                                infos["episode"]["rewards_ll"] = infos["episode"]["rewards"]
                                ep_infos.append(infos['episode'])
                            if 'behavior' in infos:
                                behavior_infos.append(infos['behavior'])
                            if 'observations' in infos:
                                observation_infos.append(infos['observations'])
                            cur_mean_reward_sum_ll += torch.sum(torch.mean(rewards[:, self.alg_ll.actor_critic.teams[0], :], dim = 1), dim = 1)
                            # cur_mean_team_reward_sum_ll += torch.mean(rewards[:, self.alg_ll.teams[0], 1], dim = 1)
                            cur_episode_length_ll += 1
                            new_ids_ll = torch.cumsum(dones > 0, axis=-1)[..., self.env.reset_ids].nonzero(as_tuple=False)
                            rewbuffer_ll.extend(cur_mean_reward_sum_ll[new_ids_ll][:, 0].cpu().numpy().tolist())
                            # trewbuffer_ll.extend(cur_mean_team_reward_sum_ll[new_ids_ll][:, 0].cpu().numpy().tolist())
                            lenbuffer_ll.extend(cur_episode_length_ll[new_ids_ll][:, 0].cpu().numpy().tolist())
                            cur_mean_reward_sum_ll[new_ids_ll] = 0
                            # cur_mean_team_reward_sum_ll[new_ids_ll] = 0
                            cur_episode_length_ll[new_ids_ll] = 0


                        self.alg_ll.actor_critic.update_ac_ratings(infos)  # NOTE: self-play

                    stop_ll = time.time()
                    collection_time_ll = stop_ll - start_ll

                    # Learning step
                    start_ll = stop_ll
                    self.alg_ll.compute_returns(critic_obs_ll)
                                
                mean_value_loss_ll, mean_surrogate_loss_ll, mean_entropy_loss_ll, mean_stats_ll = self.alg_ll.update()
                                
                if  it % self.population_update_interval == 0:  # NOTE: self-play
                    self.alg_ll.update_population()
                                
                stop_ll = time.time()
                learn_time_ll = stop_ll - start_ll

                if log_param_abs_delta:
                    # Low-level
                    param_vec_ll_actor = parameters_to_vector(self.alg_ll.actor_critic.teamacs[0].ac.actor._network.parameters()).detach()
                    param_vec_ll_critic = parameters_to_vector(self.alg_ll.actor_critic.teamacs[0].ac.critics[0]._network.parameters()).detach()
                    param_vec_ll_encoder = parameters_to_vector(self.alg_ll.actor_critic.teamacs[0].ac.actor._encoder.parameters()).detach()

                    param_vec_ll_actor_mean_diff = (param_vec_ll_actor - prev_param_vec_ll_actor).abs().mean()
                    param_vec_ll_critic_mean_diff = (param_vec_ll_critic - prev_param_vec_ll_critic).abs().mean()
                    param_vec_ll_encoder_mean_diff = (param_vec_ll_encoder - prev_param_vec_ll_encoder).abs().mean()

                    prev_param_vec_ll_actor = param_vec_ll_actor
                    prev_param_vec_ll_critic = param_vec_ll_critic
                    prev_param_vec_ll_encoder = param_vec_ll_encoder

                    # High-level
                    param_vec_hl_actor = parameters_to_vector(self.alg_hl.actor_critic.teamacs[0].ac.actor._network.parameters()).detach()
                    param_vec_hl_critic = parameters_to_vector(self.alg_hl.actor_critic.teamacs[0].ac.critics[0]._network.parameters()).detach()
                    param_vec_hl_encoder = parameters_to_vector(self.alg_hl.actor_critic.teamacs[0].ac.actor._encoder.parameters()).detach()

                    param_vec_hl_actor_mean_diff = (param_vec_hl_actor - prev_param_vec_hl_actor).abs().mean()
                    param_vec_hl_critic_mean_diff = (param_vec_hl_critic - prev_param_vec_hl_critic).abs().mean()
                    param_vec_hl_encoder_mean_diff = (param_vec_hl_encoder - prev_param_vec_hl_encoder).abs().mean()

                    prev_param_vec_hl_actor = param_vec_hl_actor
                    prev_param_vec_hl_critic = param_vec_hl_critic
                    prev_param_vec_hl_encoder = param_vec_hl_encoder

                if self.log_dir is not None:
                    self.log(locals(), self.num_steps_per_env_ll, log_ll=True)
                # if it % self.save_interval == 0:
                #     self.save_ll(os.path.join(self.log_dir, 'll_model', 'model_{}.pt'.format(it)))
                # ep_infos.clear()

            else:
                # ### High-level Training ###
                self.alg_hl.actor_critic.train()
                self.alg_ll.actor_critic.eval()
                self.env.set_train_level(high_level=True)
                self.env.set_reset_agent_ids(reset_id_num=self.env.team_size)

                start_hl = time.time()
                
                # Rollout
                with torch.inference_mode(): 
                    for i_hl in range(self.num_steps_per_env_hl):
                        actions_hl_raw = self.alg_hl.act(obs, critic_obs)
                        self.env.set_hl_action_probs(self.alg_hl.actor_critic.action_probs)
                        reward_ep_ll = torch.zeros(self.env.num_envs, self.num_agents, 1, dtype=torch.float, device=self.device)
                        for i_ll in range(self.dt_hl):
                            actions_hl = self.env.project_into_track_frame(actions_hl_raw)
                            obs_ll = torch.concat((obs, actions_hl), dim=-1)
                            critic_obs_ll = torch.concat((critic_obs, actions_hl), dim=-1)
                            actions_ll = self.alg_ll.act(obs_ll, critic_obs_ll)

                            # self.env.viewer.update_attention(self.alg_ll.actor_critic.teamacs[0].ac.actor._encoder.attention_weights)

                            self.env.set_ll_action_stats(self.alg_ll.actor_critic.action_mean, self.alg_ll.actor_critic.action_std)
                            obs, privileged_obs, rewards, dones, infos = self.env.step(actions_ll)
                            critic_obs = privileged_obs if privileged_obs is not None else obs
                            obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                            if rewards.shape[-1]==2:
                                rewards = rewards[:, :, 0].unsqueeze(dim=-1)
                            reward_ep_ll += rewards  # .squeeze()
                        
                        if 'PPO' in self.cfg["algorithm_class_hl_name"]:
                            self.alg_hl.process_env_step(reward_ep_ll, dones, infos)
                        elif 'SARSA' in self.cfg["algorithm_class_hl_name"]:
                            self.alg_hl.process_env_step(reward_ep_ll, obs, dones, infos)

                        if self.log_dir is not None:

                            # Book keeping
                            if 'episode' in infos:
                                infos["episode"]["rewards_hl"] = infos["episode"]["rewards"]
                                ep_infos.append(infos['episode'])
                            if 'behavior' in infos:
                                behavior_infos.append(infos['behavior'])
                            if 'observations' in infos:
                                observation_infos.append(infos['observations'])
                            cur_mean_reward_sum_hl += torch.sum(torch.mean(reward_ep_ll[:, self.alg_ll.actor_critic.teams[0], :], dim = 1), dim = 1)
                            cur_episode_length_hl += self.dt_hl
                            new_ids_hl = torch.cumsum(dones > 0, axis=-1)[..., self.env.reset_ids].nonzero(as_tuple=False)
                            rewbuffer_hl.extend(cur_mean_reward_sum_hl[new_ids_hl][:, 0].cpu().numpy().tolist())
                            lenbuffer_hl.extend(cur_episode_length_hl[new_ids_hl][:, 0].cpu().numpy().tolist())
                            cur_mean_reward_sum_hl[new_ids_hl] = 0
                            cur_episode_length_hl[new_ids_hl] = 0

                        self.alg_hl.actor_critic.update_ac_ratings(infos)  # NOTE: self-play

                    stop_hl = time.time()
                    collection_time_hl = stop_hl - start_hl

                    # Learning step
                    start_hl = stop_hl
                    self.alg_hl.compute_returns(critic_obs)
                
                mean_value_loss_hl, mean_surrogate_loss_hl, mean_entropy_loss_hl, mean_stats_hl = self.alg_hl.update()
                if  it % self.population_update_interval == 0:  # NOTE: self-play
                    self.alg_hl.update_population()
                stop_hl = time.time()
                learn_time_hl = stop_hl - start_hl
                if self.log_dir is not None:
                    self.log(locals(), self.num_steps_per_env_hl, log_ll=False)

            if it % self.save_interval == 0:
                self.save_ll(os.path.join(self.log_dir, 'll_model', 'model_{}.pt'.format(it)))
                self.save_hl(os.path.join(self.log_dir, 'hl_model', 'model_{}.pt'.format(it)))
            ep_infos.clear()
            behavior_infos.clear()
        
        
        self.current_learning_iteration += num_learning_iterations
        self.save_hl(os.path.join(self.log_dir, 'hl_model', 'model_{}.pt'.format(self.current_learning_iteration)))
        self.save_ll(os.path.join(self.log_dir, 'll_model', 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, steps_per_env, log_ll, width=80, pad=35):
        self.tot_timesteps += steps_per_env * self.env.num_envs
        if not log_ll:
          tot_time = locs['collection_time_hl'] + locs['learn_time_hl']
        else:
          tot_time = locs['collection_time_ll'] + locs['learn_time_ll']
        self.tot_time += tot_time
        iteration_time = tot_time


        def write_tf_data(name, field, string):
            if locs[name]:
                for key in locs[name][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for info in locs[name]:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(info[key], torch.Tensor):
                            info[key] = torch.Tensor([info[key]])
                        if len(info[key].shape) == 0:
                            info[key] = info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    self.writer.add_scalar(field + '/' + key, value, locs['it'])
                    string += f"""{f'Mean ' + field + f' {key}:':>{pad}} {value:.4f}\n"""
            return string

        ep_string = f''
        ep_string = write_tf_data('ep_infos', 'Episode', ep_string)
        _ = write_tf_data('behavior_infos', 'Behavior', ep_string)
        _ = write_tf_data('observation_infos', 'Observations', ep_string)
        # if locs['ep_infos']:
        #     for key in locs['ep_infos'][0]:
        #         infotensor = torch.tensor([], device=self.device)
        #         for ep_info in locs['ep_infos']:
        #             # handle scalar and zero dimensional tensor infos
        #             if not isinstance(ep_info[key], torch.Tensor):
        #                 ep_info[key] = torch.Tensor([ep_info[key]])
        #             if len(ep_info[key].shape) == 0:
        #                 ep_info[key] = ep_info[key].unsqueeze(0)
        #             infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        #         value = torch.mean(infotensor)
        #         self.writer.add_scalar('Episode/' + key, value, locs['it'])
        #         ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        # if locs['behavior_infos']:
        #     for key in locs['behavior_infos'][0]:
        #         infotensor = torch.tensor([], device=self.device)
        #         for behavior_info in locs['behavior_infos']:
        #             # handle scalar and zero dimensional tensor infos
        #             if not isinstance(behavior_info[key], torch.Tensor):
        #                 behavior_info[key] = torch.Tensor([behavior_info[key]])
        #             if len(behavior_info[key].shape) == 0:
        #                 behavior_info[key] = behavior_info[key].unsqueeze(0)
        #             infotensor = torch.cat((infotensor, behavior_info[key].to(self.device)))
        #         value = torch.mean(infotensor)
        #         self.writer.add_scalar('Behavior/' + key, value, locs['it'])
          
        if locs['param_vec_ll_actor_mean_diff'] is not None:
            self.writer.add_scalar('Params/' + 'll_delta_actor', locs['param_vec_ll_actor_mean_diff'].unsqueeze(0).to(self.device), locs['it'])
            self.writer.add_scalar('Params/' + 'll_delta_critic', locs['param_vec_ll_critic_mean_diff'].unsqueeze(0).to(self.device), locs['it'])
            self.writer.add_scalar('Params/' + 'll_delta_encoder', locs['param_vec_ll_encoder_mean_diff'].unsqueeze(0).to(self.device), locs['it'])
        if locs['param_vec_hl_actor_mean_diff'] is not None:
            self.writer.add_scalar('Params/' + 'hl_delta_actor', locs['param_vec_hl_actor_mean_diff'].unsqueeze(0).to(self.device), locs['it'])
            self.writer.add_scalar('Params/' + 'hl_delta_critic', locs['param_vec_hl_critic_mean_diff'].unsqueeze(0).to(self.device), locs['it'])
            self.writer.add_scalar('Params/' + 'hl_delta_encoder', locs['param_vec_hl_encoder_mean_diff'].unsqueeze(0).to(self.device), locs['it'])
        fps = int(steps_per_env * self.env.num_envs / (tot_time))

        if not log_ll:
          mean_std_hl = self.alg_hl.actor_critic.action_std[:, 0].mean(dim=0)
          self.writer.add_scalar('Loss/value_function_hl', locs['mean_value_loss_hl'], locs['it'])
          self.writer.add_scalar('Loss/surrogate_hl', locs['mean_surrogate_loss_hl'], locs['it'])
          self.writer.add_scalar('Loss/entropy_hl', locs['mean_entropy_loss_hl'], locs['it'])
          self.writer.add_scalar('Loss/learning_rate_hl', self.alg_hl.learning_rate, locs['it'])
          for idx, mean_std in enumerate(mean_std_hl):
              self.writer.add_scalar('Policy/mean_noise_std_hl_' + str(idx), mean_std.item(), locs['it'])
          self.writer.add_scalar('Perf/collection time_hl', locs['collection_time_hl'], locs['it'])
          self.writer.add_scalar('Perf/learning_time_hl', locs['learn_time_hl'], locs['it'])
          for key, value in locs['mean_stats_hl'].items():
              self.writer.add_scalar('Loss/' + key + '_hl', value, locs['it'])
        else:
          mean_std_ll = self.alg_ll.actor_critic.action_std[:, 0].mean(dim=0)
          self.writer.add_scalar('Loss/value_function_ll', locs['mean_value_loss_ll'], locs['it'])
          self.writer.add_scalar('Loss/surrogate_ll', locs['mean_surrogate_loss_ll'], locs['it'])
          self.writer.add_scalar('Loss/entropy_ll', locs['mean_entropy_loss_ll'], locs['it'])
          self.writer.add_scalar('Loss/learning_rate_ll', self.alg_ll.learning_rate, locs['it'])
          for idx, mean_std in enumerate(mean_std_ll):
              self.writer.add_scalar('Policy/mean_noise_std_ll_' + str(idx), mean_std.item(), locs['it'])
          self.writer.add_scalar('Perf/collection time_ll', locs['collection_time_ll'], locs['it'])
          self.writer.add_scalar('Perf/learning_time_ll', locs['learn_time_ll'], locs['it'])
          for key, value in locs['mean_stats_ll'].items():
              self.writer.add_scalar('Loss/' + key + '_ll', value, locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        if len(locs['rewbuffer_hl']) > 0:
            self.writer.add_scalar('Train/mean_reward_hl', statistics.mean(locs['rewbuffer_hl']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length_hl', statistics.mean(locs['lenbuffer_hl']), locs['it'])
            self.writer.add_scalar('Train/mean_reward_hl/time', statistics.mean(locs['rewbuffer_hl']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length_hl/time', statistics.mean(locs['lenbuffer_hl']), self.tot_time)
        if len(locs['rewbuffer_ll']) > 0:
            self.writer.add_scalar('Train/mean_reward_ll', statistics.mean(locs['rewbuffer_ll']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length_ll', statistics.mean(locs['lenbuffer_ll']), locs['it'])
            self.writer.add_scalar('Train/mean_reward_ll/time', statistics.mean(locs['rewbuffer_ll']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length_ll/time', statistics.mean(locs['lenbuffer_ll']), self.tot_time)

        str_print = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if not log_ll:
          if len(locs['rewbuffer_hl']) > 0:
              log_string = (f"""{'#' * width}\n"""
                            f"""{str_print.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time_hl']:.3f}s, learning {locs['learn_time_hl']:.3f}s)\n"""
                            f"""{'HL Value function loss:':>{pad}} {locs['mean_value_loss_hl']:.4f}\n"""
                            f"""{'HL Surrogate loss:':>{pad}} {locs['mean_surrogate_loss_hl']:.4f}\n"""
                            f"""{'HL Entropy loss:':>{pad}} {locs['mean_entropy_loss_hl']:.4f}\n""")
              for idx, mean_std in enumerate(mean_std_hl):
                log_string+=(f"""{'HL Mean action ' + str(idx) + ' noise std:':>{pad}} {mean_std.item():.2f}\n""")
              log_string += (f"""{'HL Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer_hl']):.2f}\n"""
                            f"""{'HL Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer_hl']):.2f}\n""")
                          #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
          else:
              log_string = (f"""{'#' * width}\n"""
                            f"""{str_print.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time_hl']:.3f}s, learning {locs['learn_time_hl']:.3f}s)\n"""
                            f"""{'HL Value function loss:':>{pad}} {locs['mean_value_loss_hl']:.4f}\n"""
                            f"""{'HL Surrogate loss:':>{pad}} {locs['mean_surrogate_loss_hl']:.4f}\n"""
                            f"""{'HL Entropy loss:':>{pad}} {locs['mean_entropy_loss_hl']:.4f}\n""")
              for idx, mean_std in enumerate(mean_std_hl):
                log_string+=(f"""{'HL Mean action ' + str(idx) + ' noise std:':>{pad}} {mean_std.item():.2f}\n""")
                          #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
          if len(locs['rewbuffer_ll']) > 0:
              log_string = (f"""{'#' * width}\n"""
                            f"""{str_print.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time_ll']:.3f}s, learning {locs['learn_time_ll']:.3f}s)\n"""
                            f"""{'LL Value function loss:':>{pad}} {locs['mean_value_loss_ll']:.4f}\n"""
                            f"""{'LL Surrogate loss:':>{pad}} {locs['mean_surrogate_loss_ll']:.4f}\n"""
                            f"""{'LL Entropy loss:':>{pad}} {locs['mean_entropy_loss_ll']:.4f}\n""")
              for idx, mean_std in enumerate(mean_std_ll):
                log_string+=(f"""{'LL Mean action ' + str(idx) + ' noise std:':>{pad}} {mean_std.item():.2f}\n""")
              log_string += (f"""{'LL Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer_ll']):.2f}\n"""
                            f"""{'LL Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer_ll']):.2f}\n""")
                          #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
          else:
              log_string = (f"""{'#' * width}\n"""
                            f"""{str_print.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time_ll']:.3f}s, learning {locs['learn_time_ll']:.3f}s)\n"""
                            f"""{'LL Value function loss:':>{pad}} {locs['mean_value_loss_ll']:.4f}\n"""
                            f"""{'LL Surrogate loss:':>{pad}} {locs['mean_surrogate_loss_ll']:.4f}\n"""
                            f"""{'LL Entropy loss:':>{pad}} {locs['mean_entropy_loss_ll']:.4f}\n""")
              for idx, mean_std in enumerate(mean_std_ll):
                log_string+=(f"""{'LL Mean action ' + str(idx) + ' noise std:':>{pad}} {mean_std.item():.2f}\n""")
                          #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save_hl(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg_hl.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg_hl.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)
    
    def save_ll(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg_ll.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg_ll.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load_hl(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg_hl.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg_hl.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def load_ll(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg_ll.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg_ll.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']
    
    def load(self, path_ll, path_hl, load_optimizer = True):
        self.load_ll(path_ll, load_optimizer= load_optimizer)
        self.load_hl(path_hl, load_optimizer= load_optimizer)

    def load_multi_path(self, paths_hl, paths_ll, load_optimizer=True):
        model_dicts_hl, model_dicts_ll = [], []
        opt_dicts_hl, opt_dicts_ll = [], []
        dicts_hl, dicts_ll = [], []
        for path_hl, path_ll in zip(paths_hl, paths_ll):
            # HL
            dict_hl = torch.load(path_hl)
            dicts_hl.append(dict_hl)
            model_dicts_hl.append(dict_hl['model_state_dict'])
            opt_dicts_hl.append(dict_hl['optimizer_state_dict'])
            # LL
            dict_ll = torch.load(path_ll)
            dicts_ll.append(dict_ll)
            model_dicts_ll.append(dict_ll['model_state_dict'])
            opt_dicts_ll.append(dict_ll['optimizer_state_dict'])
        self.alg_hl.actor_critic.load_multi_state_dict(model_dicts_hl)
        self.alg_ll.actor_critic.load_multi_state_dict(model_dicts_ll)
        if load_optimizer:
            self.alg_hl.optimizer.load_state_dict(opt_dicts_hl[0])
            self.alg_ll.optimizer.load_state_dict(opt_dicts_ll[0])
        self.current_learning_iteration = dicts_hl[0]['iter']

        # #add loaded models to buffers and set current ratings
        # infos = []
        # self.alg.actor_critic.past_models = model_dicts
        # # self.alg.actor_critic.agentratings = []
        # # self.alg.actor_critic.past_ratings_mu = []
        # # self.alg.actor_critic.past_ratings_sigma = []
        # for dict in dicts:
        #     info = dict['infos']
        #     # mu = info['trueskill']['mu']
        #     # sigma = info['trueskill']['sigma']
        #     # active_rating = self.alg.actor_critic.new_rating(mu, sigma)
        #     # self.alg.actor_critic.agentratings.append(active_rating)
        #     # self.alg.actor_critic.past_ratings_mu.append(mu)
        #     # self.alg.actor_critic.past_ratings_sigma.append(sigma)
        #     infos.append(info)
        # return infos
    
    def populate_adversary_buffer(self, paths_hl, paths_ll):
        for path_hl, path_ll in zip(paths_hl, paths_ll):
            # HL
            dict_hl = torch.load(path_hl)
            info_hl = dict_hl['infos']
            #mu = info['trueskill']['mu']
            #sigma = info['trueskill']['sigma']
            #self.alg.actor_critic.past_ratings_mu.append(mu)
            #self.alg.actor_critic.past_ratings_sigma.append(sigma)
            self.alg_hl.actor_critic.past_models.append(dict_hl['model_state_dict'])
            # LL
            dict_ll = torch.load(path_ll)
            info_ll = dict_hl['infos']
            #mu = info['trueskill']['mu']
            #sigma = info['trueskill']['sigma']
            #self.alg.actor_critic.past_ratings_mu.append(mu)
            #self.alg.actor_critic.past_ratings_sigma.append(sigma)
            self.alg_ll.actor_critic.past_models.append(dict_ll['model_state_dict'])
        
    def get_inference_policy_hl(self, device=None):
        self.alg_hl.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg_hl.actor_critic.to(device)
        return self.alg_hl.actor_critic.act_inference

    def get_inference_policy_ll(self, device=None):
        self.alg_ll.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg_ll.actor_critic.to(device)
        return self.alg_ll.actor_critic.act_inference
