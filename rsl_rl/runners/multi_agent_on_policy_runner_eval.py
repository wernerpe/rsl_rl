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

from rsl_rl.algorithms import PPO, IMAPPO, JRMAPPO
from rsl_rl.modules import MAActorCritic, MultiTeamCMAAC
from rsl_rl.env import VecEnv
import yaml
import os

class MAOnPolicyRunner:
    actor_critic_class: MultiTeamCMAAC
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 num_agents = 2):
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
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        if self.cfg["algorithm_class_name"] == 'JRMAPPO' and self.cfg["policy_class_name"] != 'MultiTeamCMAAC':
            raise ValueError("Please use MultiTeamCMAAC in combination with Joint-Ratio Multi Agent PPO")
        elif self.cfg["algorithm_class_name"] == 'IMAPPO' and self.cfg["policy_class_name"] != 'MAActorCritic':
            raise ValueError("Please use MAActorCritic in combination with Independent Multi Agent PPO")

        actor_critic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        num_agents,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) 
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.population_update_interval = self.cfg["population_update_interval"]
        self.save_interval = self.cfg["save_interval"]

        self.max_epsiode_len_eval = self.env.max_episode_length
        self.eval_interval_steps = self.cfg["eval_interval"]

        # self.attention_network = self.alg.actor_critic.teamacs[0].ac.encoder._network
        # self.attention_tensor = torch.zeros((env.num_envs, env.num_agents-1))
        #self.num_ego_obs = self.alg.actor_critic.teamacs[0].ac.encoder.num_ego_obs
        #self.num_ado_obs = self.alg.actor_critic.teamacs[0].ac.encoder.num_ado_obs
        #self.num_agents = self.alg.actor_critic.teamacs[0].ac.encoder.num_agents

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions], actor_critic.team_size)

        # Log
        self.log_dir = log_dir
        self.track_cfg = train_cfg['track_cfgs']
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def store_cfgs(self,):
                with open(self.log_dir + '/cfg_train.yml', 'w') as outfile:
                    yaml.dump(self.train_cfg, outfile, default_flow_style=False)
                    
                with open(self.log_dir + '/cfg.yml', 'w') as outfile:
                    yaml.dump(self.env.cfg, outfile, default_flow_style=False)

    def get_attention(self, obs, attention_tensor):

        obs_ego = obs[..., :self.num_ego_obs]
        obs_ado = obs[..., self.num_ego_obs:self.num_ego_obs+(self.num_agents-1)*self.num_ado_obs]

        for ado_id in range(self.num_agents-1):
            ado_ag_obs = obs_ado[..., ado_id::(self.num_agents-1)]
            attention_tensor[:, ado_id] = self.attention_network(torch.cat((obs_ego, ado_ag_obs), dim=-1).detach()).squeeze()
        return attention_tensor

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.init_at_random_ep_len = init_at_random_ep_len
        
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
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        trewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_mean_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_mean_team_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    # Visualization
                    #self.env.viewer.update_values(self.alg.values_separate)
                    #self.env.viewer.update_ranks(self.env.ranks)
                    #attention = self.get_attention(obs[:, 0, :], self.attention_tensor)
                    #attention = self.alg.actor_critic.teamacs[0].ac.encoder.attention_weights.mean(dim=0, keepdim=True)
                    #self.env.viewer.update_attention(attention)

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    #obs, privileged_obs, rewards, dones, infos = self.env.step_with_importance_sampling_check(actions, self.alg.value_std_cur_norm)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_mean_reward_sum += torch.sum(torch.mean(rewards[:, self.alg.actor_critic.teams[0], :], dim = 1), dim = 1)
                        cur_mean_team_reward_sum += torch.mean(rewards[:, self.alg.actor_critic.teams[0], 1], dim = 1)
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_mean_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        trewbuffer.extend(cur_mean_team_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_mean_reward_sum[new_ids] = 0
                        cur_mean_team_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    #self.alg.actor_critic.update_ac_ratings(infos)

                stop = time.time()
                collection_time = stop - start

                # self.env.viewer.save_uncertain_imgs()

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, aux_info_loss = self.alg.update()
            if  it % self.population_update_interval == 0:
                self.alg.update_population()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()


            if it%self.eval_interval_steps == 0:
                with torch.inference_mode():
                    self.eval_episode(it, num_learning_iterations)

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def eval_episode(self, it, num_learning_iterations):
        obs, _ = self.env.reset()
        
        already_done = self.env.reset_buf > 1000
        eval_ep_rewards_tot = 0.*already_done
        eval_ep_rewards_team = 0.*already_done
        eval_ep_duration = 0.*already_done
        eval_ep_terminal_ranks = 0.*self.env.ranks
            
        self.alg.actor_critic.eval()
        policy = self.alg.actor_critic.act_inference
        for ev_it in range(self.env.max_episode_length+1):
            actions = policy(obs)

            # Visualization
            svalues = torch.concat([self.alg.actor_critic.evaluate(obs[:, agent_id, :].unsqueeze(1)).detach() for agent_id in self.alg.actor_critic.teams[0]], dim=-2)
            self.env.viewer.update_values(svalues)
            self.env.viewer.update_ranks(self.env.ranks)
            # attention = self.get_attention(obs[:, 0, :], self.attention_tensor)
            #attention = self.alg.actor_critic.teamacs[0].ac.encoder.attention_weights.mean(dim=0, keepdim=True)
            #self.env.viewer.update_attention(attention)

            obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
            
            eval_ep_duration += ev_it*dones*(~already_done)
            eval_ep_terminal_ranks += self.env.ranks[:, :] * (~already_done)*dones
            eval_ep_rewards_tot += torch.sum(rewards[:,0,:], dim = 1).view(-1,1)*(~already_done)
            eval_ep_rewards_team += (rewards[:,0,1]).view(-1,1)*(~already_done)
            
            already_done |= dones
            if ~torch.any(~already_done):
                break
        
        ratings = self.alg.update_ratings(eval_ep_terminal_ranks, eval_ep_duration, self.env.max_episode_length)
        mean_ep_duration = torch.mean(eval_ep_duration).item()
        mean_ep_rewards_tot = torch.mean(eval_ep_rewards_tot).item()
        mean_ep_rewards_team = torch.mean(eval_ep_rewards_team).item()
        mean_ep_ranks = torch.mean(eval_ep_terminal_ranks, dim = 0)[0].item()

        width = 80
        pad = 35
        str = f" \033[1m EVAL EPISODE {it}/{self.current_learning_iteration + num_learning_iterations} \033[0m "
        log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n"""
                    f"""{'AVG Duration:':>{pad}} {mean_ep_duration:.4f}\n"""
                    f"""{'AVG Reward:':>{pad}} {mean_ep_rewards_tot:.4f}\n"""
                    f"""{'AVG Reward Team:':>{pad}} {mean_ep_rewards_team:.4f}\n"""
                    f"""{'AVG Rank:':>{pad}} {mean_ep_ranks:.4f}\n"""
                    f"""{'Trueskill:':>{pad}} {ratings[0][0].mu:.4f}\n"""
                    )
        self.writer.add_scalar('Eval/episode_duration', mean_ep_duration, it)
        self.writer.add_scalar('Eval/episode_reward', mean_ep_rewards_tot, it)
        self.writer.add_scalar('Eval/episode_reward_team', mean_ep_rewards_team, it)
        self.writer.add_scalar('Eval/episode_ranks', mean_ep_ranks, it)
        self.writer.add_scalar('Eval/rating 0', ratings[0][0].mu, it)
        self.writer.add_scalar('Eval/rating 1', ratings[1][0][0].mu, it)
        
        print(log_string)

        self.alg.actor_critic.train()
        obs, _ = self.env.reset()
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        ego_action_mean_per_dim = self.alg.actor_critic.ego_action_mean.mean(dim=0)
        ego_action_std_per_dim = self.alg.actor_critic.ego_action_std.mean(dim=0)
        ego_action_mean_magnitude_per_dim = self.alg.actor_critic.ego_action_mean.abs().mean(dim=0)

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        if locs['aux_info_loss']:
            for key, value in locs['aux_info_loss'].items():
                self.writer.add_scalar('Loss/'+key, value, locs['it'])
                
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        #self.writer.add_scalar('Agent/Trueskill', self.alg.actor_critic.agentratings[0][0].mu, locs['it'])

        for action_id, (action_mean, action_std, action_mag_mean) in enumerate(zip(ego_action_mean_per_dim, ego_action_std_per_dim, ego_action_mean_magnitude_per_dim)):
            self.writer.add_scalar('Policy/ego_action_mean' + str(action_id), action_mean.item(), locs['it'])
            self.writer.add_scalar('Policy/ego_action_std' + str(action_id), action_std.item(), locs['it'])
            self.writer.add_scalar('Policy/ego_action_magnitude_mean' + str(action_id), action_mag_mean.item(), locs['it'])
        
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            self.writer.add_scalar('Train/min_episode_length', min(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/max_episode_length', max(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/episode_1step_freq', locs['lenbuffer'].count(1.0)/len(locs['lenbuffer']), locs['it'])

        if len(locs['trewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_team_reward', statistics.mean(locs['trewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_team_reward/time', statistics.mean(locs['trewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)


        string = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{string.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean total reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean team reward:':>{pad}} {statistics.mean(locs['trewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #  f"""{'Current Trueskill Agent:':>{pad}} {self.alg.actor_critic.agentratings[0][0].mu:.2f}\n""")
                        
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{string.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
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

    def save(self, path, infos=None):
        infos ={}# {'trueskill': {'mu':self.alg.actor_critic.agentratings[0][0].mu, 'sigma':self.alg.actor_critic.agentratings[0][0].sigma}}
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def load_multi_path(self, paths, load_optimizer=True):
        model_dicts = []
        opt_dicts = []
        dicts = []
        for path in paths:
            dict = torch.load(path)
            dicts.append(dict)
            model_dicts.append(dict['model_state_dict'])
            opt_dicts.append(dict['optimizer_state_dict'])
        self.alg.actor_critic.load_multi_state_dict(model_dicts)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(opt_dicts[0])
        self.current_learning_iteration = dicts[0]['iter']

        #add loaded models to buffers and set current ratings
        infos = []
        self.alg.actor_critic.past_models = model_dicts
        # self.alg.actor_critic.agentratings = []
        # self.alg.actor_critic.past_ratings_mu = []
        # self.alg.actor_critic.past_ratings_sigma = []
        for dict in dicts:
            info = dict['infos']
            # mu = info['trueskill']['mu']
            # sigma = info['trueskill']['sigma']
            # active_rating = self.alg.actor_critic.new_rating(mu, sigma)
            # self.alg.actor_critic.agentratings.append(active_rating)
            # self.alg.actor_critic.past_ratings_mu.append(mu)
            # self.alg.actor_critic.past_ratings_sigma.append(sigma)
            infos.append(info)
        return infos
    
    def populate_adversary_buffer(self, paths):
        for path in paths:
            dict = torch.load(path)
            info = dict['infos']
            #mu = info['trueskill']['mu']
            #sigma = info['trueskill']['sigma']
            #self.alg.actor_critic.past_ratings_mu.append(mu)
            #self.alg.actor_critic.past_ratings_sigma.append(sigma)
            self.alg.actor_critic.past_models.append(dict['model_state_dict'])

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_value_functions(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate_inference
