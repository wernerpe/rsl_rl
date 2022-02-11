import torch

from dmaracing.utils.helpers import CmdLineArguments, getcfg

from math import log
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from dmaracing.utils.rl_helpers import get_mappo_runner
import numpy as np
import os
import time
#import trueskill
from scipy.stats import norm

from rsl_rl.modules import ActorCritic

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def train():
    env = DmarEnv(cfg, args)
    #env.viewer.mark_env(0)
    obs = env.obs_buf
    model_paths = []
    modelnrs = []
    for run, chkpt in zip(runs, chkpts):
        dir, modelnr = get_run(logdir, run = run, chkpt=chkpt)
        modelnrs.append(modelnr)
        model_paths.append("{}/model_{}.pt".format(dir, modelnr))
        print("Loading model" + model_paths[-1])
    runner = get_mappo_runner(env, cfg_train, logdir, env.device, cfg['sim']['numAgents'])
    
    policy_infos = runner.load_multi_path(model_paths)
    policy = runner.get_inference_policy(device=env.device)
    valuef = runner.get_value_functions(device=env.device)

    ac_ego = ActorCritic(
      env.num_obs,
      env.num_obs,
      env.num_actions,
      cfg_train['policy']['actor_hidden_dims'],
      cfg_train['policy']['critic_hidden_dims'],
      cfg_train['policy']['activation'],
      cfg_train['policy']['init_noise_std']).to(env.device)
    
    time_per_step = cfg['sim']['dt']*cfg['sim']['decimation']

    num_races = 0
    num_agent_0_wins = 0

    it_per_data_batch = 60
    num_steps = 6500
    lossfn = torch.nn.MSELoss().to(env.device)
    optimizer = torch.optim.Adam(ac_ego.parameters(), lr=cfg_train['algorithm']['learning_rate'])

    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")

    bc_logdir = os.getcwd() + '/rsl_rl/test/logs' + '/' + timestamp
    env._writer = SummaryWriter(log_dir=bc_logdir, flush_secs=10)
    env.log_video_freq = 1

    for i in range(num_steps):
        actions = policy(obs)
        values = valuef(obs)

        # Replace first action with cloned policy action
        actions_env0 = torch.concat([ac_ego.actor(obs[0, 0]).unsqueeze(0), actions[0, 1:]])
        actions_env = torch.concat([actions_env0.unsqueeze(0), actions[1:]])
        
        obs,_, rew, dones, info = env.step(actions_env)

        act_ego = actions[1:, 0]
        val_ego = values[1:, 0]
        obs_ego = obs[1:, 0]

        for _ in range(it_per_data_batch):
            optimizer.zero_grad()
            loss_train = lossfn(act_ego, ac_ego.actor(obs_ego))  # + lossfn(val_ego, ac_ego.critic(obs_ego))
            loss_train.backward(retain_graph=True)
            optimizer.step()
        progstring = (f"""{'#'*40}\n\n"""
                      f"""{'Step: ':>{10}}{str(i)}{'/'}{str(num_steps)}\n"""
                      f""" {'Train Error: ':>{10}}{loss_train:.4f}\n""")
        print(progstring)

        
        dones_idx = torch.unique(torch.where(dones)[0])
        if len(dones_idx):
            num_races += len(dones_idx)
            num_agent_0_wins +=len(torch.where(info['ranking'][:,0] == 0))

            if 0 in dones_idx:
              print("-----> Logging Video <-----")



if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = True 
    args.test = True
    path_cfg_base = os.path.dirname(os.getcwd()) + '/dmaracing/'
    path_cfg = path_cfg_base + 'cfg'

    cfg, cfg_train, logdir = getcfg(path_cfg)

    logdir = path_cfg_base + logdir

    chkpts = [-1, 18000, 16600, 4000]
    runs = [-1, -1, -1, -1]
    cfg['sim']['numEnv'] = 100
    cfg['sim']['numAgents'] = 4
    cfg['learn']['timeout'] = 300
    cfg['learn']['offtrack_reset'] = 5.0
    cfg['learn']['reset_tile_rand'] = 20
    cfg['sim']['test_mode'] = True
    cfg['viewer']['logEvery'] = -1
    cfg['track']['seed'] = 12
    cfg['track']['num_tracks'] = 20

    #cfg['track']['CHECKPOINTS'] = 3
    #cfg['track']['TRACK_RAD'] = 800
    cfg['viewer']['multiagent'] = True

    set_dependent_cfg_entries(cfg)

    train()    
