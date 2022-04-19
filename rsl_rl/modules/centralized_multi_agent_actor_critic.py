import numpy as np

import torch
from torch.distributions import Normal
import copy
from rsl_rl.modules import ActorCritic, ActorCriticAttention
import trueskill

class CMAActorCritic():
    def __init__(self,  
                        num_actor_obs,
                        num_critic_obs,
                        num_agents,
                        num_agents_per_team,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_agents = num_agents
        self.num_agents_per_team = num_agents_per_team
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims= critic_hidden_dims
        self.activation = activation
        self.init_noise_std = init_noise_std
        self.kwargs = kwargs
        self.is_attentive = kwargs['attentive']

        if self.is_attentive:
            self.ac1 = ActorCriticAttention(num_ego_obs=35,
                                            num_ado_obs=6,
                                            num_actions=num_actions,
                                            num_agents=num_agents,
                                            actor_hidden_dims=actor_hidden_dims,
                                            critic_hidden_dims=critic_hidden_dims,
                                            activation=activation,
                                            init_noise_std=init_noise_std, 
                                            **kwargs)
        else:
            self.ac1 = ActorCritic( num_actor_obs,
                                    num_critic_obs,
                                    num_actions,
                                    actor_hidden_dims,
                                    critic_hidden_dims,
                                    activation,
                                    init_noise_std, 
                                    **kwargs)
#        if kwargs:
#            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        
        
        self.opponent_acs = [copy.deepcopy(self.ac1) for _ in range(num_agents-1)]
        self.is_recurrent = False

        self.agentratings = []
        for idx in range(num_agents):
            self.agentratings.append((trueskill.Rating(mu=0),))

        self.max_num_models = 40
        self.draw_probs_unnorm = np.ones((self.max_num_models,))
        self.draw_probs_unnorm[0:-3] = 0.4/(self.max_num_models-3)
        self.draw_probs_unnorm[-3:] = 0.6/3

        self.past_models = [self.ac1.state_dict()]
        self.past_ratings_mu = [0]
        self.past_ratings_sigma = [self.agentratings[0][0].sigma]

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
        for ac in self.opponent_acs:
            ac.load_state_dict(path)

    def load_multi_state_dict(self, paths):
        self.ac1.load_state_dict(paths[0])
        for idx, path in enumerate(paths[1:]):
            self.opponent_acs[idx].load_state_dict(path)

    def eval(self):
        self.ac1.eval()
        for ac in self.opponent_acs:
            ac.eval()

    def train(self):
       self.ac1.train()
       return None
    
    def to(self, device):
        self.ac1.to(device)
        for ac in self.opponent_acs:
            ac.to(device)
        return self

    def act(self, observations, **kwargs):
        actions = []
        actions.append(self.ac1.act(observations[:, 0,:]).unsqueeze(1))
        op_actions = [ac.act_inference(observations[:, idx+1,:]).unsqueeze(1) for idx, ac in enumerate(self.opponent_acs)]
        actions += op_actions 
        actions = torch.cat(tuple(actions), dim = 1)
        return actions
    
    def act_inference(self, observations, **kwargs):
        actions = []
        actions.append(self.ac1.act_inference(observations[:, 0,:]).unsqueeze(1))
        op_actions = [ac.act_inference(observations[:, idx+1,:]).unsqueeze(1) for idx, ac in enumerate(self.opponent_acs)]
        actions += op_actions 
        actions = torch.cat(tuple(actions), dim = 1)
        return actions
        
    def act_ac_train(self, observations, **kwargs):
        actions = self.ac1.act(observations)
        return actions
    
    def get_actions_log_prob(self, actions):
        return self.ac1.distribution.log_prob(actions).sum(dim=-1)
    
    def evaluate_inference(self, observations, **kwargs):
        values = []
        values.append(self.ac1.evaluate(observations[:, 0,:]).unsqueeze(1))
        op_values = [ac.evaluate(observations[:, idx+1,:]).unsqueeze(1) for idx, ac in enumerate(self.opponent_acs)]
        values += op_values 
        values = torch.cat(tuple(values), dim = 1)
        return values
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.ac1.critic(critic_observations)
        return value

    def update_ac_ratings(self, infos):
        #update performance metrics of current policies
        if 'ranking' in infos:         
            avgranking = infos['ranking'][0].cpu().numpy() #torch.mean(1.0*infos['ranking'], dim = 0).cpu().numpy()
            update_ratio = infos['ranking'][1]
            new_ratings = trueskill.rate(self.agentratings, avgranking)
            for old, new, it in zip(self.agentratings, new_ratings, range(len(self.agentratings))):
                mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
                sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
                self.agentratings[it] = (trueskill.Rating(mu, sigma),)

    def redraw_ac_networks(self):
        #update population of competing agents, here simply load 
        #old version of agent 1 into ac2 slot
        self.past_models.append(self.ac1.state_dict())
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
            prob = self.draw_probs_unnorm

        idx = np.random.choice(len(self.past_models), self.num_agents-1, p = prob)
        for op_id, past_model_id in enumerate(idx):

            state_dict = self.past_models[past_model_id]
            mu = self.past_ratings_mu[past_model_id]
            sigma = self.past_ratings_sigma[past_model_id]
            self.opponent_acs[op_id].load_state_dict(state_dict)
            self.opponent_acs[op_id].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            rating = (trueskill.Rating(mu = mu, sigma = sigma ),) 
            self.agentratings[op_id+1] = rating

        rating_train_pol = (trueskill.Rating(mu = self.agentratings[0][0].mu, sigma = self.agentratings[0][0].sigma),)
        self.agentratings[0] = rating_train_pol 
        
    def new_rating(self, mu, sigma):
        return (trueskill.Rating(mu = mu, sigma = sigma ),) 

#multi agent actor critic with centralized critic
class CMAActorCritic():
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_agents,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims= critic_hidden_dims
        self.activation = activation
        self.init_noise_std = init_noise_std
        self.kwargs = kwargs
        self.is_attentive = kwargs['attentive']
        self.team_size = kwargs['teamsize']

        if self.is_attentive:
            self.ac1 = ActorCriticAttention(num_ego_obs=35,
                                            num_ado_obs=6,
                                            num_actions=num_actions,
                                            num_agents=num_agents,
                                            actor_hidden_dims=actor_hidden_dims,
                                            critic_hidden_dims=critic_hidden_dims,
                                            activation=activation,
                                            init_noise_std=init_noise_std, 
                                            **kwargs)
        else:
            self.ac1 = ActorCritic( num_actor_obs,
                                    num_critic_obs,
                                    num_actions,
                                    actor_hidden_dims,
                                    critic_hidden_dims,
                                    activation,
                                    init_noise_std, 
                                    **kwargs)
#        if kwargs:
#            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        
        
        self.opponent_acs = [copy.deepcopy(self.ac1) for _ in range(num_agents-1)]
        self.is_recurrent = False

        self.agentratings = []
        for idx in range(num_agents):
            self.agentratings.append((trueskill.Rating(mu=0),))

        self.max_num_models = 40
        self.draw_probs_unnorm = np.ones((self.max_num_models,))
        self.draw_probs_unnorm[0:-3] = 0.4/(self.max_num_models-3)
        self.draw_probs_unnorm[-3:] = 0.6/3

        self.past_models = [self.ac1.state_dict()]
        self.past_ratings_mu = [0]
        self.past_ratings_sigma = [self.agentratings[0][0].sigma]

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
        for ac in self.opponent_acs:
            ac.load_state_dict(path)

    def load_multi_state_dict(self, paths):
        self.ac1.load_state_dict(paths[0])
        for idx, path in enumerate(paths[1:]):
            self.opponent_acs[idx].load_state_dict(path)

    def eval(self):
        self.ac1.eval()
        for ac in self.opponent_acs:
            ac.eval()

    def train(self):
       self.ac1.train()
       return None
    
    def to(self, device):
        self.ac1.to(device)
        for ac in self.opponent_acs:
            ac.to(device)
        return self

    def act(self, observations, **kwargs):
        actions = []
        actions.append(self.ac1.act(observations[:, 0,:]).unsqueeze(1))
        op_actions = [ac.act_inference(observations[:, idx+1,:]).unsqueeze(1) for idx, ac in enumerate(self.opponent_acs)]
        actions += op_actions 
        actions = torch.cat(tuple(actions), dim = 1)
        return actions
    
    def act_inference(self, observations, **kwargs):
        actions = []
        actions.append(self.ac1.act_inference(observations[:, 0,:]).unsqueeze(1))
        op_actions = [ac.act_inference(observations[:, idx+1,:]).unsqueeze(1) for idx, ac in enumerate(self.opponent_acs)]
        actions += op_actions 
        actions = torch.cat(tuple(actions), dim = 1)
        return actions
        
    def act_ac_train(self, observations, **kwargs):
        actions = self.ac1.act(observations)
        return actions
    
    def get_actions_log_prob(self, actions):
        return self.ac1.distribution.log_prob(actions).sum(dim=-1)
    
    def evaluate_inference(self, observations, **kwargs):
        values = []
        values.append(self.ac1.evaluate(observations[:, 0,:]).unsqueeze(1))
        op_values = [ac.evaluate(observations[:, idx+1,:]).unsqueeze(1) for idx, ac in enumerate(self.opponent_acs)]
        values += op_values 
        values = torch.cat(tuple(values), dim = 1)
        return values
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.ac1.critic(critic_observations)
        return value

    def update_ac_ratings(self, infos):
        #update performance metrics of current policies
        if 'ranking' in infos:         
            avgranking = infos['ranking'][0].cpu().numpy() #torch.mean(1.0*infos['ranking'], dim = 0).cpu().numpy()
            update_ratio = infos['ranking'][1]
            new_ratings = trueskill.rate(self.agentratings, avgranking)
            for old, new, it in zip(self.agentratings, new_ratings, range(len(self.agentratings))):
                mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
                sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
                self.agentratings[it] = (trueskill.Rating(mu, sigma),)

    def redraw_ac_networks(self):
        #update population of competing agents, here simply load 
        #old version of agent 1 into ac2 slot
        self.past_models.append(self.ac1.state_dict())
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
            prob = self.draw_probs_unnorm

        idx = np.random.choice(len(self.past_models), self.num_agents-1, p = prob)
        for op_id, past_model_id in enumerate(idx):

            state_dict = self.past_models[past_model_id]
            mu = self.past_ratings_mu[past_model_id]
            sigma = self.past_ratings_sigma[past_model_id]
            self.opponent_acs[op_id].load_state_dict(state_dict)
            self.opponent_acs[op_id].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            rating = (trueskill.Rating(mu = mu, sigma = sigma ),) 
            self.agentratings[op_id+1] = rating

        rating_train_pol = (trueskill.Rating(mu = self.agentratings[0][0].mu, sigma = self.agentratings[0][0].sigma),)
        self.agentratings[0] = rating_train_pol 
        
    def new_rating(self, mu, sigma):
        return (trueskill.Rating(mu = mu, sigma = sigma ),) 