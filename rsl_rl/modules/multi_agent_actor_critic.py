import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import copy
from rsl_rl.modules import ActorCritic, ActorCriticAttention#, ActorAttention, CriticAttention
import trueskill


class MAActorCritic():
    def __init__(self,  
                        num_actor_obs,
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
        self.num_teams = kwargs['numteams']
        self.team_size = kwargs['teamsize']
        self.kl_save_threshold = kwargs['kl_save_threshold']
        self.teams = [torch.tensor([idx for idx in range(self.team_size*start, self.team_size*start+self.team_size)], dtype=torch.long) for start in range(self.num_teams)]

        if self.is_attentive:
            self.ac1 = ActorCriticAttention(
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

        self.max_num_models = kwargs['max_num_old_models']
        self.draw_probs_unnorm = np.ones((self.max_num_models,))
        self.draw_probs_unnorm[0:-3] = 0.6/(self.max_num_models-3)
        self.draw_probs_unnorm[-3:] = 0.4/3

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
    
    def state_dict(self):
        return self.ac1.state_dict()

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
    
    def get_ratings(self,):
        return self.agentratings

    def set_ratings(self, ratings):
        self.agentratings = ratings

    def update_ac_ratings(self, infos):
        #update performance metrics of current policies
        
        pass
        # if 'ranking' in infos:         
        #     avgranking = infos['ranking'][0].cpu().numpy() #torch.mean(1.0*infos['ranking'], dim = 0).cpu().numpy()
        #     update_ratio = infos['ranking'][1]
        #     new_ratings = trueskill.rate(self.agentratings, avgranking)
        #     for old, new, it in zip(self.agentratings, new_ratings, range(len(self.agentratings))):
        #         mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
        #         sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
        #         self.agentratings[it] = (trueskill.Rating(mu, sigma),)

    def redraw_ac_networks_KL_divergence(self, obs_batch):
        current_ado_models = self.opponent_acs.copy()
        kl_divs = []
        self.ac1.update_distribution(obs_batch)
        dist_ego = self.ac1.distribution
        for ado_dict in self.past_models:
            self.opponent_acs[0].load_state_dict(ado_dict)
            self.opponent_acs[0].update_distribution(obs_batch)
            dist_ado = self.opponent_acs[0].distribution
            kl_divs.append(torch.mean(torch.sum(torch.distributions.kl_divergence(dist_ado, dist_ego), dim = 1)).item())       
        self.opponent_acs = current_ado_models
        print('[MAAC POPULATION UPDATE] KLs', kl_divs)

        if np.min(kl_divs)>self.kl_save_threshold:
            self.redraw_ac_networks(save = True)
        else:
            self.redraw_ac_networks(save = False)

    def redraw_ac_networks(self, save):
        #update population of competing agents, here simply load 
        #old version of agent 1 into ac2 slot
        if save:
            self.past_models.append(copy.deepcopy(self.ac1.state_dict()))
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

        idx = np.random.choice(len(self.past_models), self.num_agents-1, p = prob)
        for op_id, past_model_id in enumerate(idx):

            state_dict = self.past_models[past_model_id]
            mu = self.past_ratings_mu[past_model_id]
            sigma = self.past_ratings_sigma[past_model_id]
            self.opponent_acs[op_id].load_state_dict(state_dict)
            self.opponent_acs[op_id].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            rating = (trueskill.Rating(mu = mu, sigma = sigma ),) 
            self.agentratings[op_id+1] = rating

        #rating_train_pol = (trueskill.Rating(mu = self.agentratings[0][0].mu, sigma = self.agentratings[0][0].sigma),)
        #self.agentratings[0] = rating_train_pol 
        
    def new_rating(self, mu, sigma):
        return (trueskill.Rating(mu = mu, sigma = sigma ),) 


#Class coordinating multiple teams consisting of multiple policies with centralized critics
class MultiTeamCMAAC(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_agents,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):

        super(MultiTeamCMAAC, self).__init__()
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
        self.num_teams = kwargs['numteams']
        self.team_size = kwargs['teamsize']
        
        self.is_recurrent = False
        self.teams = [torch.tensor([idx for idx in range(self.team_size*start, self.team_size*start+self.team_size)], dtype=torch.long) for start in range(self.num_teams)]
        self.teamacs = [CMAActorCritic(num_actor_obs,
                                       num_critic_obs,
                                       num_actions,
                                       num_agents, 
                                       actor_hidden_dims=[256, 256, 256],
                                       critic_hidden_dims=[256, 256, 256],
                                       critic_output_dim=2,
                                       activation='elu',
                                       init_noise_std=1.0,
                                       **kwargs) for _ in range(self.num_teams)]

        self.max_num_models = 40
        self.draw_probs_unnorm = np.ones((self.max_num_models,))
        self.draw_probs_unnorm[0:-5] = 0.7/(self.max_num_models-3)
        self.draw_probs_unnorm[-5:] = 0.3/5

        self.past_models = [self.teamacs[0].state_dict()]

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
    def entropy(self):
        return self.teamacs[0].entropy
    
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
        #sum factors of value prediction
        value[:,:, 1] = torch.sum(value[:,:,1], dim =1).view(-1, 1)
        return value

    def update_ac_ratings(self, infos):
        pass
        # #update performance metrics of current policies
        # if 'ranking' in infos:         
        #     avgranking = infos['ranking'][0].cpu().numpy() #torch.mean(1.0*infos['ranking'], dim = 0).cpu().numpy()
        #     update_ratio = infos['ranking'][1]
        #     new_ratings = trueskill.rate(self.agentratings, avgranking)
        #     for old, new, it in zip(self.agentratings, new_ratings, range(len(self.agentratings))):
        #         mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
        #         sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
        #         self.agentratings[it] = (trueskill.Rating(mu, sigma),)

    def redraw_ac_networks(self):

        teamac1 = self.teamacs[0]

        #update population of competing agents, here simply load 
        #old version of agent 1 into ac2 slot
        self.past_models.append(teamac1.state_dict())
        # self.past_ratings_mu.append(self.agentratings[0][0].mu)
        # self.past_ratings_sigma.append(self.agentratings[0][0].sigma)
        if len(self.past_models) > self.max_num_models:
            idx_del = np.random.randint(0, self.max_num_models-2)
            del self.past_models[idx_del]
            # del self.past_ratings_mu[idx_del]
            # del self.past_ratings_sigma[idx_del]

        #select model to load
        #renormalize dist
        #if len(self.past_models) !=self.max_num_models:
        prob = 1/np.sum(self.draw_probs_unnorm[-len(self.past_models):] + 1e-4) * (self.draw_probs_unnorm[-len(self.past_models):] + 1e-4)
        idx = np.random.choice(len(self.past_models), self.num_teams-1, p = prob)
        for agent_id, past_model_id in enumerate(idx):
            op_id = agent_id + 1

            state_dict = self.past_models[past_model_id]
            # mu = self.past_ratings_mu[past_model_id]
            # sigma = self.past_ratings_sigma[past_model_id]
            self.teamacs[op_id].load_state_dict(state_dict)
            self.teamacs[op_id].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            # rating = (trueskill.Rating(mu = mu, sigma = sigma ),) 
            # self.agentratings[op_id+1] = rating

        # rating_train_pol = (trueskill.Rating(mu = self.agentratings[0][0].mu, sigma = self.agentratings[0][0].sigma),)
        # self.agentratings[0] = rating_train_pol 
        
    def new_rating(self, mu, sigma):
        return (trueskill.Rating(mu = mu, sigma = sigma ),) 


#multi agent actor critic with centralized critic for a single team
class CMAActorCritic(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_agents,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        critic_output_dim=1,
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):

        #Adapt ac interface here -------
        
        super(CMAActorCritic, self).__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        #self.num_agents = num_agents
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

        if self.is_attentive:
            self.ac = ActorCriticAttention(#num_ego_obs=kwargs['num_ego_obs'],
                                            #num_ado_obs=kwargs['num_ado_obs'],
                                            num_actions=num_actions,
                                            num_agents=num_agents,
                                            actor_hidden_dims=actor_hidden_dims,
                                            critic_hidden_dims=critic_hidden_dims,
                                            critic_output_dim=critic_output_dim,
                                            activation=activation,
                                            init_noise_std=init_noise_std, 
                                            **kwargs)
        else:
            raise NotImplementedError

            
#        if kwargs:
#            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        
        
        # self.is_recurrent = False

        # self.agentratings = []
        # for idx in range(num_agents):
        #     self.agentratings.append((trueskill.Rating(mu=0),))

        #self.max_num_models = 40
        #self.draw_probs_unnorm = np.ones((self.max_num_models,))
        #self.draw_probs_unnorm[0:-3] = 0.4/(self.max_num_models-3)
        #self.draw_probs_unnorm[-3:] = 0.6/3

        #self.past_models = [[self.ac.actor.state_dict(), self.ac.critic.state_dict()]]
        #self.past_ratings_mu = [0]
        #self.past_ratings_sigma = [self.agentratings[0][0].sigma]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return torch.stack(tuple(self.action_means), dim = 1)#self.ac.distribution.mean

    @property
    def action_std(self):
        return torch.stack(tuple(self.action_stds), dim = 1)
    
    @property
    def std(self):
        return torch.mean(self.action_std, dim = 1)

    @property
    def entropy(self):
        return torch.stack(tuple(self.action_entropies), dim = 1)
    
    #not sure how to fix these
    #@property
    #def parameters(self):
    #    return self.acs[0].parameters
    
    # def load_state_dict(self, path):
    #    self.ac.load_state_dict(path)

    def eval(self):
        self.ac.eval()
        
    def train(self):
        self.ac.train()
        return None
    
    def to(self, device):
        self.ac.to(device)
        return self

    def act(self, observations, **kwargs):
        actions = []
        self.action_means = []
        self.action_stds = []
        self.action_entropies = []
        for idx in range(self.team_size):
            actions.append(self.ac.act(observations[:, idx, :]))
            self.action_means.append(self.ac.action_mean)
            self.action_stds.append(self.ac.action_std)
            self.action_entropies.append(self.ac.entropy)
            
        actions = torch.stack(tuple(actions), dim = 1)
        return actions
    
    def act_inference(self, observations, **kwargs):
        actions = torch.stack(tuple([self.ac.act_inference(observations[:, idx, :]).detach() for idx in range(self.team_size)]), dim = 1)
        return actions
    
    def update_distribution_and_get_actions_log_prob(self, obs, actions):
        # actions = torch.stack(tuple([self.ac.actor.act_inference(observations[:, idx, :]) for idx in range(self.team_size)]), dim = 1)
        # return self.ac.actor.distribution.log_prob(actions).sum(dim=-1)
        return torch.stack([self.ac.update_dist_and_get_actions_log_prob(obs[:, idx,:], actions[:, idx, :]) for idx in range(self.team_size)], dim=1)

    def update_distribution_and_get_actions_log_prob_mu_sigma_entropy(self, obs, actions):
        # actions = torch.stack(tuple([self.ac.actor.act_inference(observations[:, idx, :]) for idx in range(self.team_size)]), dim = 1)
        # return self.ac.actor.distribution.log_prob(actions).sum(dim=-1)
        log_prob = []
        mu = []
        sigma = []
        entropy = []
        for idx in range(self.team_size):
            lp, m, s, e = self.ac.update_distribution_and_get_actions_log_prob_mu_sigma_entropy(obs[:, idx,:], actions[:, idx, :])
            log_prob.append(lp)
            mu.append(m)
            sigma.append(s)
            entropy.append(e)
        return torch.stack(tuple(log_prob), dim=1), torch.stack(tuple(mu), dim=1), torch.stack(tuple(sigma), dim=1), torch.stack(tuple(entropy), dim=1) 
    
    def evaluate_inference(self, observations, **kwargs):
        values = []
        values = [self.ac.evaluate(observations[:, 0,:])]
        op_values = [ac.evaluate(observations[:, idx+1,:]).unsqueeze(1) for idx, ac in enumerate(self.opponent_acs)]
        values += op_values 
        values = torch.cat(tuple(values), dim = 1)
        return values
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.ac.evaluate(critic_observations)
        return value

    def update_ac_ratings(self, infos):
        pass
        # #update performance metrics of current policies
        # if 'ranking' in infos:         
        #     avgranking = infos['ranking'][0].cpu().numpy() #torch.mean(1.0*infos['ranking'], dim = 0).cpu().numpy()
        #     update_ratio = infos['ranking'][1]
        #     new_ratings = trueskill.rate(self.agentratings, avgranking)
        #     for old, new, it in zip(self.agentratings, new_ratings, range(len(self.agentratings))):
        #         mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
        #         sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
        #         self.agentratings[it] = (trueskill.Rating(mu, sigma),)

    def new_rating(self, mu, sigma):
        return (trueskill.Rating(mu = mu, sigma = sigma ),) 