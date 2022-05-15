
from lib2to3.pgen2.literals import simple_escapes
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import CMAActorCritic, MultiTeamCMAAC 
from rsl_rl.storage import CentralizedMultiAgentRolloutStorage

#only track transitions of agent 1, agent 2 blindly runs old version of policy 
#which gets exchanged periodically for the current version
#generator of multi agent rollout storage only returns data on agent 1

class JRMAPPO:
    actor_critic: MultiTeamCMAAC
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
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = CentralizedMultiAgentRolloutStorage.Transition()

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

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, num_agents):
        self.storage = CentralizedMultiAgentRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, num_agents, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Compute the actions and values
        all_agent_actions =  self.actor_critic.act(obs).detach()
        self.transition.actions = all_agent_actions[:, self.actor_critic.teams[0], :]
        self.transition.values = self.actor_critic.evaluate(critic_obs[:, self.actor_critic.teams[0], :]).detach()
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
        self.transition.rewards = rewards[:, self.actor_critic.teams[0], :].clone()
        self.transition.dones = dones
        if 'agent_active' in infos:
          self.transition.active_agents = 1.0 * infos['agent_active']
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * self.transition.values * infos['time_outs'].unsqueeze(1).unsqueeze(1).to(self.device)

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
        mean_joint_ratio_values = 0
        mean_jr_den = 0
        mean_jr_num = 0
        mean_advantage_values = 0
        mean_mu0_batch = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        elif self.actor_critic.is_attentive:
          generator = self.storage.attention_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_individual_batch, target_values_team_batch, advantages_batch, returns_individual_batch, returns_team_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, active_agents in generator:

                #self.actor_critic.act_ac_train(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0], active_agents=active_agents)
                actions_log_prob_batch, mu_batch, sigma_batch, entropy_batch = self.actor_critic.update_distribution_and_get_actions_log_prob_mu_sigma_entropy(obs_batch, actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = mu_batch.flatten(0,1)
                sigma_batch = sigma_batch.flatten(0,1)
                entropy_batch = entropy_batch.flatten(0,1)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch.flatten(0, 1) + 1.e-5) + (torch.square(old_sigma_batch.flatten(0, 1)) + torch.square(old_mu_batch.flatten(0, 1) - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)


                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss, using the joint probability ratios of all team members
                ratio = torch.squeeze(torch.exp(torch.sum(actions_log_prob_batch, dim = 1) - torch.sum(old_actions_log_prob_batch.squeeze(), dim = 1)))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_individual_clipped = target_values_individual_batch + (value_batch[:,:,0] - target_values_individual_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses_individual = (value_batch[:,:,0] - returns_individual_batch).pow(2)
                    value_losses_individual_clipped = (value_individual_clipped - returns_individual_batch).pow(2)
                    value_loss_individual = torch.max(value_losses_individual, value_losses_individual_clipped).mean()
                   
                    value_team_clipped = target_values_team_batch + (value_batch[:,:,1] - target_values_team_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses_team = (torch.sum(value_batch[:,:,1] - returns_team_batch, dim =1)).pow(2)
                    value_losses_team_clipped = (torch.sum(value_team_clipped - returns_team_batch, dim=1)).pow(2)
                    value_loss_team = torch.max(value_losses_team, value_losses_team_clipped).mean()
                    
#                  raise ValueError('check the value output index such that team and individual are correctly assigned')

                else: 
                    value_loss_individual = (returns_individual_batch - value_batch[:,:,0]).pow(2).mean()
                    value_loss_team = (torch.sum(returns_team_batch - value_batch[:,:,1], dim = 1)).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * (value_loss_team + value_loss_individual)  - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss_team.item() + value_loss_individual.item() 
                mean_surrogate_loss += surrogate_loss.item()
                mean_joint_ratio_values += ratio.mean().item()
                mean_advantage_values += advantages_batch.mean().item()
                mean_jr_num += torch.sum(actions_log_prob_batch, dim = 1).mean().item()
                mean_jr_den += torch.sum(old_actions_log_prob_batch, dim = 1).mean().item()
                mean_mu0_batch += mu_batch[:, 0].mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_joint_ratio_values /= num_updates
        mean_advantage_values /= num_updates
        mean_jr_num /= num_updates
        mean_jr_den /= num_updates
        mean_mu0_batch /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, {'mean_joint_ratio_val': mean_joint_ratio_values, 
                                                      'mean_advantage_val': mean_advantage_values, 
                                                      'mean_jr_num': mean_jr_num, 
                                                      'mean_jr_den': mean_jr_den, 
                                                      'mean_mu0': mean_mu0_batch}

    def update_population(self,):
        self.actor_critic.redraw_ac_networks()