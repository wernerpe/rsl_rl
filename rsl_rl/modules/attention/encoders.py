import torch
import torch.nn as nn


class EncoderAttention1(nn.Module):

  def __init__(self, num_ego_obs, num_ado_obs , hidden_dims, output_dim, num_agents, activation):

        super(EncoderAttention1, self).__init__()

        self.num_agents = num_agents
        self.num_ego_obs = num_ego_obs
        self.num_ado_obs = num_ado_obs

        encoder_layers = []
        encoder_layers.append(nn.Linear(num_ado_obs, hidden_dims[0]))
        encoder_layers.append(nn.LayerNorm(hidden_dims[0]))
        encoder_layers.append(nn.Tanh())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                encoder_layers.append(activation)
        encoder_layers.append(nn.Sigmoid())
        self._network = nn.Sequential(*encoder_layers)

  def forward(self, observations):

      obs_ego = observations[..., :self.num_ego_obs]
      obs_ado = observations[..., self.num_ego_obs:self.num_ego_obs+(self.num_agents-1)*self.num_ado_obs]

      latent = 0.0

      for ado_id in range(self.num_agents-1):
          ado_ag_obs = obs_ado[..., ado_id::self.num_agents-1]
          latent += self._network(ado_ag_obs) * ado_ag_obs

      return torch.cat((obs_ego, latent), dim=-1)


class EncoderAttention2(nn.Module):

  def __init__(self, num_ego_obs, num_ado_obs , hidden_dims, output_dim, num_agents, activation):

        super(EncoderAttention2, self).__init__()

        self.num_agents = num_agents
        self.num_ego_obs = num_ego_obs
        self.num_ado_obs = num_ado_obs

        encoder_layers = []
        encoder_layers.append(nn.Linear(num_ego_obs + num_ado_obs, hidden_dims[0]))
        encoder_layers.append(nn.LayerNorm(hidden_dims[0]))
        encoder_layers.append(nn.Tanh())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                encoder_layers.append(activation)
        encoder_layers.append(nn.Sigmoid())
        self._network = nn.Sequential(*encoder_layers)

  def forward(self, observations):

      obs_ego = observations[..., :self.num_ego_obs]
      obs_ado = observations[..., self.num_ego_obs:self.num_ego_obs+(self.num_agents-1)*self.num_ado_obs]

      latent = 0.0

      for ado_id in range(self.num_agents-1):
          ado_ag_obs = obs_ado[..., ado_id::self.num_agents-1]
          latent += self._network(torch.cat((obs_ego, ado_ag_obs), dim=-1)) * ado_ag_obs

      return torch.cat((obs_ego, latent), dim=-1)