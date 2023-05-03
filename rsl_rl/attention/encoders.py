import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np


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

      latent = 0*obs_ado[..., 0::(self.num_agents-1)]

      for ado_id in range(self.num_agents-1):
          ado_ag_obs = obs_ado[..., ado_id::(self.num_agents-1)]
          latent += self._network(torch.cat((obs_ego, ado_ag_obs), dim=-1)) * ado_ag_obs

      return torch.cat((obs_ego, latent), dim=-1)


class EncoderAttention3(nn.Module):

  def __init__(self, num_ego_obs, num_ado_obs , hidden_dims, output_dim, numteams, teamsize, activation):

        super(EncoderAttention3, self).__init__()

        self.num_agents = numteams * teamsize
        self.num_ego_obs = num_ego_obs
        self.num_ado_obs = num_ado_obs

        self.team_ids = [team_id for team_id in range(numteams) for _ in range(teamsize)]
        # self.team_ids = [self.team_ids for _ in range(numteams)]
        self.team_ids = torch.tensor(self.team_ids).to('cuda:0')

        encoder_layers = []
        encoder_layers.append(nn.Linear(num_ego_obs + num_ado_obs + 1, hidden_dims[0]))
        encoder_layers.append(nn.LayerNorm(hidden_dims[0]))
        encoder_layers.append(nn.Tanh())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                encoder_layers.append(activation)
        # encoder_layers.append(nn.Sigmoid())
        self._network = nn.Sequential(*encoder_layers)

  def forward(self, observations):

      obs_ego = observations[..., :self.num_ego_obs]
      obs_ado = observations[..., self.num_ego_obs:self.num_ego_obs+(self.num_agents-1)*self.num_ado_obs]

      if len(obs_ego.shape) > 2:
        teamids = self.team_ids.repeat(observations.shape[0], observations.shape[1], 1)
      else:
        teamids = self.team_ids.repeat(observations.shape[0], 1)

      latent = 0.0

      for ado_id in range(self.num_agents-1):
          ado_ag_obs = obs_ado[..., ado_id::(self.num_agents-1)]
          latent += self._network(torch.cat((obs_ego, ado_ag_obs, teamids[..., ado_id+1].unsqueeze(dim=-1)), dim=-1))

      return torch.cat((obs_ego, latent), dim=-1)


class EncoderAttention4(nn.Module):

  def __init__(self, num_ego_obs, num_ado_obs, embed_dims, attend_dims, output_dim, numteams, teamsize, activation):
  # def __init__(self, num_ego_obs, num_ado_obs , hidden_dims, output_dim, numteams, teamsize, activation):

        super(EncoderAttention4, self).__init__()

        self.num_agents = numteams * teamsize
        self.num_ego_obs = num_ego_obs
        self.num_ado_obs = num_ado_obs
        self.teamsize = teamsize

        self.team_ids_list = [team_id for team_id in range(numteams) for _ in range(teamsize)]
        team_ids = torch.tensor(self.team_ids_list, dtype=torch.float)
        self.register_buffer('team_ids', team_ids)

        # Parameters
        attention_heads = 4
        # hidden_dim = 32
        self.attend_dim = attend_dims[-1] // attention_heads

        # self.attention_weights = torch.zeros((attention_heads, self.num_agents-1)).to(device)

        # EGO encoder
        ego_encoder_layers = []
        ego_encoder_layers.append(nn.Linear(num_ego_obs, embed_dims[0]))
        # ego_encoder_layers.append(nn.LayerNorm(embed_dims[0]))
        # ego_encoder_layers.append(nn.Tanh())
        ego_encoder_layers.append(nn.LeakyReLU())
        for l in range(len(embed_dims)-1):
            ego_encoder_layers.append(nn.Linear(embed_dims[l], embed_dims[l + 1]))
            ego_encoder_layers.append(activation)
        # encoder_layers.append(nn.Sigmoid())
        self.ego_encoder = nn.Sequential(*ego_encoder_layers)

        # ADO encoder
        ado_encoder_layers = []
        ado_encoder_layers.append(nn.Linear(num_ado_obs + 1, embed_dims[0]))
        # ado_encoder_layers.append(nn.LayerNorm(embed_dims[0]))
        # ado_encoder_layers.append(nn.Tanh())
        ado_encoder_layers.append(nn.LeakyReLU())
        for l in range(len(embed_dims)-1):
            ado_encoder_layers.append(nn.Linear(embed_dims[l], embed_dims[l + 1]))
            ado_encoder_layers.append(activation)
        # encoder_layers.append(nn.Sigmoid())
        self.ado_encoder = nn.Sequential(*ado_encoder_layers)

        # Extractors
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.latent_extractors = nn.ModuleList()
        for i in range(attention_heads):
          self.key_extractors.append(nn.Linear(embed_dims[-1], self.attend_dim, bias=False))
          self.selector_extractors.append(nn.Linear(embed_dims[-1], self.attend_dim, bias=False))
          self.latent_extractors.append(nn.Sequential(nn.Linear(embed_dims[-1], self.attend_dim), nn.LeakyReLU()))

        # Projection layer
        # self.projection_net = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.projection_net = nn.Linear(attend_dims[-1], self.attend_dim)

        # # Dropout layers
        # drop_prob = 0.0
        # self.attention_drop = nn.Dropout(drop_prob)
        # self.projection_drop = nn.Dropout(drop_prob)


  def forward(self, observations):

      multi_ego = False
      obs_shape = observations.shape
      if len(observations.shape)==3:
        multi_ego = True
        observations = observations.view(-1, obs_shape[-1])

      obs_ego = observations[..., :self.num_ego_obs]
      obs_ado = observations[..., self.num_ego_obs:self.num_ego_obs+(self.num_agents-1)*self.num_ado_obs]

      # if len(obs_ego.shape) > 2:
      #   teamids = self.team_ids.repeat(observations.shape[0], observations.shape[1], 1)
      # else:
      #   teamids = self.team_ids.repeat(observations.shape[0], 1)
      teamids = self.team_ids + 0 * observations[..., 0].unsqueeze(dim=-1)

      ego_latent = self.ego_encoder(obs_ego)

      ado_latents = []
      for ado_id in range(self.num_agents-1):
          ado_ag_obs = obs_ado[..., ado_id::(self.num_agents-1)]
          ado_ag_obs = torch.cat((ado_ag_obs, teamids[..., ado_id+1].unsqueeze(dim=-1)), dim=-1)
          ado_latent = self.ado_encoder(ado_ag_obs)

          ado_latents.append(ado_latent)
        
      all_ego_slct = [selector_extractor(ego_latent) for selector_extractor in self.selector_extractors]
      all_ado_keys = [[key_extractor(ado_latent) for ado_latent in ado_latents] for key_extractor in self.key_extractors]
      all_ado_lats = [[latent_extractor(ado_latent) for ado_latent in ado_latents] for latent_extractor in self.latent_extractors]

      all_ado_latents = []
      for head_idx, (ego_slct, ado_keys, ado_lat) in enumerate(zip(all_ego_slct, all_ado_keys, all_ado_lats)):
          attend_logits = torch.matmul(ego_slct.view(ego_slct.shape[0], 1, -1), torch.stack(ado_keys).permute(1, 2, 0))
          scaled_attend_logits = attend_logits / torch.sqrt(torch.tensor(self.attend_dim))
          attend_weights = F.softmax(scaled_attend_logits, dim=2)

          # self.attention_weights[head_idx, :] = attend_weights[0, 0].detach()

          # attend_weights = self.attention_drop(attend_weights)

          # ado_latents = (torch.stack(ado_lat).permute(1, 2, 0) * attend_weights).sum(dim=2)
          all_ado_latents.append((torch.stack(ado_lat).permute(1, 2, 0) * attend_weights).sum(dim=2))

      all_ado_latents = torch.cat(all_ado_latents, dim=1)
      all_ado_latents = self.projection_net(all_ado_latents)
      # all_ado_latents = self.projection_drop(all_ado_latents)

      # new_obs = torch.cat((obs_ego, *all_ado_latents), dim=1)
      new_obs = torch.cat((obs_ego, all_ado_latents), dim=1)
      if multi_ego:
        # new_obs = new_obs.view(*obs_shape[:2], -1)
        new_obs = new_obs.view(obs_shape[0], obs_shape[1], -1)
      return new_obs


class Head(nn.Module):
    """ one head of cross-attention """

    def __init__(self, ego_size, ado_size, head_size):
        super().__init__()
        self.ado_key = nn.Linear(ado_size, head_size, bias=False)
        self.ego_query = nn.Linear(ego_size, head_size, bias=False)
        self.ado_value = nn.Sequential(nn.Linear(ado_size, head_size), nn.LeakyReLU())

    def forward(self, x_ego, x_ado):
        # x_ego: [B, 1, Ce]
        # x_ado: [B, O, Ca]
        k = self.ado_key(x_ado)    # [B, O, hs]
        q = self.ego_query(x_ego)  # [B, 1, hs]
        # compute attention scores ("affinities")
        watt = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, 1, hs) @ (B, hs, O) -> (B, 1, O)
        watt = F.softmax(watt, dim=-1) # (B, 1, O)
        # perform the weighted aggregation of the values
        v = self.ado_value(x_ado)   # (B, O, hs)
        out = watt @ v # (B, 1, O) @ (B, O, hs) -> (B, 1, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, ego_size, ado_size, n_embd, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(ego_size, ado_size, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x_ego, x_ado):
        out = torch.cat([h(x_ego, x_ado) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, ego_size, ado_size, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(ego_size, ado_size, n_embd, n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1_ego = nn.LayerNorm(ego_size)
        self.ln1_ado = nn.LayerNorm(ado_size)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x_ego, x_ado):
        x_ego = self.ln1_ego(x_ego)  # TODO: where to out LayerNorm
        x_ado = self.ln1_ado(x_ado)
        x = self.sa(x_ego, x_ado)
        x = x + self.ffwd(self.ln2(x))
        return x



class EncoderAttention4v2(nn.Module):

  def __init__(self, num_ego_obs, num_ado_obs, embed_dims, attend_dims, output_dim, numteams, teamsize, activation):

        super(EncoderAttention4v2, self).__init__()

        self.num_agents = numteams * teamsize
        self.num_ego_obs = num_ego_obs
        self.num_ado_obs = num_ado_obs
        self.teamsize = teamsize

        self.team_ids_list = [team_id for team_id in range(numteams) for _ in range(teamsize)]
        team_ids = torch.tensor(self.team_ids_list, dtype=torch.float)
        self.register_buffer('team_ids', team_ids)

        # Parameters
        attention_heads = 4
        # hidden_dim = 32
        attention_dim = attend_dims[-1] // attention_heads

        # ### Embeddings ###
        # Ego embedding encoder
        ego_encoder_layers = []
        ego_encoder_layers.append(nn.Linear(num_ego_obs, embed_dims[0]))
        ego_encoder_layers.append(nn.LayerNorm(embed_dims[0]))
        ego_encoder_layers.append(nn.Tanh())
        for l in range(len(embed_dims)-1):
            ego_encoder_layers.append(nn.Linear(embed_dims[l], embed_dims[l + 1]))
            ego_encoder_layers.append(activation)
        self.ego_encoder = nn.Sequential(*ego_encoder_layers)

        # Ado embedding encoder
        ado_encoder_layers = []
        ado_encoder_layers.append(nn.Linear(num_ado_obs, embed_dims[0]))
        ado_encoder_layers.append(nn.LayerNorm(embed_dims[0]))
        ado_encoder_layers.append(nn.Tanh())
        for l in range(len(embed_dims)-1):
            ado_encoder_layers.append(nn.Linear(embed_dims[l], embed_dims[l + 1]))
            ado_encoder_layers.append(activation)
        self.ado_encoder = nn.Sequential(*ado_encoder_layers)

        # ### Attention block ###
        self.att_block = Block(
          ego_size=embed_dims[-1], 
          ado_size=embed_dims[-1]+1,
          n_embd=attention_dim,
          n_head=attention_heads,
          )


  def forward(self, observations):

      multi_ego = False
      obs_shape = observations.shape
      if len(observations.shape)==3:
        multi_ego = True
        observations = observations.view(-1, obs_shape[-1])

      obs_ego = observations[..., :self.num_ego_obs]
      obs_ado = observations[..., self.num_ego_obs:self.num_ego_obs+(self.num_agents-1)*self.num_ado_obs]

      teamids = self.team_ids + 0 * observations[..., 0].unsqueeze(dim=-1)

      x_ego = self.ego_encoder(obs_ego.unsqueeze(dim=1))
      x_ado = []
      for ado_id in range(self.num_agents-1):
        x_ado.append(torch.cat((self.ado_encoder(obs_ado[..., ado_id::(self.num_agents-1)]), teamids[..., ado_id+1].unsqueeze(dim=-1)), dim=-1))
      x_ado = torch.stack(x_ado, dim=1)
      # x_ado = torch.stack([torch.cat((obs_ado[..., ado_id::(self.num_agents-1)], teamids[..., ado_id+1].unsqueeze(dim=-1)), dim=-1) for ado_id in range(self.num_agents-1)], dim=1)

      z_ado = self.att_block(x_ego, x_ado)
      z_ado = z_ado.squeeze(dim=1)

      new_obs = torch.cat((obs_ego, z_ado), dim=1)
      if multi_ego:
        # new_obs = new_obs.view(*obs_shape[:2], -1)
        new_obs = new_obs.view(obs_shape[0], obs_shape[1], -1)
      return new_obs


class EncoderIdentity(nn.Module):

  def __init__(self):

        super(EncoderIdentity, self).__init__()

  def forward(self, observations):

      return observations


def get_encoder(num_ego_obs, num_ado_obs, embed_dims, attend_dims, teamsize, numteams, encoder_type, activation=torch.nn.LeakyReLU()):

    encoder = None

    if encoder_type=='identity':
        encoder = EncoderIdentity()
    elif encoder_type=='attention4':
        # encoder = EncoderAttention4(
        encoder = EncoderAttention4v2(
          num_ego_obs=num_ego_obs, 
          num_ado_obs=num_ado_obs, 
          embed_dims=embed_dims, 
          attend_dims=attend_dims,
          output_dim=num_ado_obs, 
          numteams=numteams, 
          teamsize=teamsize,
          activation=activation,
        )
    else:
        raise NotImplementedError
    return encoder
