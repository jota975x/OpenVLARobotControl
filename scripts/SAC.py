import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# HYPERPARAMETERS
ACTOR_HIDDEN_DIM = 256
LOG_MIN = -20
LOG_MAX = 2
ACTION_SCALE = 1.0
ACTION_BIAS = 0.5

CRITIC_HIDDEN_DIM = 256

LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.5
ALPHA_DECAY = 0.999
ALPHA_MIN = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim: int=ACTOR_HIDDEN_DIM, action_scale=ACTION_SCALE, action_bias=ACTION_BIAS):
        super(Actor, self).__init__()
        
        # variables
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.activation = nn.ReLU()
        
        # networks
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.policy(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_MIN, max=LOG_MAX)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        
        action = torch.tanh(action)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2) + 1e-6))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim=CRITIC_HIDDEN_DIM):
        super(Critic, self).__init__()
        
        # variables
        self.activation = nn.ReLU()
        
        # networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2
    

class AgentSAC(nn.Module):
    def __init__(self, state_dim, action_dim,
                 lr=LEARNING_RATE, gamma=GAMMA, tau=TAU, 
                 alpha=ALPHA, alpha_decay=ALPHA_DECAY, alpha_min=ALPHA_MIN):
        super(AgentSAC, self).__init__()
        
        # initialize networks
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # variables
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        
    def update_critic(self, states, actions, next_states, rewards, dones):
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(-1)
        
        # update critic
        with torch.no_grad():
            next_actions, next_logprobs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * (target_q - self.alpha * next_logprobs)
            
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # target critic soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def update_actor(self, states):
        states = torch.tensor(states, dtype=torch.float32).to(device)
        
        # update actor
        actions, logprobs = self.actor.sample(states)
        q1, q2 = self.critic(states, actions)
        actor_loss = (self.alpha * logprobs - torch.min(q1, q2)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # decay alpha
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            
        
        