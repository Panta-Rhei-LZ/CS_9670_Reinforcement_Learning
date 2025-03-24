import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        # Convolutional layers - reusing the same architecture as AgentNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Linear layers for the actor network
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        self.policy = nn.Linear(512, n_actions)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x)
        fc_out = self.fc(conv_out)
        policy = F.softmax(self.policy(fc_out), dim=-1)
        return policy


class CriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # Convolutional layers - reusing the same architecture as AgentNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Linear layers for the critic network
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        self.value = nn.Linear(512, 1)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x)
        fc_out = self.fc(conv_out)
        value = self.value(fc_out)
        return value


class PPOAgent:
    def __init__(self,
                 input_dims,
                 num_actions,
                 actor_lr=0.0003,
                 critic_lr=0.0003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 policy_clip=0.2,
                 batch_size=32,
                 n_epochs=10,
                 epsilon=1.0,
                 eps_decay=0.99999975,
                 eps_min=0.1):

        self.num_actions = num_actions
        self.learn_step_counter = 0

        # Hyperparameters
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size

        # Exploration parameters (similar to DQN implementation for compatibility)
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # Networks
        self.actor = ActorNetwork(input_dims, num_actions)
        self.critic = CriticNetwork(input_dims)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Memory (simpler than replay buffer for on-policy algorithm)
        self.memory = []

        self.device = self.actor.device

    def choose_action(self, observation):
        # Epsilon-greedy exploration (for compatibility with existing code)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
            # Using a uniform distribution for random action log prob
            log_prob = -np.log(self.num_actions)
            return action, log_prob

        # Convert observation to tensor
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
            .unsqueeze(0) \
            .to(self.device)

        # Get action probabilities from actor network
        probs = self.actor(observation)

        # Sample action from the probability distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory.append({
            'state': np.array(state),
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'next_state': np.array(next_state),
            'done': done
        })

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def store_in_memory(self, state, action, reward, next_state, done):
        # DQN-style API for compatibility with main.py
        # Calculate log probability for the chosen action
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
            probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor([action]).to(self.device)).item()

        self.store_transition(state, action, log_prob, reward, next_state, done)

    def learn(self):
        if len(self.memory) == 0:
            return

        # Convert memory to numpy arrays
        states = np.array([t['state'] for t in self.memory])
        actions = np.array([t['action'] for t in self.memory])
        old_log_probs = np.array([t['log_prob'] for t in self.memory])
        rewards = np.array([t['reward'] for t in self.memory])
        next_states = np.array([t['next_state'] for t in self.memory])
        dones = np.array([t['done'] for t in self.memory])

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)

        # Calculate advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            next_values = self.critic(next_states).squeeze()

            # Calculate returns using GAE (Generalized Advantage Estimation)
            returns = []
            advantages = []
            gae = 0

            for i in reversed(range(len(rewards))):
                # If done, next value is 0
                next_val = 0 if dones[i] else next_values[i].item()
                delta = rewards[i] + self.gamma * next_val - values[i].item()
                gae = delta + self.gamma * self.gae_lambda * (0 if dones[i] else gae)
                returns.insert(0, gae + values[i].item())
                advantages.insert(0, gae)

            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train for n_epochs
        for _ in range(self.n_epochs):
            # Get new action probabilities and values
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            new_values = self.critic(states).squeeze()

            # Calculate ratios and clipped loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages

            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(new_values, returns)

            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Clear memory after update
        self.memory = []

        # Update epsilon and counter (for compatibility with DQN implementation)
        self.learn_step_counter += 1
        self.decay_epsilon()