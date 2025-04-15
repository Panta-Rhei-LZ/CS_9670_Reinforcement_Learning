import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import time
import cv2

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE = 1000
MAX_STEPS = 5000
NUM_EPISODES = 1000
SAVE_PATH = './train.model'

# Environment setup
env = gym.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.unwrapped.fast_forward = 4
env.unwrapped.frameskip = 2
env.metadata['render_fps'] = 120
env.unwrapped.clock_mode = False
state = env.reset()

# DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
    def __len__(self):
        return len(self.buffer)

# Reward function
def compute_reward(info, step):
    reward = info['score'] * 0.01
    reward += (info['x_pos'] - info.get('prev_x_pos', 0)) * 0.1
    info['prev_x_pos'] = info['x_pos']
    reward -= 0.1
    if info.get('flag_get', False):
        reward += 1000
    return reward

# Training process
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    steps_done = 0

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        episode_start = time.time()
        done = False

        # Initialize frame stack
        frames = deque(maxlen=4)
        for _ in range(4):
            frames.append(preprocess(state))
        state = np.stack(frames, axis=0)

        for step in range(1, MAX_STEPS + 1):
            env.render()

            # ε-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, device=device).unsqueeze(0))
                    action = q_values.max(1)[1].item()

            next_state, reward, done, info = env.step(action)
            reward = compute_reward(info, step)

            # Update frame stack
            frames.append(preprocess(next_state))
            next_state = np.stack(frames, axis=0)

            # Store experience
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            # Train the policy network
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Epsilon decay
        epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
        duration = time.time() - episode_start
        print(f"Episode {episode}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f} | Time: {duration:.2f}s")

        # Save model every 100 episodes
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"Model saved: {SAVE_PATH}")

    torch.save(policy_net.state_dict(), SAVE_PATH)
    print(f"Training finished. Model saved to: {SAVE_PATH}")
    env.close()

# Frame preprocessing
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(obs, dtype=np.float32) / 255.0

# Start training
if __name__ == '__main__':
    train()
