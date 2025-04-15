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

# ===========================
# 1. Hyperparameter Settings
# ===========================
GAMMA = 0.99                # Discount factor
LR = 1e-4                   # Learning rate
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_END = 0.1           # Final exploration rate
EPSILON_DECAY_EPISODES = 500  # Linear decay across episodes
BATCH_SIZE = 64             # Batch size
MEMORY_SIZE = 50000         # Replay buffer size
TARGET_UPDATE = 1000        # Target network update frequency (in steps)
MAX_STEPS = 5000            # Max steps per episode
NUM_EPISODES = 1000         # Number of training episodes
SAVE_PATH = './train.model' # Model save path
FRAME_SKIP = 4              # Number of repeated actions

# ===========================
# 2. Environment Setup
# ===========================
env = gym.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.unwrapped.fast_forward = 4
env.metadata['render_fps'] = 240
env.unwrapped.clock_mode = False
state = env.reset()

# ===========================
# 3. DQN Network Definition
# ===========================
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

# ===========================
# 4. Replay Buffer
# ===========================
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

# ===========================
# 5. Custom Reward Function
# ===========================
def compute_reward(info, step, prev_x_pos):
    reward = 0
    x_pos = info['x_pos']

    # Encourage forward movement
    reward += (x_pos - prev_x_pos) * 0.2

    # Bonus for jumping
    if info.get('jumping', False):
        reward += 0.2

    # Bonus for completing level
    if info.get('flag_get', False):
        reward += 1000

    # Penalty for falling or losing life
    if info.get('life_lost', False):
        reward -= 500

    # Penalty for enemy collision
    if info.get('enemy_nearby', False):
        reward -= 100

    # Small time penalty
    reward -= 0.02

    return reward

# ===========================
# 6. Info Display Function
# ===========================
def display_info(x_pos, prev_x_pos, reward):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"x_pos: {x_pos:.2f}  prev_x_pos: {prev_x_pos:.2f}  Reward: {reward:.2f}"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    width, height = text_size
    img = np.zeros((height + 20, width + 20, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, height + 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Position and Reward Info", img)
    cv2.waitKey(1)

# ===========================
# 7. Training Function
# ===========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4  # Frame stack of 4
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START

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
        prev_x_pos = 0

        for step in range(1, MAX_STEPS + 1):
            env.render()

            # ε-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, device=device).unsqueeze(0))
                    action = q_values.max(1)[1].item()

            # Apply frame skipping
            accum_reward = 0
            for i in range(FRAME_SKIP):
                next_state, _, done, info = env.step(action)
                accum_reward += compute_reward(info, step, prev_x_pos)
                prev_x_pos = info.get('x_pos', prev_x_pos)
                if done:
                    break

            # Show current info
            display_info(prev_x_pos, prev_x_pos, total_reward)

            # Update frame stack
            frames.append(preprocess(next_state))
            next_state = np.stack(frames, axis=0)

            memory.push(state, action, accum_reward, next_state, done)
            state = next_state
            total_reward += accum_reward

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

            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Epsilon decay
        if episode < EPSILON_DECAY_EPISODES:
            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES
        else:
            epsilon = EPSILON_END

        duration = time.time() - episode_start
        print(f"Episode {episode}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f} | Time: {duration:.2f}s")

        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"Model saved: {SAVE_PATH}")

    torch.save(policy_net.state_dict(), SAVE_PATH)
    print(f"Training finished. Final model saved to: {SAVE_PATH}")
    env.close()
    cv2.destroyAllWindows()

# ===========================
# 8. Image Preprocessing
# ===========================
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(obs, dtype=np.float32) / 255.0

# ===========================
# 9. Run Training
# ===========================
if __name__ == '__main__':
    train()
