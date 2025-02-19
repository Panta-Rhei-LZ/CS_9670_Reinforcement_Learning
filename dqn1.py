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
# 1️⃣ 配置超参数
# ===========================
GAMMA = 0.99               # 折扣因子
LR = 1e-4                  # 学习率
EPSILON_START = 1.0        # 初始探索率
EPSILON_END = 0.1          # 最终探索率
EPSILON_DECAY = 10000      # 探索率衰减步数
BATCH_SIZE = 64            # 批大小
MEMORY_SIZE = 50000        # 经验回放容量
TARGET_UPDATE = 1000       # 目标网络更新频率
MAX_STEPS = 5000           # 每回合最大步数
NUM_EPISODES = 1000        # 训练回合数
SAVE_PATH = './train.model' # 保存模型路径

# ===========================
# 2️⃣ 创建环境 (兼容旧版 gym)
# ===========================
env = gym.make('SuperMarioBros-v0')  # ⚠️ 无 render_mode
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.unwrapped.fast_forward = 4      # 加速 4 倍
env.unwrapped.frameskip = 2         # 每 2 帧渲染一次
#env.unwrapped.silence = True        # 静音
env.metadata['render_fps'] = 120    # 120 FPS
env.unwrapped.clock_mode = False    # 禁用延迟限制
state = env.reset()

# ===========================
# 3️⃣ DQN 神经网络定义
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
# 4️⃣ Replay Buffer (经验回放)
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
# 5️⃣ 奖励函数 (核心部分)
# ===========================
def compute_reward(info, step):
    """奖励设计：以通关速度为核心"""
    reward = info['score'] * 0.01  # 将分数作为基础奖励
    reward += (info['x_pos'] - info.get('prev_x_pos', 0)) * 0.1
    info['prev_x_pos'] = info['x_pos']
    reward -= 0.1  # 每帧扣除小奖励，鼓励快速通关
    if info.get('flag_get', False):  # 通关大额奖励
        reward += 1000
    return reward

# ===========================
# 6️⃣ 训练部分 (DQN)
# ===========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4  # 使用多帧堆叠 (Frame Stack)
    output_dim = env.action_space.n

    # 创建主网络和目标网络
    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    steps_done = 0

    # 开始训练循环
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        episode_start = time.time()
        done = False

        # 初始化多帧堆叠 (Frame Stack)
        frames = deque(maxlen=4)
        for _ in range(4):
            frames.append(preprocess(state))
        state = np.stack(frames, axis=0)

        for step in range(1, MAX_STEPS + 1):
            env.render()  # ⚠️ 使用 render() 而非 render_mode

            # 1️⃣ ε-贪婪策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, device=device).unsqueeze(0))
                    action = q_values.max(1)[1].item()

            # 2️⃣ 执行动作
            next_state, reward, done, info = env.step(action)
            reward = compute_reward(info, step)

            # 3️⃣ 更新多帧堆叠
            frames.append(preprocess(next_state))
            next_state = np.stack(frames, axis=0)

            # 4️⃣ 保存经验
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            # 5️⃣ 训练网络
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                # 计算当前 Q 值
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # 计算目标 Q 值 (Double DQN)
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 6️⃣ 目标网络更新
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # 探索率衰减
        epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)

        duration = time.time() - episode_start
        print(f"Episode {episode}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f} | Time: {duration:.2f}s")

        # 每 100 回合保存一次模型
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"✅ 模型已保存: {SAVE_PATH}")

    # 最终保存模型
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print(f"🎉 训练完成，模型已保存为: {SAVE_PATH}")
    env.close()

# ===========================
# 7️⃣ 图像预处理 (降低维度加速训练)
# ===========================
import cv2
def preprocess(obs):
    """将环境图像灰度化并缩放"""
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(obs, dtype=np.float32) / 255.0

# ===========================
# 🔥 运行训练
# ===========================
if __name__ == '__main__':
    train()
