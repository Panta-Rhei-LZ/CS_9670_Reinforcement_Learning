import os
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
# 🚀 1️⃣ 提升游戏运行速度 (10倍)
# ===========================
os.environ['NES_PY_SPEED_MODE'] = '10'  # 内部加速10倍
os.environ['NES_PY_NO_LIMIT'] = '1'  # 取消帧率限制
os.environ['NES_PY_NO_AUDIO'] = '1'  # 禁用音频提高速度


# ===========================
# 🚀 2️⃣ 自定义 FrameSkip 包装器 (兼容旧版 gym)
# ===========================
class CustomFrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        """每隔 skip 帧进行一次动作，提高训练速度"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """重复执行同一动作 skip 次"""
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        """重置环境"""
        return self.env.reset(**kwargs)


# ===========================
# 🧠 3️⃣ 配置超参数
# ===========================
GAMMA = 0.99  # 折扣因子
LR = 1e-4  # 学习率
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.1  # 最终探索率
EPSILON_DECAY = 5000  # 探索率衰减步数 (加快探索衰减)
BATCH_SIZE = 64  # 批大小
MEMORY_SIZE = 50000  # 经验回放容量
TARGET_UPDATE = 500  # 目标网络更新频率 (更频繁更新)
MAX_STEPS = 3000  # 每回合最大步数
NUM_EPISODES = 500  # 训练回合数 (快速实验)
SAVE_PATH = './train.model'  # 保存模型路径


# ===========================
# 🎮 4️⃣ 创建游戏环境
# ===========================
def create_env(render_mode=False):
    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # ⚡ 提升游戏运行速度
    env.unwrapped.clock_mode = False  # 移除NES时钟限制
    env.unwrapped.silence = True  # 禁用音频 (提升速度)
    env.metadata['render_fps'] = 600  # 设置更高帧率 (600FPS)

    # 🚀 每4帧决策一次 (提升训练速度)
    env = CustomFrameSkip(env, skip=4)  # 每4帧动作一次

    # ❌ 训练时禁用渲染
    if not render_mode:
        env.render = lambda *args, **kwargs: None

    return env


env = create_env()

# ===========================
# 💻 5️⃣ 检查是否在用GPU
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===========================
# 🧠 6️⃣ DQN 神经网络定义
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
# 📂 7️⃣ Replay Buffer (经验回放)
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
# 🏆 8️⃣ 奖励函数 (激励快速通关)
# ===========================
def compute_reward(info, step):
    """奖励设计：优先学习越过障碍和快速通关"""
    reward = info['score'] * 0.01  # 将分数作为基础奖励

    # 多奖励前进距离 (鼓励跑得更远)
    reward += (info['x_pos'] - info.get('prev_x_pos', 0)) * 0.1
    info['prev_x_pos'] = info['x_pos']

    # 如果卡在同一位置太久：扣分 (鼓励尝试跳跃)
    if step % 100 == 0 and info['x_pos'] == info.get('prev_x_pos', 0):
        reward -= 5  # 长时间卡住扣分

    # 通关大额奖励
    if info.get('flag_get', False):
        reward += 1000  # 通关大额奖励

    return reward


# ===========================
# 🧠 9️⃣ 训练部分 (DQN)
# ===========================
def train():
    policy_net = DQN(4, env.action_space.n).to(device)
    target_net = DQN(4, env.action_space.n).to(device)
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

        # 多帧堆叠 (Frame Stack)
        frames = deque(maxlen=4)
        for _ in range(4):
            frames.append(preprocess(state))
        state = np.stack(frames, axis=0)

        for step in range(1, MAX_STEPS + 1):
            # 🚀 游戏加速后运行更快
            if random.random() < epsilon:
                action = env.action_space.sample()  # 随机探索
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, device=device).unsqueeze(0))
                    action = q_values.max(1)[1].item()

            next_state, reward, done, info = env.step(action)
            reward = compute_reward(info, step)

            frames.append(preprocess(next_state))
            next_state = np.stack(frames, axis=0)

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = (
                    states.to(device),
                    actions.to(device),
                    rewards.to(device),
                    next_states.to(device),
                    dones.to(device),
                )

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 🎯 更频繁地更新目标网络
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
        duration = time.time() - episode_start
        print(
            f"Episode {episode}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f} | Time: {duration:.2f}s")

        if episode % 50 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"✅ 已保存模型: {SAVE_PATH}")

    # 🏁 训练结束后保存最终模型
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print(f"🎉 训练完成，模型已保存为: {SAVE_PATH}")
    env.close()


# ===========================
# 📊 10️⃣ 图像预处理 (降低维度加速训练)
# ===========================
def preprocess(obs):
    """将环境图像灰度化并缩放"""
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(obs, dtype=np.float32) / 255.0


# ===========================
# 🚀 11️⃣ 启动训练
# ===========================
if __name__ == '__main__':
    train()
