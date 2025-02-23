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
GAMMA = 0.99                # 折扣因子
LR = 1e-4                   # 学习率
EPSILON_START = 1.0         # 初始探索率
EPSILON_END = 0.1           # 最终探索率
EPSILON_DECAY_EPISODES = 500  # 在500个回合内线性衰减探索率
BATCH_SIZE = 64             # 批大小
MEMORY_SIZE = 50000         # 经验回放容量
TARGET_UPDATE = 1000        # 目标网络更新频率（步数）
MAX_STEPS = 5000            # 每回合最大步数（指动作选择次数）
NUM_EPISODES = 1000         # 训练回合数
SAVE_PATH = './train.model' # 模型保存路径
FRAME_SKIP = 4              # 重复同一动作的帧数

# ===========================
# 2️⃣ 创建环境 (兼容旧版 gym)
# ===========================
env = gym.make('SuperMarioBros-v0')  # 无 render_mode
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.unwrapped.fast_forward = 4        # 加速 4 倍（如果支持）
# 这里我们将内部的 frameskip 参数可以保持原样，
# 但我们在训练循环中额外重复执行同一动作来实现 frame skip
# env.unwrapped.frameskip = 1
# env.unwrapped.silence = True        # 可选择静音
env.metadata['render_fps'] = 240       # FPS设置
env.unwrapped.clock_mode = False      # 禁用延迟限制
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
def compute_reward(info, step, prev_x_pos):
    reward = 0
    x_pos = info['x_pos']  # 当前的位置

    # 根据前进距离给予奖励（鼓励前进）
    distance_reward = (x_pos - prev_x_pos) * 0.2
    reward += distance_reward

    # 对跳跃行为给予奖励
    if info.get('jumping', False):
        reward += 0.2

    # 通过关卡奖励
    if info.get('flag_get', False):
        reward += 1000

    # 掉入陷阱或失去生命的惩罚
    if info.get('life_lost', False):
        reward -= 500

    # 遇到敌人惩罚
    if info.get('enemy_nearby', False):
        reward -= 100

    # 小惩罚，鼓励尽快行动
    reward -= 0.02

    return reward

# ===========================
# 显示信息函数
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
        prev_x_pos = 0

        for step in range(1, MAX_STEPS + 1):
            # 启用环境渲染
            env.render()

            # ε-贪婪策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, device=device).unsqueeze(0))
                    action = q_values.max(1)[1].item()

            # 使用 frame skip 逻辑：重复执行相同动作 FRAME_SKIP 次
            accum_reward = 0
            for i in range(FRAME_SKIP):
                next_state, _, done, info = env.step(action)
                # 计算奖励并累加（使用当前 step 和 prev_x_pos）
                accum_reward += compute_reward(info, step, prev_x_pos)
                # 更新 prev_x_pos 为最新值
                prev_x_pos = info.get('x_pos', prev_x_pos)
                if done:
                    break

            # 调用显示信息函数（显示最新 x_pos 和累计奖励）
            display_info(prev_x_pos, prev_x_pos, total_reward)

            # 更新多帧堆叠：将最后一次观察加入队列
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

        if episode < EPSILON_DECAY_EPISODES:
            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES
        else:
            epsilon = EPSILON_END

        duration = time.time() - episode_start
        print(f"Episode {episode}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f} | Time: {duration:.2f}s")

        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"✅ 模型已保存: {SAVE_PATH}")

    torch.save(policy_net.state_dict(), SAVE_PATH)
    print(f"🎉 训练完成，模型已保存为: {SAVE_PATH}")
    env.close()
    cv2.destroyAllWindows()

# ===========================
# 7️⃣ 图像预处理 (降低维度加速训练)
# ===========================
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
