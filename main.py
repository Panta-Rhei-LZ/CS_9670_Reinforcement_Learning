import torch
import os
import glob
import re
from datetime import datetime

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent
from ppo_agent import PPOAgent
from wrappers import apply_wrappers
from nes_py.wrappers import JoypadSpace

from utils import *

# Configuration
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
NUM_OF_EPISODES = 50_000
SAVE_INTERVAL = 20  # Save every 20 episodes
MAX_MODELS = 10  # Maximum number of non-milestone models to keep
MILESTONE_INTERVAL = 5000  # Models at these intervals are considered milestones

# Setup model directory
model_base_path = "models"
model_path = os.path.join(model_base_path, get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)


# Function to extract episode number from model filename
def get_episode_number(filename):
    match = re.search(r'model_(\d+)_iter\.pt', filename)
    if match:
        return int(match.group(1))
    return 0


# Function to find the newest model
def find_newest_model():
    all_model_folders = glob.glob(os.path.join(model_base_path, "*"))
    if not all_model_folders:
        return None, None

    all_models = []
    for folder in all_model_folders:
        models = glob.glob(os.path.join(folder, "model_*_iter.pt"))
        all_models.extend(models)

    if not all_models:
        return None, None

    # Sort by creation time (newest first)
    newest_model = max(all_models, key=os.path.getctime)
    episode_num = get_episode_number(newest_model)

    return newest_model, episode_num


# Function to clean up old models (keeping milestones)
def cleanup_models(current_folder):
    models = glob.glob(os.path.join(current_folder, "model_*_iter.pt"))

    # Separate milestone and regular models
    milestone_models = []
    regular_models = []

    for model in models:
        episode_num = get_episode_number(model)
        if episode_num % MILESTONE_INTERVAL == 0:
            milestone_models.append(model)
        else:
            regular_models.append(model)

    # Sort regular models by episode number (oldest first)
    regular_models.sort(key=get_episode_number)

    # Remove oldest models if we have more than MAX_MODELS
    models_to_remove = regular_models[:-MAX_MODELS] if len(regular_models) > MAX_MODELS else []

    for model in models_to_remove:
        os.remove(model)
        print(f"Removed old model: {os.path.basename(model)}")


# Check CUDA availability
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

# Initialize environment
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Initialize agent
agent = PPOAgent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Find and load the newest model if it exists
start_episode = 0
newest_model_path, newest_episode = find_newest_model()

if newest_model_path and not SHOULD_TRAIN:
    print(f"Loading model: {newest_model_path}")
    agent.load_model(newest_model_path)
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0
elif newest_model_path and SHOULD_TRAIN:
    print(f"Continuing training from model: {newest_model_path}")
    agent.load_model(newest_model_path)
    start_episode = newest_episode
    print(f"Starting from episode {start_episode}")

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

# Training loop
for i in range(start_episode, NUM_OF_EPISODES):
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0

    # Clear agent memory at the start of episode (for PPO)
    if isinstance(agent, PPOAgent):
        agent.memory = []

    while not done:
        a, _ = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(a)
        total_reward += reward

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            # Only learn after each step for DQN, not for PPO
            if not isinstance(agent, PPOAgent):
                agent.learn()

        state = new_state

    # For PPO, learn after the episode is complete
    if SHOULD_TRAIN and isinstance(agent, PPOAgent):
        agent.learn()

    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Learn step counter:", agent.learn_step_counter)
    if isinstance(agent, Agent):  # Only show replay buffer size for DQN
        print("Size of replay buffer:", len(agent.replay_buffer))

    # Save model every SAVE_INTERVAL episodes or if it's a milestone
    if SHOULD_TRAIN and ((i + 1) % SAVE_INTERVAL == 0 or (i + 1) % MILESTONE_INTERVAL == 0):
        model_file = os.path.join(model_path, f"model_{i + 1}_iter.pt")
        agent.save_model(model_file)
        print(f"Saved model: {model_file}")

        # Clean up old models after saving
        cleanup_models(model_path)

env.close()