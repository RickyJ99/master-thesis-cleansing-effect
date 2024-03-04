# %% [markdown]
# # Import dependencies

# %%


# %% [markdown]
# # Load Enviroment

# %%
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# %% [markdown]
# # Environments

# %%
environment_name = "CartPole-v1"  # case-sensitive
env = gym.make(environment_name, render_mode="rgb_array")  # making the env

# %% [markdown]
# # Understanding the environment
# Episodes = Think of an episode as one full game within the environment.

# %%
episodes = 5  # test five times
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info, info1 = env.step(action)
        score += reward
    print("Episode:{} Score:{}".format(episode, score))
env.close()
