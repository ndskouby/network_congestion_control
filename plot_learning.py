import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results
import pandas as pd
import numpy as np

window = 100

dqn_df = load_results("logs/dqn/")
dqn_rewards = pd.Series(dqn_df["r"].values)
dqn_smooth = dqn_rewards.rolling(window=window, min_periods=window//2).mean()
dqn_eps = np.arange(len(dqn_rewards))


ppo_df = load_results("logs/ppo/")
ppo_rewards = pd.Series(ppo_df["r"].values)
ppo_smooth = ppo_rewards.rolling(window=window, min_periods=window//2).mean()
ppo_eps = np.arange(len(ppo_rewards))

plt.figure()

plt.plot(dqn_eps, dqn_rewards, alpha=0.25, label="DQN (raw)", color="blue")
plt.plot(ppo_eps, ppo_rewards, alpha=0.25, label="PPO (raw)", color="orange")

plt.plot(dqn_eps, dqn_smooth, label=f"DQN (rolling mean, {window})", color="blue")
plt.plot(ppo_eps, ppo_smooth, label=f"PPO (rolling mean, {window})", color="orange")

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Learning Curve")
plt.legend()
plt.show()