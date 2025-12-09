import numpy as np
import gymnasium as gym
import envs
import matplotlib.pyplot as plt

from agents.aimd_agent import AIMDAgent
from stable_baselines3 import DQN, PPO


def run_aimd_agent(env, agent, seed, max_steps=500):
    obs, _ = env.reset(seed=int(seed))
    agent.reset()

    total_reward = 0
    throughputs = []
    rtts = []
    losses = []

    for _ in range(max_steps):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        throughputs.append(obs[4])
        rtts.append(obs[2])
        losses.append(obs[3])

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "throughput": np.mean(throughputs),
        "rtt": np.mean(rtts),
        "loss": np.mean(losses),
    }


def run_rl_agent(env, model_class, model_path, seed, max_steps=500):
    try:
        model = model_class.load(model_path, device="cpu")
    except:
        print(f"Could not load RL model at {model_path}, skipping.")
        return None

    obs, _ = env.reset(seed=int(seed))

    total_reward = 0
    throughputs = []
    rtts = []
    losses = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        throughputs.append(obs[4])
        rtts.append(obs[2])
        losses.append(obs[3])

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "throughput": np.mean(throughputs),
        "rtt": np.mean(rtts),
        "loss": np.mean(losses),
    }

def run_experiment(N=10, max_steps=500):
    env = gym.make("CongestionControl-v0")

    metrics = ["reward", "throughput", "rtt", "loss"]

    results = {
        "AIMD": {m: [] for m in metrics},
        "DQN":  {m: [] for m in metrics},
        "PPO":  {m: [] for m in metrics},
    }

    aimd = AIMDAgent(alpha=0.5, beta=0.5)

    seeds = np.arange(1, N + 1)

    print(f"Running Experiment (N = {N})")

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{N}, seed={seed} ---")

        aimd_res = run_aimd_agent(env, aimd, seed, max_steps)
        for m in metrics:
            results["AIMD"][m].append(aimd_res[m])

        dqn_res = run_rl_agent(env, DQN, "models/dqn_cc.zip", seed, max_steps)
        if dqn_res:
            for m in metrics:
                results["DQN"][m].append(dqn_res[m])

        ppo_res = run_rl_agent(env, PPO, "models/ppo_cc.zip", seed, max_steps)
        if ppo_res:
            for m in metrics:
                results["PPO"][m].append(ppo_res[m])

    env.close()
    return results

def plot_comparison(results, metric, ylabel):
    algorithms = ["AIMD", "DQN", "PPO"]
    data = [results[alg][metric] for alg in algorithms]

    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]

    x = np.arange(len(algorithms))

    plt.figure(figsize=(7, 5))
    plt.bar(x, means, yerr=stds, capsize=7, alpha=0.8)
    plt.xticks(x, algorithms)
    plt.ylabel(ylabel)
    plt.title(f"{metric} (N=100)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"plots/{metric}/{metric}.png")


results = run_experiment(N=100)

print("\n\nFINAL STATISTICS (mean +/- std)")
for alg in ["AIMD", "DQN", "PPO"]:
    print(f"\n{alg}:")
    for m in results[alg]:
        arr = np.array(results[alg][m])
        if len(arr) > 0:
            print(f"  {m}: {arr.mean():.3f} +/- {arr.std():.3f}")

plot_comparison(results, "reward", "Total Episode Reward")
plot_comparison(results, "throughput", "Throughput (Mbps)")
plot_comparison(results, "rtt", "RTT (ms)")
plot_comparison(results, "loss", "Loss Rate")