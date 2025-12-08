import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def create_model():
    """Create a new PPO model."""
    env = gym.make("CongestionControl-v0")
    env = Monitor(env)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )
    return model


def train_model(model_name, n_steps=1_000_000):
    """Train a new model and save it with a specific name."""
    print(f"Start training {model_name}...")
    
    # Create fresh environment for this training run
    train_env = gym.make("CongestionControl-v0")
    train_env = Monitor(train_env)
    
    eval_env = gym.make("CongestionControl-v0")
    eval_env = Monitor(eval_env)
    
    # Create eval callback that saves to the specific model name
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}/",  # Saves best to subdir
        log_path=f"./logs/{model_name}/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=f"./tensorboard/{model_name}/"  # Separate TB logs
    )
    
    # Train
    model.learn(
        total_timesteps=n_steps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"models/{model_name}/final")
    
    print(f"Training complete for {model_name}")
    print(f"  Final model: models/{model_name}/final.zip")
    print(f"  Best model: models/{model_name}/best_model.zip")
    
    return model


def smoke_test_reward(model_path):
    """Test a saved model."""
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)
    
    # Create test environment
    test_env = gym.make("CongestionControl-v0")
    test_env = Monitor(test_env)
    
    test_rewards = []
    for episode in range(10):
        obs, info = test_env.reset()

        # Print what scenario we got
        # print(f"\n  Episode {episode+1} config:")
        # print(f"    Bandwidth: {test_env.unwrapped.link_capacity_mbps:.1f} Mbps")
        # print(f"    RTT: {test_env.unwrapped.base_rtt_s*1000:.1f} ms")
        # print(f"    Queue: {test_env.unwrapped.queue_capacity_pkts} pkts")

        total_reward = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            if done or truncated:
                break
        test_rewards.append(total_reward)
        print(f"  Episode {episode+1}: {total_reward:.2f}")

    print(f"\nAverage reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Min: {np.min(test_rewards):.2f}, Max: {np.max(test_rewards):.2f}")
    return np.mean(test_rewards)


def _smoke_test_reward(model_path):
    """Test a saved model."""
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)
    
    # Create test environment
    test_env = gym.make("CongestionControl-v0")
    test_env = Monitor(test_env)
    
    test_rewards = []
    for episode in range(10):
        obs, _ = test_env.reset(seed=42)  # Same seed = same scenario
        
        total_reward = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            if done or truncated:
                break
        test_rewards.append(total_reward)
    
    print(f"Fixed scenario: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")

    print(f"\nAverage reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Min: {np.min(test_rewards):.2f}, Max: {np.max(test_rewards):.2f}")
    return np.mean(test_rewards)


def compare_models(model_paths):
    """Compare multiple models."""
    print("="*70)
    print("COMPARING MODELS")
    print("="*70)
    
    results = {}
    for name, path in model_paths.items():
        print(f"\n{name}:")
        avg_reward = smoke_test_reward(path)
        results[name] = avg_reward
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:30s}: {reward:10.2f}")
    print("="*70)


if __name__ == "__main__":
    # ========== TRAINING EXAMPLES ==========
    
    # Train 500k model
    # train_model("model-500k", n_steps=500_000)
    
    # Train 1M model
    train_model("model-1M", n_steps=1_000_000)
    
    # Train 2M model
    # train_model("model-2M", n_steps=2_000_000)
    
    
    # ========== TESTING EXAMPLES ==========
    
    # Test single model
    # smoke_test_reward("models/model-500k/best_model")
    smoke_test_reward("models/model-1M/best_model")
    
    # Compare multiple models
    # compare_models({
    #     "500k steps (best)": "models/model-500k/best_model",
    #     "500k steps (final)": "models/model-500k/final",
    #     "1M steps (best)": "models/model-1M/best_model",
    #     "1M steps (final)": "models/model-1M/final",
    #     "2M steps (best)": "models/model-2M/best_model",
    # })