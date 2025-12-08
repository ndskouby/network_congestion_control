import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create env
env = gym.make("CongestionControl-v0")
env = Monitor(env)

# Evaluation environment
eval_env = gym.make("CongestionControl-v0")
eval_env = Monitor(eval_env)

# Set up evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=5000,  # Evaluate every 5000 steps
    deterministic=True,
    render=False
)


def create_model():
    # Create agent - PPO
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


def train_model(model, model_name=None, n_steps=1_000_000):
    print("Start training...")
    
    if model_name is None:
        model_name = "models/model-latest"
    else:
        model_name = f"models/{model_name}"

    model.learn(
        total_timesteps=n_steps,
        callback=eval_callback,
        progress_bar=True
    )

    model.save(model_name)

    print(f"Training complete for {model_name}")


def smoke_test_reward(model):
    print("\nTesting trained model over 10 episodes...")
    test_rewards = []
    for episode in range(10):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        test_rewards.append(total_reward)
        print(f"  Episode {episode+1}: {total_reward:.2f}")

    print(f"\nAverage reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Min: {np.min(test_rewards):.2f}, Max: {np.max(test_rewards):.2f}")


if __name__ == "__main__":
    # For training a new model
    # model = create_model()
    # train_model(model)

    # For testing an existing one
    model = PPO.load("models/best_model")
    smoke_test_reward(model)