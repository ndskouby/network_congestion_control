from gymnasium.envs.registration import register

# Register main environment
register(
    id="CongestionControl-v0",
    entry_point="envs.cc_env:SimpleCongestionEnv"
)

# Register curriculum environments
register(
    id="CongestionControl-Easy-v0",
    entry_point="envs.cc_env:SimpleCongestionEnv",
    kwargs={'config_file': 'configs/easy_env.yaml'}
)

register(
    id="CongestionControl-Medium-v0",
    entry_point="envs.cc_env:SimpleCongestionEnv",
    kwargs={'config_file': 'configs/medium_env.yaml'}
)

register(
    id="CongestionControl-Hard-v0",
    entry_point="envs.cc_env:SimpleCongestionEnv",
    kwargs={'config_file': 'configs/hard_env.yaml'}
)