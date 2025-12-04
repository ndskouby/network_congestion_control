from gymnasium.envs.registration import register

register(
    id="CongestionControl-v0",
    entry_point="envs.cc_env:SimpleCongestionEnv"
)