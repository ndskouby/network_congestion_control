import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleCongestionEnv(gym.env):
    metadata = {"render_modes": ["human"]}
    def __init__(
            self,
            link_capacity_mbps=10.0,
            queue_capacity_pkts=100,
            pkt_size_bytes=1500,
            dt=0.5, # step duration (s)
            rate_step_frac=0.2, # rate change multiplier
            alpha=1.0, beta=0.5, gamma=1.0, # reward parameters
            max_epsiode_steps=500
    ):
        super.__init__()
        self.link_capacity_mbps = link_capacity_mbps
        self.queue_capacity_pkts = queue_capacity_pkts
        self.pkt_size_bytes = pkt_size_bytes
        self.dt = dt
        self.rate_step_frac = rate_step_frac
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_steps = max_epsiode_steps

        # action space: decrease, hold, increase
        self.action_space = spaces.Discrete(3)

        # observation space: 5D bounded vector
        # send_rate(mbps), queue_norm[0,1], smoothed_rtt(s), loss_rate[0,1], throughput(mbps)
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1000.0, 1.0, 5.0, 1.0, 1000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)



