import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

class SimpleCongestionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(
            self,
            config_file='configs/default_env.yaml',  # ← NEW: configurable
            dt=0.5,
            rate_step_frac=0.02,
            alpha=2.0,
            beta=0.5,
            gamma=0.1,  # ← Already fixed to 0.1
            max_epsiode_steps=500
    ):
        super().__init__()
        self.dt = dt
        self.rate_step_frac = rate_step_frac
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_steps = max_epsiode_steps
        self.config_file = config_file

        try:
            with open(self.config_file, 'r') as file:
                self.cfg = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: config file {self.config_file} not found")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")

        # action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # observation space: 5D bounded vector
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1000.0, 1.0, 5.0, 1.0, 1000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ... rest of the methods stay the same ...
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        super().reset(seed=seed)

        # initialize randomized link parameters
        self.link_capacity_mbps = float(self.np_random.uniform(self.cfg['bw_min'], self.cfg['bw_max']))
        self.base_rtt_s = float(self.np_random.uniform(self.cfg['rtt_min'], self.cfg['rtt_max']))/1000
        self.queue_capacity_pkts = int(self.np_random.integers(self.cfg['qmin'], self.cfg['qmax']))
        self.pkt_size_bytes = int(self.np_random.integers(self.cfg['pkt_min'], self.cfg['pkt_max']))
        self.loss_prob = float(self.np_random.uniform(self.cfg['loss_min'], self.cfg['loss_max']))

        self.cross_on = False
        self.cross_unitl = 0

        self.sending_rate_mbps = float(self.link_capacity_mbps * 0.5)
        self.queue_occupancy = 0
        self.smoothed_rtt = self.base_rtt_s
        self.smoothed_loss = 0.0
        self.recent_throughput = 0.0
        self.current_step = 0

        obs = np.array([
            self.sending_rate_mbps,
            float(self.queue_occupancy) / float(self.queue_capacity_pkts),
            float(self.smoothed_rtt),
            float(self.smoothed_loss),
            float(self.recent_throughput)
        ], dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        # apply action
        if isinstance(action, np.ndarray):
            self.sending_rate_mbps = float(action)
        else:
            if action == 0:
                self.sending_rate_mbps *= (1.0 - 10*self.rate_step_frac)
            elif action == 1:
                self.sending_rate_mbps *= (1.0 - self.rate_step_frac)
            elif action == 3:
                self.sending_rate_mbps *= (1 + self.rate_step_frac)
            elif action == 4:
                self.sending_rate_mbps *= (1 + 10*self.rate_step_frac)
        
        self.sending_rate_mbps = float(np.clip(
            self.sending_rate_mbps,
            self.observation_space.low[0],
            self.observation_space.high[0]
        ))

        offered_bits = self.sending_rate_mbps * 1e6 * self.dt
        pkt_bits = self.pkt_size_bytes * 8
        offered_pkts = int(offered_bits // pkt_bits)

        capacity_bits = self.link_capacity_mbps * 1e6 * self.dt
        service_pkts = int(capacity_bits // pkt_bits)

        space = max(0, self.queue_capacity_pkts - self.queue_occupancy)
        enqueued = min(space, offered_pkts)
        dropped = offered_pkts - enqueued
        
        if self.np_random.random() < self.loss_prob:
            dropped_extra = int(0.1 * enqueued)
            enqueued = max(0, enqueued - dropped_extra)
            dropped += dropped_extra

        if self.cfg['enable_cross_traffic']:
            if (not self.cross_on) and self.np_random.random() < self.cfg['cross_start_prob']:
                self.cross_on = True
                self.cross_unitl = self.current_step + int(self.np_random.integers(self.cfg['cross_dur_min'], self.cfg['cross_dur_max']))
            if self.cross_on:
                extra_load_pkts = int(self.cfg['cross_rate']/ (self.pkt_size_bytes*8) * self.dt * 1e6)
                service_pkts = max(0, service_pkts - extra_load_pkts)
                if self.current_step >= self.cross_unitl:
                    self.cross_on = False

        served = min(self.queue_occupancy + enqueued, service_pkts)
        self.queue_occupancy = int(max(0, self.queue_occupancy + enqueued - served))

        throughput_bps = served * pkt_bits / (self.dt + 1e-9)
        throughput_mbps = throughput_bps / 1e6

        base_rtt = self.base_rtt_s
        queue_occ_bits = self.queue_occupancy * pkt_bits
        queue_delay_s = queue_occ_bits / (max(1.0, self.link_capacity_mbps * 1e6))
        rtt = base_rtt + queue_delay_s

        loss_frac = 0.0
        if offered_pkts > 0:
            loss_frac = dropped / offered_pkts
        
        self.smoothed_rtt = 0.9 * self.smoothed_rtt + 0.1 * rtt
        self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * loss_frac
        self.recent_throughput = 0.9 * self.recent_throughput + 0.1 * throughput_mbps

        utilization = self.recent_throughput / self.link_capacity_mbps
        normalized_rtt = self.smoothed_rtt / self.base_rtt_s

        reward = (
            self.alpha * utilization * 10
            - self.beta * normalized_rtt 
            - self.gamma * (self.smoothed_loss * 100)
        )

        # utilization = self.recent_throughput / self.link_capacity_mbps
        # normalized_rtt = min(self.smoothed_rtt / self.base_rtt_s, 5.0)  # Cap at 5x

        # reward = (
        #     10.0 * utilization                    # 0-10 scale
        #     - 1.0 * max(0, normalized_rtt - 1.0)  # Only penalize RTT above baseline
        #     - 5.0 * self.smoothed_loss            # 0-5 for 100% loss
        # )

        obs = np.array([
            self.sending_rate_mbps,
            float(self.queue_occupancy) / float(self.queue_capacity_pkts),
            float(self.smoothed_rtt),
            float(self.smoothed_loss),
            float(self.recent_throughput)
        ], dtype=np.float32)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = {
            "throughput_mbps": throughput_mbps,
            "served_pkts": served,
            "dropped_pkts": dropped,
            "rtt_s": rtt,
            "queue_pkts": self.queue_occupancy
        }
        return obs, reward, done, False, info

    def render(self, mode="human"):
        utilization = self.recent_throughput / self.link_capacity_mbps
        normalized_rtt = self.smoothed_rtt / self.base_rtt_s
        reward = (
            self.alpha * utilization * 10
            - self.beta * normalized_rtt 
            - self.gamma * (self.smoothed_loss * 100)
        )

        # utilization = self.recent_throughput / self.link_capacity_mbps
        # normalized_rtt = min(self.smoothed_rtt / self.base_rtt_s, 5.0)  # Cap at 5x

        # reward = (
        #     10.0 * utilization                    # 0-10 scale
        #     - 1.0 * max(0, normalized_rtt - 1.0)  # Only penalize RTT above baseline
        #     - 5.0 * self.smoothed_loss            # 0-5 for 100% loss
        # )

        print(f"step={self.current_step:4d} reward={reward:.4f} rate={self.sending_rate_mbps:.2f}Mbps q={self.queue_capacity_pkts} rtt={self.smoothed_rtt*1000}ms loss={self.smoothed_loss:.3f}")