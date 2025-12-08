import numpy as np

class AIMDAgent:
    def __init__(self, alpha=1.0, beta=0.5, min_rate=0.1, max_rate=200.0):
        # initialize agent parameters
        self.alpha = alpha
        self.beta = beta
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.sending_rate = 1.0

    def reset(self, initial_rate = 1.0):
        # set starting rate
        self.sending_rate = initial_rate

    def act(self, obs):
        # pull the loss from the observation
        observed_loss = obs[3]

        # when loss is observed multiplicatively decrease by beta
        if observed_loss > 0.0:
            self.sending_rate *= self.beta
        else:
            # when no loss observed additively increase by alpha
            self.sending_rate += self.alpha
        
        self.sending_rate = np.clip(self.sending_rate, self.min_rate, self.max_rate)

        # update the sending rate directly
        return np.array([self.sending_rate], dtype=np.float32)