    # def step(self, action):
    #     """Execute a trade and update environment state."""
    #     self.current_step += 1
    #     if self.current_step >= len(self.data) - 1:
    #         done = True
    #     else:
    #         done = False

    #     current_price = self.data.iloc[self.current_step]['Close']
    #     current_direction_prediction = self.data.iloc[self.current_step]['Direction']
    #     reward = 0

    #     # Execute action
    #     if action == 1:  # Buy
    #         if self.balance >= current_price:
    #             self.actions_history.append(1)

    #             self.shares_held += 1

    #             self.info["Hold"].append(current_price)

    #             self.balance -= current_price
    #             self.total_profit -= current_price
                
    #             if current_direction_prediction == 1:
    #                 rand_num = np.random.rand()
    #                 if rand_num < self.success_rate:
    #                     reward = 10
    #                 else:
    #                     reward = -10
    #             else:
    #                 rand_num = np.random.rand()
    #                 if rand_num < self.success_rate:
    #                     reward = -100

    #     elif action == 2:  # Sell
    #         if self.shares_held > 0:
    #             self.actions_history.append(2)

    #             purchase_val = sum(self.info["Hold"])
    #             sell_val = current_price * self.shares_held
    #             self.info["Hold"] = []
                
    #             if current_direction_prediction == -1:
    #                 rand_num = np.random.rand()
    #                 if rand_num < self.success_rate:
    #                     reward = 10
    #                 else:
    #                     reward = -10
    #             else:
    #                 rand_num = np.random.rand()
    #                 if rand_num < self.success_rate:
    #                     reward = -100

    #             reward += sell_val - purchase_val

    #             if sell_val - purchase_val > 0:
    #                 reward += 2 * (sell_val - purchase_val)
    #             else:
    #                 reward += sell_val - purchase_val
                    
    #             self.balance += current_price * self.shares_held
    #             self.total_profit += current_price * self.shares_held

    #             self.shares_held = 0
    #         else:
    #             if current_direction_prediction == -1:
    #                 reward = 10
    #             else:
    #                 reward = -100
    #     elif action == 0: # Hold
    #         self.actions_history.append(0)
    #         if current_direction_prediction == -1:
    #             reward = 10
    #         else:
    #             reward = -100

    #     # Update total profit
    #     self.balance_history.append(self.balance)
    #     self.profit_history.append(self.total_profit)

    #     truncated = False  # 🔥 Required for Gymnasium! Means episode wasn't forcefully stopped.

    #     return self._next_observation(), reward, done, truncated, self.info

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import matplotlib.pyplot as plt

class TradingEnv(gym.Env):
    def __init__(self, data: tuple[pd.DataFrame], initial_balance: int = 1000):
        super(TradingEnv, self).__init__()

        # Market data
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.total_profit = 0

        # Store historical values
        self.balance_history = [self.balance]
        self.profit_history = [self.total_profit]
        self.actions_history = []

        # All historical values for plotting
        self.all_balance_history = []
        self.all_profit_history = []
        self.all_actions_history = []

        # Info dict
        self.info = {}
        
        # Define Observation Space (Market Features + Account Info)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Define Action Space (1=Buy, 2=Sell)
        self.action_space = spaces.Discrete(2)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0

        if len(self.balance_history) > 1:
            self.all_balance_history.append(self.balance_history)
            self.all_profit_history.append(self.profit_history)
            self.all_actions_history.append(self.actions_history)

        self.balance_history = [self.balance]
        self.profit_history = [self.total_profit]
        self.actions_history = []

        observation = self._next_observation()
        self.info = {}
        return observation, self.info 
    
    def _next_observation(self):
        """Get the next state (market direction and account balance)."""
        obs = [
            self.data.iloc[self.current_step]['Direction'],  # Market direction (1 or -1)
            self.balance  # Account balance
        ]
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute a trade and update environment state."""
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        current_price = self.data.iloc[self.current_step]['Close']
        market_direction = self.data.iloc[self.current_step]['Direction']
        reward = 0

        if market_direction == 1 and self.balance >= current_price:
            self.shares_held += 2
            self.balance -= current_price * 2
            reward = 10  # Reward for buying in uptrend
            action = 1  # Buy
        elif market_direction == -1 and self.shares_held > 0:
            self.balance += self.data.iloc[self.current_step]['Close'] * self.shares_held
            self.shares_held = 0
            reward = 10  # Reward for selling in downtrend
            action = 2  # Sell
        else:
            action = 0
        
        self.balance_history.append(self.balance)
        self.profit_history.append(self.balance - self.initial_balance)
        self.actions_history.append(action)
        
        truncated = False
        return self._next_observation(), reward, done, truncated, self.info
    
    def render(self):
        """Print trading information."""
        if self.actions_history[-1] == 1:
            act = "BUY"
        elif self.actions_history[-1] == 2:
            act = "SELL"
        else:
            act = "HOLD"
        
        if self.data.iloc[self.current_step]['Direction'] == 1:
            sig = "BUY"
        elif self.data.iloc[self.current_step]['Direction'] == -1:
            sig = "SELL"
        
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}, Signal: {sig}, Action: {act}")
