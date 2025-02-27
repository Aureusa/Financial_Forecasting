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
import itertools

class TradingEnv(gym.Env):
    def __init__(self, data: tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame], initial_balance: int = 1000):
        super(TradingEnv, self).__init__()

        # Unpack data
        close_data, asset_ticker, dir_predictions, predictions_std = self._unpack_data(data=data)

        # Convert to ndarray
        self.close_data = np.array(close_data)
        self.asset_ticker = np.array(asset_ticker)
        self.dir_predictions = np.array(dir_predictions)
        self.predictions_std = np.array(predictions_std)

        # Current step
        self.current_step = 0
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = {}
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

    def _unpack_data(
            self,
            data: tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]
        ) -> tuple[list,list,list,list]:
        # Unpack data
        close_data, asset_ticker, dir_predictions, predictions_std = data

        # Retrieve the info from the dicts
        close_data = close_data["Close"].tolist()
        asset_ticker = asset_ticker["Stocks"].tolist()
        dir_predictions = dir_predictions["Direction"].tolist()
        predictions_std = predictions_std["Uncertainties"].tolist()

        # Make all lists the same length as the longest one, filling with np.nan
        padded_close_data = list(itertools.zip_longest(*close_data, fillvalue=np.nan))
        padded_dir_predictions = list(itertools.zip_longest(*dir_predictions, fillvalue=np.nan))
        padded_predictions_std = list(itertools.zip_longest(*predictions_std, fillvalue=np.nan))

        # Convert back to list of lists
        close_data = [list(row) for row in zip(*padded_close_data)]
        dir_predictions = [list(row) for row in zip(*padded_dir_predictions)]
        predictions_std = [list(row) for row in zip(*padded_predictions_std)]

        return close_data, asset_ticker, dir_predictions, predictions_std
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = {}
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
            self.close_data[:,self.current_step],
            self.dir_predictions[:,self.current_step],
            self.predictions_std[:,self.current_step],
        ]
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute a trade and update environment state."""
        self.current_step += 1
        done = self.current_step >= len(self.close_data[0]) - 1

        current_prices = self.close_data[:,self.current_step]
        tickers = self.asset_ticker[:]
        dir_predictions = self.dir_predictions[:,self.current_step]
        predictions_std = self.predictions_std[:,self.current_step]

        action_list = []
        for num, tik in enumerate(tickers):
            tik_price = current_prices[num]

            # Check if value is NaN
            if np.isnan(tik_price):
                action_list.append(0)
                continue
            
            # Instantiates the dictionary
            if tik not in self.shares_held:
                self.shares_held[tik] = 0
            
            # Get the prediction of the direction and uncertainty
            tik_dir = dir_predictions[num]
            tik_std = predictions_std[num]

            if tik_dir == 1 and self.balance >= 100:
                # Compute the num of shares both
                frac_shares = 100 / tik_price

                # Add the shares to the portfolio
                self.shares_held[tik] += frac_shares

                # Substract the price from the balance
                self.balance -= 100

                # Declare a buying action
                action = 1 # 1 (BUY)
                action_list.append(action)
            elif tik_dir == -1 and self.shares_held[tik] > 0:
                # Add the price to the balance
                self.balance += tik_price * self.shares_held[tik]

                # Delete the shares from the portfolio
                self.shares_held[tik] = 0

                # Declare a selling action
                action = 2 # 2 (SELL)
                action_list.append(action)
            else:
                # Declare a hold action
                action = 0 # 0 (HOLD)
                action_list.append(action)

        if done:
            for num, tik in enumerate(tickers):
                tik_price = current_prices[num]

                # Check if value is NaN
                if np.isnan(tik_price):
                    action_list.append(0)
                    continue
                
                # Instantiates the dictionary
                if tik not in self.shares_held:
                    self.shares_held[tik] = 0
                
                # Get the prediction of the direction and uncertainty
                tik_dir = dir_predictions[num]
                tik_std = predictions_std[num]

                if self.shares_held[tik] > 0:
                    # Add the price to the balance
                    self.balance += tik_price * self.shares_held[tik]

                    # Delete the shares from the portfolio
                    self.shares_held[tik] = 0

                    # Declare a selling action
                    action = 2 # 2 (SELL)
                    action_list.append(action)

        
        self.balance_history.append(self.balance)
        self.profit_history.append(self.balance - self.initial_balance)
        self.actions_history.append(action_list)
        
        truncated = False
        return self._next_observation(), 0, done, truncated, self.info
    
    def render(self):
        """Print trading information."""      
        upper_border = "=" * 44

        msg = f"| Step: {self.current_step}"
        right_border = (len(upper_border) - len(msg) - 1) * " " +"|\n"
        msg = upper_border + "\n" + msg + right_border

        sep_line = len(upper_border) * "-" + "\n"

        # Upper line on Shares Held (----)
        msg += sep_line

        shares_held = "| Shares Held:"
        right_border = (len(upper_border) - len(shares_held) - 1) * " " +"|\n"
        shares_held += right_border

        # Add shares message
        msg += shares_held

        # Add all the stocks to the msg
        for key, value in self.shares_held.items():
            shares = f"| {key}: {value}"
            right_border = (len(upper_border) - len(shares) - 1) * " " +"|\n"
            shares += right_border

            msg += shares

        msg += sep_line

        # Signal
        signal = "| Signal:" 
        right_border = (len(upper_border) - len(signal) - 1) * " " +"|\n"
        signal += right_border

        # Create first line signal message
        msg += signal

        for num, tik in enumerate(self.asset_ticker):
            sig_ = self.dir_predictions[:,self.current_step]
            if sig_[num] == 1:
                sig_tik = f"| {tik}: BUY"
            else:
                sig_tik = f"| {tik}: SELL"

            right_border = (len(upper_border) - len(sig_tik) - 1) * " " +"|\n"
            sig_tik += right_border

            msg += sig_tik

        # Separation line between signal and action
        msg += sep_line

        action = "| Action:"
        right_border = (len(upper_border) - len(action) - 1) * " " +"|\n"
        action += right_border

        # Create first line action message
        msg += action

        # Populate action message
        for num, tik in enumerate(self.asset_ticker):
            act = self.actions_history[self.current_step - 1]
            if act[num] == 0:
                act_msg = f"| {tik}: HOLD"
            elif act[num] == 1:
                act_msg = f"| {tik}: BUY"
            elif act[num] == 2:
                act_msg = f"| {tik}: SELLL"

            right_border = (len(upper_border) - len(act_msg) - 1) * " " +"|\n"
            act_msg += right_border

            msg += act_msg

        # Separation line between action and balance
        msg += sep_line

        # Create balance message
        balance = f"| Balance: {self.balance}"
        right_border = (len(upper_border) - len(balance) - 1) * " " +"|\n"
        balance += right_border

        # Add balance to the msg
        msg += balance

        # Separation line between balance and profit
        msg += sep_line

        # Create profit message
        profit = f"| Profit: {self.balance - self.initial_balance}"
        right_border = (len(upper_border) - len(profit) - 1) * " " +"|\n"
        profit += right_border

        msg += profit

        # Lower border
        msg += upper_border
            
        print(msg)
        