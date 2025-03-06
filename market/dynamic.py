import numpy as np
import pickle


from market.data import DataMaker
from market.model import RollingModel


MLP_DATA = 3
LSTM_DATA = 2
CLOSING_DATA = 1
DATES_DATA = 0
TRAINING_WINDOW = 150
INFERENCE_NUM = 10
CHECKPOINT_NUM = 250
RETRAIN_FREQUENCY = range(0, 100001, INFERENCE_NUM) # Retrain every 10 steps
CHECKPOINT_FREQUENCY = range(0, 100001, CHECKPOINT_NUM) # Save checkpoint every 500 steps

class TradingEnv:
    def __init__(
            self, 
            stock_codes: list[str],
            start_date: str,
            end_date: str,
            points_per_set: int,
            labels_per_set: int,
            lookback: int = 3,
            initial_balance: int = 1000,
            alpha: float = 0
        ):
        self.alpha_val = alpha

        # Forge data
        self.stock_codes, self.stocks_features, self.stocks_labels = DataMaker().forge_data(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            points_per_set=points_per_set,
            labels_per_set=labels_per_set,
            lookback=lookback
        )

        # Define done condition
        self.done_cond = len(self.stocks_features[0,0,:,0]) - TRAINING_WINDOW - 1

        # Model Builder
        self.builder = RollingModel()

        # Define a dictionary to store the models
        self.models_dict = {}

        # Define a dictionary to store the direction predictions
        self.dir_predictions = {}

        # Define a dictionary to store the actual directions
        self.dir_ground_truth = {}

        # Define a dictionary to store the uncertainties
        self.dir_uncertainties = {}

        # Define a dictionary to store the actions
        self.actions = {}
        
        # Define a dict to keep the trading dates
        self.trading_dates = {}

        # Define a dict that holds the real closing prices
        self.real_closing_prices = {}

        # Define a dict that holds the predicted closing prices
        self.predicted_closing_prices = {}

        # Current step
        self.current_step = 0
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio = [initial_balance]
        self.shares_held = {}
        self.total_profit = 0

        # Store historical values
        self.balance_history = [self.balance]
        self.profit_history = [self.total_profit]

    def save_state(self, filename: str):
        state = {
            'dir_predictions': self.dir_predictions,
            'dir_ground_truth': self.dir_ground_truth,
            'dir_uncertainties': self.dir_uncertainties,
            'actions': self.actions,
            'trading_dates': self.trading_dates,
            'portfolio': self.portfolio,
            'balance_history': self.balance_history,
            'profit_history': self.profit_history,
            'predicted_closing_prices': self.predicted_closing_prices,
            'real_closing_prices': self.real_closing_prices,
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        
    
    def yolooo(self, dollars_per_trade: int = 100, transaction_fee: float = 0.00):
        if self.current_step in CHECKPOINT_FREQUENCY:
            filename = f"checkpoint_{self.current_step}_alpha_{self.alpha_val*100}.pkl"
            self.save_state(filename)
            
        if self.current_step in RETRAIN_FREQUENCY:
            self._train_models()
            self._predict()

        # Check if the episode is done
        done = self.current_step >= self.done_cond

        # Get the current prices
        current_prices = self.stocks_labels[:,CLOSING_DATA,TRAINING_WINDOW+self.current_step,0].astype(np.float32)

        current_dates = self.stocks_features[:,DATES_DATA,TRAINING_WINDOW+self.current_step,0]

        # Take action
        self._take_action(current_prices, current_dates, dollars_per_trade, transaction_fee)

        # If done sell everything
        if done:
            self._sell_everything(current_prices, current_dates, transaction_fee)
        
        # Append the balance and profit to the history
        self.balance_history.append(self.balance)
        self.profit_history.append(self.balance - self.initial_balance)
        
        # Calculate the portfolio
        portfolio = self._evaluate_holdings(current_prices)

        # Append the portfolio to the history
        self.portfolio.append(portfolio)

        # Update the current step
        self.current_step += 1
        
        print("Progress: ", self.current_step, "/", self.done_cond)
        
        return done
    
    def _sell_everything(self, current_prices, current_dates, transaction_fee):
        for num, tik in enumerate(self.stock_codes):
            tik_price = current_prices[num]
            current_date = current_dates[num]

            # Check if value is NaN
            if np.isnan(tik_price):
                action = 0 # 0 (HOLD)
                self.actions[tik].append(action)
                continue
            
            # Instantiates the dictionary
            if tik not in self.shares_held:
                self.shares_held[tik] = 0

            # Instantiates the dictionary for trading dates
            if tik not in self.trading_dates:
                self.trading_dates[tik] = []

            if self.shares_held[tik] > 0:
                # Compute the profit
                profit = tik_price * self.shares_held[tik]

                # Add the price to the balance
                self.balance += profit - profit * transaction_fee

                # Delete the shares from the portfolio
                self.shares_held[tik] = 0

                # Declare a selling action
                action = 2 # 2 (SELL)

            self.actions[tik].append(action)
            self.trading_dates[tik].append(current_date)

    
    def _evaluate_holdings(self, current_prices):
        holdings_evaluation = 0
        for num, tik in enumerate(self.stock_codes):
            # Instantiates the dictionary
            if tik not in self.shares_held:
                self.shares_held[tik] = 0

            # Get the shares
            shares = self.shares_held[tik]

            # Get the price
            price = current_prices[num]

            # Compute the evaluation
            holdings_evaluation += price * shares

        # Compute the portfolio
        portfolio = self.balance + holdings_evaluation

        return portfolio
    
    def _take_action(self, current_prices, current_dates, dollars_per_trade: int, transaction_fee: float):
        for num, tik in enumerate(self.stock_codes):
            tik_price = current_prices[num]
            current_date = current_dates[num]

            # Check if value is NaN
            if np.isnan(tik_price):
                action = 0 # 0 (HOLD)
                self.actions[tik].append(action)
                continue
            
            # Instantiates the dictionary for shares
            if tik not in self.shares_held:
                self.shares_held[tik] = 0

            # Instantiates the dictionary for actions
            if tik not in self.actions:
                self.actions[tik] = []

            # Instantiates the dictionary for trading dates
            if tik not in self.trading_dates:
                self.trading_dates[tik] = []
            
            # Get the prediction of the direction
            tik_dir = self.dir_predictions[tik][self.current_step]

            if tik_dir == 1 and self.balance >= dollars_per_trade:
                # Compute the num of shares both
                frac_shares = dollars_per_trade / tik_price

                # Add the shares to the portfolio
                self.shares_held[tik] += frac_shares

                # Substract the price from the balance
                self.balance -= dollars_per_trade - dollars_per_trade * transaction_fee

                # Declare a buying action
                action = 1 # 1 (BUY)
            elif tik_dir == -1 and self.shares_held[tik] > 0:
                profit = tik_price * self.shares_held[tik]
                # Add the price to the balance
                self.balance += profit - profit * transaction_fee

                # Delete the shares from the portfolio
                self.shares_held[tik] = 0

                # Declare a selling action
                action = 2 # 2 (SELL)
            else:
                # Declare a hold action
                action = 0 # 0 (HOLD)

            self.actions[tik].append(action)
            self.trading_dates[tik].append(current_date)
    
    def _train_models(self):
        for num, stock in enumerate(self.stock_codes):
            # Get MLP training data, sliced by the current step to the training window
            training_data_mlp = self.stocks_features[num, MLP_DATA, self.current_step:TRAINING_WINDOW+self.current_step, :].astype(np.float32)
            training_labels_mlp = self.stocks_labels[num, MLP_DATA, self.current_step:TRAINING_WINDOW+self.current_step, :].astype(np.float32)

            # Get LSTM training data, sliced by the current step to the training window
            training_data_lstm = self.stocks_features[num, LSTM_DATA, self.current_step:TRAINING_WINDOW+self.current_step, :].astype(np.float32)
            training_labels_lstm = self.stocks_labels[num, LSTM_DATA, self.current_step:TRAINING_WINDOW+self.current_step, :].astype(np.float32)

            if self.current_step == 0: # Train the initial models
                mlp_model, lstm_model = self.builder.train(
                    stockCode=stock,
                    training_data_lstm=training_data_lstm,
                    training_labels_lstm=training_labels_lstm,
                    training_data_mlp=training_data_mlp,
                    training_labels_mlp=training_labels_mlp,
                    epochs=50,
                    alpha=0.75
                )
            else: # Train the models with the rolling window
                old_mlp_model, old_lstm_model = self.models_dict[stock]

                mlp_model, lstm_model = self.builder.train(
                    stockCode=stock,
                    training_data_lstm=training_data_lstm,
                    training_labels_lstm=training_labels_lstm,
                    training_data_mlp=training_data_mlp,
                    training_labels_mlp=training_labels_mlp,
                    epochs=50,
                    alpha=self.alpha_val,
                    old_mlp_model=old_mlp_model,
                    old_lstm_model=old_lstm_model
                )

            if stock not in self.models_dict:
                self.models_dict[stock] = (mlp_model, lstm_model)
            else:
                self.models_dict[stock] = (mlp_model, lstm_model)
    
    def _predict(self):
        for num, stock in enumerate(self.stock_codes):
            # Get MLP testing data
            testing_data_mlp = self.stocks_features[
                num,
                MLP_DATA,
                TRAINING_WINDOW + self.current_step:TRAINING_WINDOW + self.current_step + INFERENCE_NUM,
                :
            ].astype(np.float32)

            # Get LSTM testing data
            testing_data_lstm = self.stocks_features[
                num,
                LSTM_DATA,
                TRAINING_WINDOW + self.current_step:TRAINING_WINDOW + self.current_step + INFERENCE_NUM,
                :
            ].astype(np.float32)

            # Get the actual closing prices as features
            actual_closing_prices = self.stocks_features[
                num,
                CLOSING_DATA,
                TRAINING_WINDOW + self.current_step:TRAINING_WINDOW + self.current_step + INFERENCE_NUM,
                1
            ].astype(np.float32)

            # Get the actual closing prices as labels
            closing_prices = self.stocks_labels[
                num,
                CLOSING_DATA,
                TRAINING_WINDOW + self.current_step:TRAINING_WINDOW + self.current_step + INFERENCE_NUM,
                0
            ].astype(np.float32)

            # Load the models
            mlp_model, lstm_model = self.models_dict[stock]

            # Predict the closing prices and uncertainty
            predictions, uncertainty = self.builder.predict(
                mlp_model=mlp_model,
                lstm_model=lstm_model,
                testing_data_mlp=testing_data_mlp,
                testing_data_lstm=testing_data_lstm,
                mc_realizations=1000
            )

            # Instantiates the dictionary if not exists
            if stock not in self.dir_uncertainties:
                self.dir_uncertainties[stock] = []
            if stock not in self.dir_predictions:
                self.dir_predictions[stock] = []
            if stock not in self.dir_ground_truth:
                self.dir_ground_truth[stock] = []
            if stock not in self.real_closing_prices:
                self.real_closing_prices[stock] = []
            if stock not in self.predicted_closing_prices:
                self.predicted_closing_prices[stock] = []

            # Check the direction of the predictions
            for pred_num, pred in enumerate(predictions):
                self.dir_uncertainties[stock].append(uncertainty[pred_num])
                if pred > actual_closing_prices[pred_num]:
                    # If the prediction is greater than the previouse actual closing price
                    self.dir_predictions[stock].append(1) # 1 (UP)
                elif pred < actual_closing_prices[pred_num]:
                    # If the prediction is less than the previouse actual closing price
                    self.dir_predictions[stock].append(-1) # -1 (DOWN)

                # Append the predicted closing prices
                self.predicted_closing_prices[stock].append(pred)

            # Check the actual direction of the stock
            for close_num, close in enumerate(closing_prices):
                if close > actual_closing_prices[close_num]:
                    self.dir_ground_truth[stock].append(1) # 1 (UP)
                elif close < actual_closing_prices[close_num]:
                    self.dir_ground_truth[stock].append(-1) # -1 (DOWN)
                
                # Append the real closing prices
                self.real_closing_prices[stock].append(close)

    def render(self):
        """Print trading information."""      
        upper_border = "┌" + "─" * 42 + "┐"

        msg = f"│ Step: {self.current_step}"
        right_border = (len(upper_border) - len(msg) - 1) * " " +"│\n"
        msg = upper_border + "\n" + msg + right_border

        sep_line = "├" + (len(upper_border) - 2) * "─" + "┤\n"

        # Upper line on Shares Held (----)
        msg += sep_line

        shares_held = "│ Shares Held:"
        right_border = (len(upper_border) - len(shares_held) - 1) * " " +"│\n"
        shares_held += right_border

        # Add shares message
        msg += shares_held

        # Add all the stocks to the msg
        for key, value in self.shares_held.items():
            shares = f"│ {key}: {value}"
            right_border = (len(upper_border) - len(shares) - 1) * " " +"│\n"
            shares += right_border

            msg += shares

        msg += sep_line

        # Signal
        signal = "│ Signal:" 
        right_border = (len(upper_border) - len(signal) - 1) * " " +"│\n"
        signal += right_border

        # Create first line signal message
        msg += signal

        for num, tik in enumerate(self.asset_ticker):
            sig_ = self.dir_predictions[:,self.current_step]
            if sig_[num] == 1:
                sig_tik = f"│ {tik}: BUY"
            elif sig_[num] == -1:
                sig_tik = f"│ {tik}: SELL"
            else:
                sig_tik = f"│ {tik}: HOLD"
                
            right_border = (len(upper_border) - len(sig_tik) - 1) * " " +"│\n"
            sig_tik += right_border

            msg += sig_tik

        # Separation line between signal and action
        msg += sep_line

        action = "│ Action:"
        right_border = (len(upper_border) - len(action) - 1) * " " +"│\n"
        action += right_border

        # Create first line action message
        msg += action

        # Populate action message
        for num, tik in enumerate(self.asset_ticker):
            act = self.actions_history[self.current_step - 1]
            if act[num] == 0:
                act_msg = f"│ {tik}: HOLD"
            elif act[num] == 1:
                act_msg = f"│ {tik}: BUY"
            elif act[num] == 2:
                act_msg = f"│ {tik}: SELLL"

            right_border = (len(upper_border) - len(act_msg) - 1) * " " +"│\n"
            act_msg += right_border

            msg += act_msg

        # Separation line between action and balance
        msg += sep_line

        # Create balance message
        balance = f"│ Balance: {self.balance}"
        right_border = (len(upper_border) - len(balance) - 1) * " " +"│\n"
        balance += right_border

        # Add balance to the msg
        msg += balance

        # Separation line between balance and profit
        msg += sep_line

        # Create profit message
        profit = f"│ Profit: {self.balance - self.initial_balance}"
        right_border = (len(upper_border) - len(profit) - 1) * " " +"│\n"
        profit += right_border

        msg += profit

        # Lower border
        lower_border = "└" + "─" * 42 + "┘"
        msg += lower_border
            
        print(msg)
        