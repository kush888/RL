from binance import Client
import pandas as pd
import datetime
import random
from gym.spaces import Discrete, Box
import numpy as np
from collections import deque
from utils import write_to_file
from viz import TradingGraph
from stable_baselines3.common.env_checker import check_env
from gym import Env
import os
from stable_baselines3 import PPO
from ta.trend import SMAIndicator
from ta.trend import SMAIndicator, macd, PSARIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi


api_key = "IkglEvMVJST0OmJA3Jfhi7nGUirfrYRnGsGdBTUoKNkpOPiDmSnfElk3zujUrabT"
secret_key = "hXOnb96VFSBSfvrHJYAdBv9UGR61CnbpqXZpDhoqGqc0QxbLNI9BdsCZsRrtyou2"
client = Client(api_key, secret_key)
log_path = os.path.join('Training', 'Logs')
save_path = os.path.join('Training', "Saved Models", "PPO Model Crypto 1")


class CustomEnv(Env):
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=120, trading_cycle=480,  render_range=100, normalize_value=40000):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.num_of_parameters = 6
        self.env_steps_size = trading_cycle
        self.render_range=render_range
        self.no_of_orders = 0
        self.order_placed_now = False
        self.punish_value = 0
        self.total_reward = 0
        self.normalize_value = normalize_value

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=100000, shape=(lookback_window_size, self.num_of_parameters))
        self.state = deque(maxlen=self.lookback_window_size)
        self.state_size = (self.lookback_window_size, self.num_of_parameters)

    def reset(self, mode="training"):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.no_of_orders = 0
        self.punish_value = 0

        print (self.total_reward)
        self.total_reward = 0
        if mode == "training":
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - self.env_steps_size)
            self.end_step = self.start_step + self.env_steps_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
        self.current_step = self.start_step
        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.state.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume'],
                                        self.balance
                                        ])

        self.visualization = TradingGraph(Render_range=self.render_range) # init visualization
        self.trades = deque(maxlen=self.render_range)

        return np.array(self.state)


    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1
        time = self.df.loc[self.current_step, 'Time'] # for visualization
        high = self.df.loc[self.current_step, 'High'] # for visualization
        low = self.df.loc[self.current_step, 'Low'] # for visualization

        current_price = self.df.loc[self.current_step, 'Close']
        self.order_placed_now = False
        if action == 0:  # Hold
            pass
        elif action == 1 and self.balance > self.initial_balance/100:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.no_of_orders += 1
            self.order_placed_now = True
            self.trades.append({'time' : time, 'High' : high, 'Low' : low, 'total': self.crypto_bought, 'type': "buy", "current_price": current_price})

        elif action == 2 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.no_of_orders += 1
            self.order_placed_now = True
            self.trades.append({'time' : time, 'High' : high, 'Low' : low, 'total': self.crypto_sold, 'type': "sell", "current_price": current_price})

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        # Calculate reward
        reward = self.get_reward()
        self.total_reward += reward
        if self.net_worth <= self.initial_balance / 2 or self.current_step >= self.end_step:
            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done, {}

    # Get the data points for the given current_step
    def _next_observation(self):
        self.state.append([self.df.loc[self.current_step, 'Open'],
                           self.df.loc[self.current_step, 'High'],
                           self.df.loc[self.current_step, 'Low'],
                           self.df.loc[self.current_step, 'Close'],
                           self.df.loc[self.current_step, 'Volume'],
                           self.balance
                           ])
        return np.array(self.state)


# Calculate reward
    def get_reward(self):
        self.punish_value += self.net_worth * 0.00001
        if self.no_of_orders > 1 and self.order_placed_now:
            if self.trades[-1]['type'] == "buy":
                reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - self.trades[-2]['total']*self.trades[-1]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
            elif self.trades[-1]['type'] == "sell":
                reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - self.trades[-2]['total']*self.trades[-2]['current_price']
                reward -= self.punish_value
                self.trades[-1]["Reward"] = reward
                self.punish_value = 0
                return reward
        else:
            return 0 - self.punish_value


    def render(self, visualize=False):
        if visualize:
            time = self.df.loc[self.current_step, 'Time']
            open = self.df.loc[self.current_step, 'Open']
            close = self.df.loc[self.current_step, 'Close']
            high = self.df.loc[self.current_step, 'High']
            low = self.df.loc[self.current_step, 'Low']
            volume = self.df.loc[self.current_step, 'Volume']

            # Render the environment to the screen
            self.visualization.render(time, open, high, low, close, volume, self.net_worth, self.trades)

def addIndicators(df):
    # Add Simple Moving Average (SMA) indicators
    df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()

    # Add Bollinger Bands indicator
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2, fillna=True)
    df['psar'] = indicator_psar.psar()

    # Add Moving Average Convergence Divergence (MACD) indicator
    df["MACD"] = macd(close=df["Close"], window_slow=26, window_fast=12, fillna=True) # mazas

    # Add Relative Strength Index (RSI) indicator
    df["RSI"] = rsi(close=df["Close"], window=14, fillna=True) # mazas

    return df


def test_random(env, train_episodes=50, visualize=False):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset()

        while True:
            env.render(visualize)

            action = np.random.randint(3, size=1)[0]
            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)

def test_model(env, model, visualize=True, test_episodes=2):

    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            if env.current_step == env.end_step or done:
                average_net_worth += env.net_worth
                print("net_worth:", episode, env.net_worth)
                break

    print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth/test_episodes))

df = pd.read_pickle('1 year data')
df = df.iloc[:, :6]
df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
df = df.sort_values('Time')
df = df.set_index('Time')
df = df.astype(float)
# df = addIndicators(df)


lookback_window_size = 120
train_df = df[:-14400]
test_df = df[-14400:] # 10 days

trading_cycle = 240
rand = random.randint(0, 14400-trading_cycle*2)
test_df = df[rand:rand+2*trading_cycle]


train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size, trading_cycle=trading_cycle)

# model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_path)
model = PPO.load(save_path, env=train_env)

model.learn(total_timesteps=1000000)
model.save(save_path)
test_model(test_env, model)

# Random_games(train_env, train_episodes=10, visualize=True)


#510217
#5,09,976