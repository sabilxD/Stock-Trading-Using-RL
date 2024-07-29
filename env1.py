import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import datetime as dt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnv(gym.Env):
    """A stock trading environment for Gymnasium"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df , render_mode=None):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.render_mode = render_mode
        self.reward_range = (0, np.inf)
        
        #we have three types of actions buy sell hold ,in that we have buy x% sell x% and hold 
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHLC values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / self.df.loc[:self.current_step, 'Open'].mean(),
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / self.df.loc[:self.current_step, 'High'].mean(),
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / self.df.loc[:self.current_step, 'Low'].mean(),
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / self.df.loc[:self.current_step, 'Close'].mean(),
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / self.df.loc[:self.current_step, 'Volume'].mean()
        ])
        obs = np.append(frame, [[
            self.balance / INITIAL_ACCOUNT_BALANCE,
            self.max_net_worth / INITIAL_ACCOUNT_BALANCE,
            self.shares_held /
            self.df.loc[:self.current_step, 'Volume'].mean(),
            self.cost_basis / self.df.loc[:self.current_step, 'Close'].mean(),
            self.total_shares_sold /
            self.df.loc[:self.current_step, 'Volume'].mean(),
            self.total_sales_value / (self.df.loc[:self.current_step, 'Volume'].mean(
            ) * self.df.loc[:self.current_step, 'Close'].mean()),
        ]], axis=0)

        return obs



    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.df.index[self.current_step], "Open"], self.df.loc[self.df.index[self.current_step], "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = 0 if self.shares_held + shares_bought == 0 else (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0
        truncated = False

        obs = self._next_observation()

        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        obs = self._next_observation()
        return obs, {}  # Return the observation and an empty info dictionary

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')


if (__name__ == '__main__'):

    # Load your historical stock data into a pandas dataframe
    df = pd.read_csv('A.csv', index_col='Date', parse_dates=True)
    df.dropna(inplace=True)
    df = df.sort_values('Date')
    df = df.reset_index(drop=True)

    # Create the stock trading environment
    env = DummyVecEnv([lambda: StockTradingEnv(df, render_mode='human')])

    # Instantiate the agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Reset the environment for a new episode
    obs = env.reset()


    # Loop through steps within the episode
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, _ = env.step(action)
        env.render()
    env.close()
