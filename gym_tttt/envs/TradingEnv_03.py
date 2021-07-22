import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5

INITIAL_ACCOUNT_BALANCE = 10000

PRICE_LEN = 60
PREDICT_LEN = 60

SPREAD = 5e-4
RANKING = [30, 20, 10]
MAX_STEPS = 7_200

import pandas_datareader as web
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
df = df.reset_index()

class TradingEnv_03(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df=df):
        super(TradingEnv_03, self).__init__()

        self.df = df
        self.SCALER = MinMaxScaler()
        self.SCALER.fit([[1.08], [1.165]])

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Discrete(3)
        # [[Hold, Buy, Sell], [Short, Long, None]]

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 60))
        # [[Hold, Buy, Sell, None, ...], price, predict ]
        self.reset()

    def step(self, action):
        # Execute one time step within the environment
        self.reward = 0
        self._take_action(action)

        self.current_step += 1

        reward = self.reward
        done = (self.current_step >= len(self.df)) or (self.current_step >= self.start_step + MAX_STEPS)
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Information
        self.cumulate_reward = 0
        self.n_transaction = 0

        # Set the current step to a random point within the data frame
        self.current_step = np.random.randint(PRICE_LEN, len(self.df))
        self.start_step = self.current_step

        # Position of the state
        self._current_price()
        self._close_trade()
        self.position = [.5 for i in range(PRICE_LEN)]

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Price: {self._inverse_scaler(self.current_price)}')
        print(f'Reward : {self.reward}')
        print(f'Cumulated Reward : {self.cumulate_reward}')
        print(f'Cumulated Trades : {self.n_transaction}')

    def plot(self):
        plt.clf()
        plt.plot(self.prices)
        plt.show()

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        self.prices = self.df.loc[self.current_step - PRICE_LEN: self.current_step-1, 'Close'].values
        self.position = self.position[1:]
        self.position.append(self.trend/2)
        position = np.array(self.position)

        # Append additional data and scale each value to between 0-1
        obs = np.array([self.prices, position])

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        self._current_price()
        t_reward = (self._inverse_scaler(self.current_price) - self._inverse_scaler(self.open_trade_price)) * (self.trend-1)
            # if buying:      win --> positive
                            #loss --> negative
            # if selling:     win --> positive
                            #loss --> negative
        if action == 1: #Holding
            if self.trend == 1:
                pass
            else:
                self._close_trade(t_reward)
        elif action == 2: #Buying
            if self.trend == 2:
                self.reward = t_reward
            else:
                self._open_trade(trend=2, t_reward=t_reward)
        elif action == 0: #Selling
            if self.trend == 0:
                self.reward = t_reward
            else:
                self._open_trade(trend=0, t_reward=t_reward)
        else:
            raise ValueError(f"Action must be integer between [0;2] instead of {action}")


    def _current_price(self):
        self.current_price = np.random.uniform(self.df.loc[self.current_step, "Low"], self.df.loc[self.current_step, "High"])

    def _open_trade(self, trend, t_reward):
        if self.trend != 1: #Check if was holding
            self._close_trade(t_reward)
        self.trend = trend  # 0 for selling 1 for holding trend and 2 for buying trend
        self.open_trade_step = self.current_step  # date of open trade
        self.open_trade_price = self.current_price  # price of open trade

    def _close_trade(self, t_reward=0):
        # self.open_trade_step = self.current_step # date of open trade
        self.open_trade_price = self.current_price # price of open trade
        self.reward = (t_reward - SPREAD)*100
        # Putting position to Holding
        self.trend = 1  # 0 for selling 1 for holding trend and 2 for buying trend
        self.cumulate_reward += self.reward
        self.n_transaction += 1

    def _inverse_scaler(self, value):
        value = self.SCALER.inverse_transform([[value]])[0,0]
        return value

    def _get_online_data(self):
        pass




if __name__ == '__main__':
    # from Trading.Env.EnvTrading_ import Env_
    import pandas as pd
    import pandas_datareader as web

    df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
    df = pd.read_csv
    df = pd.read_csv('https://raw.githubusercontent.com/ttnice/AlgoTrade3D/master/data/scaled/EURUSD-2019.csv')
    venv = Env_(df)
    obs = venv.reset()
    venv.current_step = 65
    venv.step(1)
    done = False
    while not done:
        venv.plot()
        action = int(input("Action : "))
        for i in range(50):
            obs, reward, done, info = venv.step(action)
            print(f'Reward : {reward}')
            print(venv._inverse_scaler(venv.current_price))