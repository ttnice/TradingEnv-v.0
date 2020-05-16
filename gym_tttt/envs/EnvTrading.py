import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5

INITIAL_ACCOUNT_BALANCE = 10000

PRICE_LEN = 60
PREDICT_LEN = 60

SPREAD = 2e-4
RANKING = [30, 20, 10]
MAX_STEPS = 60


class Env(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(Env_, self).__init__()

        self.df = df
        self.SCALER = MinMaxScaler().fit([[1.08], [1.165]])

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=0, high=1, shape=(2, 3))
        # [[Hold, Buy, Sell], [Short, Long, None]]

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 60))
        # [[Hold, Buy, Sell, None, ...], price, predict ]
        self.reset()

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        reward = self.reward
        done = self.current_step >= len(self.df)
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # All the data to render
        self.invalid = 0
        self.number = np.zeros((4, 2, 2))
        '''
        {
            'timeout': {
                'profit': [0, 0],
                'loss'  : [0, 0],
            },
            'born': {
                'profit': [0, 0],
                'loss'  : [0, 0],
            },
            'changeToBuy': {
                'profit': [0, 0],
                'loss'  : [0, 0],
            },
            'changeToSell': {
                'profit': [0, 0],
                'loss'  : [0, 0],
            },    
        }'''

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.df) -PRICE_LEN -PREDICT_LEN)

        # Position of the state
        current_price = self._current_price()
        self._close_trade(current_price)
        self.position = [.5 for i in range(PRICE_LEN)]

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        print(f'Step: {self.current_step}')
        print(f'Number of Invalid demand : {self.invalid}')
        print(f'Number of trade : {sum(self.number[:, :, 1])}')
        print(f'Result : {sum(self.number[:, 0, 1])-sum(self.number[:, 1, 1])} = {sum(self.number[:, 0, 1])} - {sum(self.number[:, 1, 1])})\n\n')
        print(f'{self.number}')

    def plot(self):
        pass

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        price = self.df.loc[self.current_step: self.current_step + PRICE_LEN, 'Close'].values
        predict = self.df.loc[self.current_step + PRICE_LEN: self.current_step + PRICE_LEN + PREDICT_LEN, 'Close'].values
        position = np.array(self.position)

        # Append additional data and scale each value to between 0-1
        obs = np.array([price, predict, position])

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self._current_price()
        buy_profit = action[1][0]
        sell_profit = action[1][1]
        temp_reward = self.SCALER.inverse_transform([[current_price - self.open_trade_price]])[0,0] * self.trend
        temp_position = .5
        if buy_profit < current_price or sell_profit > current_price:
            # buy/sell_profit Invalid
            self.invalid += 1
            self.reward = -1
        elif self.current_step - self.open_trade_step <= MAX_STEPS:
            # Reach MaxSteps, the longest possible trade or reaching len(df)
            if temp_reward > 0: # win by TimeOut
                self._number_manager(temp_reward, 0, 0)
                rank = RANKING[2]
            else: # loss by TimeOut
                self._number_manager(temp_reward, 0, 1)
                rank = RANKING[0]
            self.reward = (temp_reward - SPREAD) * rank
            self._close_trade(current_price)

        elif (self.take_profit - current_price)*self.trend < 0:
            # Reach TakeProfit, big positive reward
            self._number_manager(temp_reward, 1, 0)
            self.reward = (temp_reward - SPREAD) * RANKING[0]
            self._close_trade(current_price)
        elif (self.stop_loss - current_price)*self.trend > 0:
            # Reach StopLoss, small negative reward
            self._number_manager(temp_reward, 1, 1)
            self.reward = (temp_reward - SPREAD) * RANKING[2]
            self._close_trade(current_price)
        else:
            action_type = np.argmax(action[0])
            if action_type == 0:
                # Holding
                self.reward = temp_reward
            elif action_type == 1:
                # Buying
                temp_position = 0
                if self.trend == 1:
                    # Confirming buying trend
                    self.reward = temp_reward
                else:
                    if temp_reward > 1:
                        self._number_manager(temp_reward, 3, 0)
                    else:
                        self._number_manager(temp_reward, 3, 1)
                    self.reward = (temp_reward - SPREAD) * RANKING[1]
                    # Open Buying Trade
                    self._open_trade(current_price, 1, buy_profit)
            elif action_type == 2:
                # Selling
                temp_position = 1
                if self.trend == -1:
                    # Confirming buying trend
                    self.reward = temp_reward
                else:
                    if temp_reward > 1:
                        self._number_manager(temp_reward, 4, 0)
                    else:
                        self._number_manager(temp_reward, 4, 1)
                    self.reward = (temp_reward - SPREAD) * RANKING[1]
                    # Open Selling Trade
                    self._open_trade(current_price, -1, sell_profit)
        self.position = self.position[1:]+[temp_position]

    def _current_price(self):
        current_price = random.uniform(self.df.loc[self.current_step, "Low"], self.df.loc[self.current_step, "High"])
        return current_price

    def _open_trade(self, current_price, trend, take_profit):
        # Putting position to Holding
        self.position = np.zeros((3,))
        self.position[trend%3] = 1  # description of past decision [Hold, Buy, Sell]
        self.trend = trend  # 1 for buying trend and -1 for selling trend
        self.open_trade_step = self.current_step  # date of open trade
        self.open_trade_price = current_price  # price of open trade
        self.take_profit = take_profit # Take_profit value to stop the winning trade
        self.stop_loss = (3*current_price-take_profit)/2 # Stop_loss value to stop the loosing trade

    def _close_trade(self, current_price):
        # Putting position to Holding
        self.position = np.array([1, 0, 0])  # description of past decision [Hold, Buy, Sell]
        self.trend = 0  # 1 for buying trend and -1 for selling trend
        self.open_trade_step = self.current_step # date of open trade
        self.open_trade_price = current_price # price of open trade

    def _number_manager(self, reward, x, y):
        self.number[x, y, 0] += 1
        self.number[x, y, 1] += reward


if __name__ == '__main__':
    import pandas as pd
    import pandas_datareader as web

    df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
    df = df.reset_index()
    venv = Env_(df)
    obs = venv.reset()
    print(obs)
    print(type(obs))