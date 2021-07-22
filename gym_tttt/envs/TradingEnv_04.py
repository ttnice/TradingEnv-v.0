import random
import gym
from gym import spaces
import numpy as np

O = "open"
H = "high"
L = "low"
C = "close"
BORNE_INF = 120
MAX_STEP = 500
FEES = 0.001 # Soit 0.1%
INDICATORS = ['sma_20_50', 'sma_20_100', 'sma_50_100', 'ema_20_50', 'ema_20_100',
       'ema_50_100', 'vwap_20_50', 'vwap_20_100', 'vwap_50_100',
       'awesome_oscillator', 'ppo', 'ppo_hist', 'pvo', 'pvo_hist', 'rsi_close',
       'rsi_high', 'rsi_low', 'srsi', 'srsi_d', 'srsi_k', 'stoch',
       'stoch_signal', 'tsi', 'ultimate_o', 'will', 'cmf', 'em', 'sem', 'mfi',
       'bbh', 'bbl', 'bbm', 'dch', 'dcl', 'dcm', 'kch', 'kcl', 'kcm', 'adx',
       'adx_neg', 'adx_pos', 'aroon_d', 'aroon_u', 'aroon_i', 'cci', 'dpo',
       'kst', 'kst_diff', 'kst_sig', 'macd', 'macd_d', 'macd_s', 'psar_d',
       'psar_u', 'stc', 'trix', 'vortex']


class TradingEnv_04(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, indicators=INDICATORS, ohlc=None):
        super(TradingEnv_04, self).__init__()
        # Actions of the format
        # 0 : Sell,
        # 1 : Selling trend,
        # 2 : Holding trend,
        # 3 : Buying trend,
        # 4 : Buy
        self.action_space = spaces.Discrete(5)
        # indicators values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(len(indicators), ),
            dtype=np.float16
        )
        self.reward_range = (-1, 1)

        self.df = df
        self.len = len(df)
        self.indicators = indicators
        if ohlc is not None:
            O, H, L, C = ohlc # for example = ["Open", "High", "Low", "Close"]

        self.initial_step = BORNE_INF
        self.current_step = BORNE_INF
        self.tt_step = 0
        self.last_action = 2
        self.is_opened = False
        self.opened_price = 0
        self.tt_gain = 0
        self.nb_trades = 0

    def step(self, action):
        self.current_step += 1
        self.tt_step += 1

        if self.current_step > self.len:
            self.current_step = BORNE_INF

        reward = self._take_action(action)
        done = self.tt_step > MAX_STEP
        obs = self._next_observation()
        return obs, reward, done, {"action": action}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.initial_step = BORNE_INF
        self.current_step = np.random.randint(120, self.len)
        self.tt_step = 0
        self.last_action = 2
        self.is_opened = False
        self.opened_price = 0
        self.tt_gain = 0
        self.nb_trades = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Executed: {self.tt_step}')
        print(f'tt_gain: {self.tt_gain}')
        print(f'Last gain: {self.last_action} ')

    def _next_observation(self):
        df_ = self.df.loc[self.current_step, INDICATORS].values
        return df_

    def _take_action(self, action):
        reward = 0
        price = self.df.loc[self.current_step, C]
        if action in [0, 1]: # Close Buy
            if self.last_action in [3, 4] and self.is_opened: # Buying trend -> Selling Trend
                reward = self._close_trade((self.opened_price - price) / price)
            if action == 0 and not self.is_opened: # Sell
                self._open_trade(price)
        elif action in [3, 4]: # Close Buy
            if self.last_action in [0, 1] and self.is_opened: # Selling trend -> Buying Trend
                reward = self._close_trade((price - self.opened_price) / price)
            if action == 4 and not self.is_opened: # Buy
                self._open_trade(price)
        else: # Holding Trend
            if self.last_action in [3, 4] and self.is_opened: # Buying trend -> Selling Trend
                reward = self._close_trade((self.opened_price - price) / price)
            if self.last_action in [0, 1] and self.is_opened: # Selling trend -> Buying Trend
                reward = self._close_trade((price - self.opened_price) / price)
        self.last_action = action
        return reward

    def _open_trade(self, price):
        self.is_opened = True
        self.opened_price = price

    def _close_trade(self, percent_win):
        self.is_opened = False
        return percent_win - FEES

