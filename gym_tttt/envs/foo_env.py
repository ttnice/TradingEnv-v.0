import gym
from gym import spaces


class FooEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FooEnv, self).__init__()

        self.reward_range = (0, 500)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=0, high=0, shape=(2, 1))

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 2))
        self.time = 0
        self.history = []

    def step(self, action):
        # Execute one time step within the environment
        self.time += 1
        reward = action[1] + action[0]
        obs = [action[1], action[0]]
        done = True if self.time >= 5 else False
        self.history.append(reward)

        return obs, reward, done, {}

    def reset(self):
        self.time = 0

        return [[0, 0]]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(self.history)

