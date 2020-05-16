from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_tttt.envs:FooEnv',
)
register(
    id='test-v0',
    entry_point='gym_tttt.envs:StockTradingEnv',
)
register(
    id='env-v0',
    entry_point='gym_tttt.envs:Env',
)
register(
    id='env_-v0',
    entry_point='gym_tttt.envs:Env_',
)

