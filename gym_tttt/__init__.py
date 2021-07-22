from gym.envs.registration import register

register(
    id='TradingEnv_00-v0',
    entry_point='gym_tttt.envs:TradingEnv_00',
)
register(
    id='TradingEnv_01-v0',
    entry_point='gym_tttt.envs:TradingEnv_01',
)
register(
    id='TradingEnv_02-v0',
    entry_point='gym_tttt.envs:TradingEnv_02',
)
register(
    id='TradingEnv_03-v0',
    entry_point='gym_tttt.envs:TradingEnv_03',
)

