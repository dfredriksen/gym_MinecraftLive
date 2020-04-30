from gym.envs.registration import register

register(
    id='MinecraftLive-v0',
    entry_point='gym_MinecraftLive.envs:MinecraftLiveEnv',
)