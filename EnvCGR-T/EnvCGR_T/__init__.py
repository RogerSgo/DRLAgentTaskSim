# Registro del entorno

from gymnasium.envs.registration import register


register(
    id="EnvCGR_T/ConTask-v0",
    entry_point="EnvCGR_T.envs:ConTask",
)