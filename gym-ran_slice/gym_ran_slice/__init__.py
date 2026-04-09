from gymnasium.envs.registration import register

from .ran_slice import RanSlice
from .ran_slice_forecast import RanSliceForecast

register(
    id='RanSlice-v1',
    entry_point='gym_ran_slice:RanSlice'
)

register(
    id='RanSliceForecast-v1',
    entry_point='gym_ran_slice:RanSliceForecast'
)