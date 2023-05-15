from typing import Callable

def recipe(array_lambda: Callable, device_lambda: Callable, **params):
    """
    Place each array from calling `array_lambda` on device from calling `device_lambda`.
    """
    result = []
    if not params:
        for device in device_lambda():
            with device:
                result.append(array_lambda())
        return result
    for step_args in zip(*params.values()):
        step_kwargs = {argname: value for argname, value in zip(params.keys(), step_args)}
        with device_lambda(**step_kwargs):
            result.append(array_lambda(**step_kwargs))
    return result