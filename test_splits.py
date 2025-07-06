import numpy as np


def split_with_horizons(arr, pred_horizon=16, execute_horizon=8, sub_goal_horizon=4):
    result = []
    step = execute_horizon // sub_goal_horizon
    window_size = pred_horizon // sub_goal_horizon + 1

    for i in range(0, len(arr), step):
        window = arr[i:i + window_size]
        if len(window) < window_size:
            new_window = arr[-1] * np.ones(window_size)
            new_window[0:len(window)] = window
            window = new_window
        result.append(window)
        if i + step >= len(arr):
            break
    return result

arr1 = np.array([0,4,6,12,16,20,24])
print(split_with_horizons(arr1))
arr2 = np.array([0,8,16,24])
print(split_with_horizons(arr2, sub_goal_horizon=8))