import numpy as np

a_mins7d = np.array([0.52776, -0.0662, 0.1272, -360, -10.10, -120, 0.008])
a_maxs7d = np.array([0.68825, 0.09425, 0.1600, 360, 11.68, 240, 0.016])


np.save('/home/alison/Documents/GitHub/SculptDiff/checkpoints/subgoal_new_data_16_pred_june30_updated_augs/action_mins.npy', a_mins7d)
np.save('/home/alison/Documents/GitHub/SculptDiff/checkpoints/subgoal_new_data_16_pred_june30_updated_augs/action_maxs.npy', a_maxs7d)