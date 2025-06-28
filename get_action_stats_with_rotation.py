import os
import numpy as np
from scipy.spatial.transform import Rotation
from sanity_check_dataset import rotate_action

action_mins = np.ones(7) * 1000
action_maxs = np.ones(7) * -1000

for i in range(20):
    j = 1
    r_idx = 0
    traj_path = '/home/alison/Documents/June18_Human_Demos_Train/Trajectory' + str(i)

    while os.path.exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
        # load unnormalized action
        action7d = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')


        if action7d[5] < -90:
            if action7d[3] < 0:
                action7d[4] = -action7d[4]
            else:
                action7d[3] = -action7d[3]
                action7d[4] = -action7d[4]

        if action7d[5] > 90:
            action7d[4] = -action7d[4]

        if np.abs(action7d[5]) < 90:
            action7d[4] = -action7d[4]

        # check if each action elem is less than action_mins
        action_mins = np.minimum(action_mins, action7d)
        action_maxs = np.maximum(action_maxs, action7d)


        # load in the center
        ctr = np.load(traj_path + '/pcl_center' + str(j-1) + '.npy')

        for k in range(360):
            rotated_action = rotate_action(action7d, ctr, k)
            
            if rotated_action[5] < -90:
                if rotated_action[3] < 0:
                    rotated_action[4] = -rotated_action[4]
                else:
                    rotated_action[3] = -rotated_action[3]
                    rotated_action[4] = -rotated_action[4]

            if rotated_action[5] > 90:
                rotated_action[4] = -rotated_action[4]

            if np.abs(rotated_action[5]) < 90:
                rotated_action[4] = -rotated_action[4]

        j+=1

print("Action mins: ", action_mins)
print("Action maxs: ", action_maxs)