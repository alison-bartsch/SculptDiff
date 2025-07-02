import os
import math
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

def fix_real_action(action7d):
    # if action7d[5] < -120:
    #     action7d[5] = 180 + 180 - np.abs(action7d[5])
    #     return action7d

    # else:
    #     return action7d

    if action7d[5] > 225:
        action7d[5] = -(360 - action7d[5])
    # check if in the unexecutable zone
    if action7d[5] < -117 and action7d[5] >= -135:
        action7d[5] = -117
    # check if need to wrap angles for unexecutable zone
    if action7d[5] < -120:
        action7d[5] = 180 + 180 - np.abs(action7d[5])
    # check if need to wrap angles for unexecutable zone
    if action7d[5] > 207:
        action7d[5] = 207

    if action7d[3] < -45:
        action7d[3] = 360 + action7d[3]
    elif action7d[3] > 45:
        action7d[3] = action7d[3] - 360

    return action7d

def rotate_action(action, center, rot):
    # given the center and rot about z in degrees, create the transform to for the points action[0:2]
    pts = np.array([[action[0], action[1], action[2]]])
    # rotate pts about center by rot degrees
    pts = pts - center
    R = Rotation.from_euler('z', np.radians(-rot), degrees=False).as_matrix()
    pts = R @ pts.T
    pts = pts.T + center
    x = pts[0, 0]
    y = pts[0, 1]

    # R_obj_in_world = Rotation.from_euler('zxy', [action[5], action[3], action[4]], degrees=True)
    # R_newframe_in_world = Rotation.from_euler('z', rot, degrees=True)
    # R_obj_in_newframe = R_newframe_in_world.inv() * R_obj_in_world
    # rz_new, rx_new, ry_new = R_obj_in_newframe.as_euler('zxy', degrees=True)

    # testing xyz convention
    R_obj_in_world = Rotation.from_euler('xyz', [action[3], action[4], action[5]], degrees=True)
    R_newframe_in_world = Rotation.from_euler('z', rot, degrees=True)
    R_obj_in_newframe = R_newframe_in_world.inv() * R_obj_in_world
    rx_new, ry_new, rz_new = R_obj_in_newframe.as_euler('xyz', degrees=True)

    action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same

    # first check rz to wrap within expected range
    if action_aug[5] > 225:
        action_aug[5] = -(360 - action_aug[5])
    # check if in the unexecutable zone
    if action_aug[5] < -117 and action_aug[5] >= -135:
        action_aug[5] = -117
    # check if need to wrap angles for unexecutable zone
    if action_aug[5] < -120:
        action_aug[5] = 180 + 180 - np.abs(action_aug[5])
    # check if need to wrap angles for unexecutable zone
    if action_aug[5] > 207:
        action_aug[5] = 207
        
    return action_aug

action_mins = np.ones(7) * 1000
action_maxs = np.ones(7) * -1000

for i in tqdm(range(20)):
    j = 1
    r_idx = 0
    # traj_path = '/home/alison/Documents/June18_Human_Demos_Train/Trajectory' + str(i)
    traj_path = '/home/alison/Clay_Data/June18_Human_Demos/pottery/Test/Trajectory' + str(i)

    while os.path.exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
        # load unnormalized action
        action7d = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
        # print("\nAction before fix: ", action7d)
        action7d = fix_real_action(action7d)

        # check if each action elem is less than action_mins
        action_mins = np.minimum(action_mins, action7d)
        action_maxs = np.maximum(action_maxs, action7d)


        # load in the center
        # ctr = np.load(traj_path + '/pcl_center' + str(j-1) + '.npy')
        ctr = np.array([0.608, 0.014, 0.125])

        for k in range(360):
            rotated_action = rotate_action(action7d, ctr, k)

            # check if each action elem is less than action_mins
            action_mins = np.minimum(action_mins, rotated_action)
            action_maxs = np.maximum(action_maxs, rotated_action)

        j+=1

print("Action mins: ", action_mins)
print("Action maxs: ", action_maxs)