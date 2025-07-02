import numpy as np
import open3d as o3d
from os.path import exists

# # ----------------- Forward Backward dataloader ------------------
# traj_path = '/home/alison/Documents/June18_Human_Demos_Train' + '/Trajectory' + str(2)
# pred_horizon = 8



# # states = []
# actions = []
# # centers = []
# j = 0

# while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  

#     if j != 0:
#         # load unnormalized action
#         a = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
#         actions.append(a)
#     j+=1

# # TODO: then will need to pad the previous action to also be the pred_horizon length
#     # NOTE: instead of zero padding, we will pad with obs pos (normalized)

# episode_len = len(actions)
# start_ts = np.random.choice(episode_len)
# # state = states[start_ts]


# action = actions[start_ts:]
# action = np.stack(action, axis=0)

# # add in termination token -1 continue, 1 stop
# stop_token = -1 * np.ones((action.shape[0], 1))
# stop_token[-1] = 1
# action = np.concatenate((action, stop_token), axis=1)

# action_len = episode_len - start_ts

# if start_ts != 0:
#     obs_pos = actions[start_ts-1]
# else:
#     obs_pos = np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04])

# # add padding to obs_pos of one 0 vector to make 8d
# obs_pos = np.concatenate((obs_pos, -1 * np.ones((1))), axis=0)

# if action_len < pred_horizon:
#     padded_action = np.zeros((pred_horizon, 8))
#     padded_action[:action_len] = action
#     for i in range(action_len, pred_horizon):
#         padded_action[i] = action[-1]
# else:
#     padded_action = action[:pred_horizon]

# # get previous actions
# prev_actions = actions[0:start_ts]
# # reverse the previous actions to get the backward trajectory
# prev_actions = prev_actions[::-1] # this way we get most recent previous action first
# prev_actions = np.stack(prev_actions, axis=0)
# prev_stop_tokens = -1 * np.ones((prev_actions.shape[0], 1))
# prev_actions = np.concatenate((prev_actions, prev_stop_tokens), axis=1)
# prev_action_len = start_ts

# if prev_action_len < pred_horizon:
#     padded_prev_action = np.zeros((pred_horizon, 8))
#     padded_prev_action[:prev_action_len] = prev_actions
#     for i in range(prev_action_len, pred_horizon):
#         padded_prev_action[i] = np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04])
# else:
#     padded_prev_action = prev_actions[:pred_horizon]

# # reverse the padded previous actions again 
# padded_prev_action = padded_prev_action[::-1]

# # combine the prev actions and actions to get continuous action sequence
# full_padded_action = np.concatenate((padded_prev_action, padded_action), axis=0)


# max_idx = j-2
# min_idx = 0

# gt_actions = []
# # handling the case where we need to pad the previous action
# if start_ts - pred_horizon < 0:
#     print("Need to pad previous actions")
#     for i in range(pred_horizon - start_ts):
#         print("appending obs pos")
#         gt_actions.append(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]))   
#     for j in range(start_ts):
#         print("appending true prev")
#         gt_actions.append(np.load(traj_path + '/action7d_unnormalized' + str(j) + '.npy'))
# else:
#     print("No need to pad previous actions")
#     print("start ts - pred_horizon: ", start_ts - pred_horizon)
#     for k in range(start_ts - pred_horizon, start_ts):
#         print("appending all prev")
#         gt_actions.append(np.load(traj_path + '/action7d_unnormalized' + str(k) + '.npy'))

# # handling the case where we need to pad the future actions
# if start_ts + pred_horizon > max_idx:
#     print("Need to pad future actions")
#     for l in range(max_idx - start_ts):
#         print("appending true future")
#         gt_actions.append(np.load(traj_path + '/action7d_unnormalized' + str(start_ts + l) + '.npy'))
#     for m in range(pred_horizon - (max_idx - start_ts)):
#         print("appending last future")
#         gt_actions.append(np.load(traj_path + '/action7d_unnormalized' + str(max_idx) + '.npy'))
# else:
#     print("No need to pad future actions")
#     for n in range(pred_horizon):
#         print("appending all future")
#         gt_actions.append(np.load(traj_path + '/action7d_unnormalized' + str(start_ts + n) + '.npy'))

# # get the difference between the gt actions and the padded actions
# gt_actions = np.stack(gt_actions, axis=0)
# # get the first 7 dimensions of full_padded_action
# full_padded_action = full_padded_action[:, :7]
# print("Full padded action: ", full_padded_action.shape)
# print("GT actions: ", gt_actions.shape)
# diff_actions = gt_actions - full_padded_action[:, :7]
# print("Diff Actions: ", diff_actions)


# ------------------ Sub-goal dataloader -------------------

traj_path = '/home/alison/Documents/June18_Human_Demos_Train' + '/Trajectory' + str(2)
subgoal_stepsize = 4
pred_horizon = 16

# print("Traj path: ", traj_path)

states = []
actions = []


j = 0
while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
    s = np.load(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy')
    states.append(s)

    if j != 0:
        actions.append(np.ones(7))
    j += 1

# episode_len = len(actions)
full_episode_len = len(actions)
start_ts = np.random.choice(full_episode_len - subgoal_stepsize - 1)
full_action_len = full_episode_len - start_ts

# action = actions[start_ts:]
if full_action_len >= pred_horizon:
    action = actions[start_ts:start_ts + pred_horizon]
    state_list = states[start_ts:(start_ts + pred_horizon + subgoal_stepsize):subgoal_stepsize]
else:
    action = actions[start_ts:]
    state_list = states[start_ts::subgoal_stepsize]

action_len = len(action)
action = np.stack(action, axis=0)

# add in termination token -1 continue, 1 stop
stop_token = -1 * np.ones((action.shape[0], 1))
stop_token[-1] = 1
action = np.concatenate((action, stop_token), axis=1)

if start_ts != 0:
    obs_pos = actions[start_ts-1]
else:
    obs_pos = np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04])

# add padding to obs_pos of one 0 vector to make 8d
obs_pos = np.concatenate((obs_pos, -1 * np.ones((1))), axis=0)

states_seq_size = int((pred_horizon + subgoal_stepsize) / subgoal_stepsize)

if action_len < pred_horizon:
    print("here")
    padded_action = np.zeros((pred_horizon, 8))
    padded_action[:action_len] = action
    for i in range(action_len, pred_horizon):
        padded_action[i] = action[-1]

    padded_states = np.zeros((states_seq_size, 2048, 3))
    padded_states[:len(state_list)] = np.stack(state_list, axis=0)
    # duplicate last element of state_list to fill the rest
    padded_states[len(state_list):] = np.tile(state_list[-1], (len(padded_states[len(state_list):]), 1, 1))
else:
    print("need to pad")
    padded_action = action[:pred_horizon]
    padded_states = np.stack(state_list, axis=0)


# loop through the padded states and create point clouds in open3d to visualize
for i, state in enumerate(padded_states):
    pcl_o3d = o3d.geometry.PointCloud()
    pcl_o3d.points = o3d.utility.Vector3dVector(state)
    pcl_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(state), 1)))
    
    # visualize the point cloud
    o3d.visualization.draw_geometries([pcl_o3d])