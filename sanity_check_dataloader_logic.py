import numpy as np
import open3d as o3d
from os.path import exists

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