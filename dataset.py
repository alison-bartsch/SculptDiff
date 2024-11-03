import math
import torch
import json
import numpy as np
import open3d as o3d 
from os.path import exists
from PIL import Image
from scipy.spatial.transform import Rotation

class ClayDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pred_horizon, n_datapoints, n_raw_trajectories, center_action):
        """
        The Dataloader for the clay sculpting dataset at the Trajectory level (compatible with ACT and Diffusion Policy). 

        :param episode_idxs: list of indices of the episodes to load
        :param dataset_dir: directory where the dataset is stored
        :param n_datapoints: number of datapoints (i.e. desired number of final trajectories after augmentation)
        :param n_raw_trajectories: number of raw trajectories in the dataset
        :param center_action: whether to center the action before normalizing
        """
        super(ClayDataset).__init__()
        self.dataset_dir = dataset_dir
        self.pred_horizon = pred_horizon
        self.n_datapoints = n_datapoints
        self.n_raw_trajectories = n_raw_trajectories
        self.center_action = center_action

        # determine the number of datapoints per trajectory - needs to be a round number
        self.n_datapoints_per_trajectory = self.n_datapoints / self.n_raw_trajectories
        if not self.n_datapoints_per_trajectory.is_integer():
            raise ValueError('The number of datapoints per trajectory needs to be a round number, please input a valid number of datapoints given the number of raw trajectories')

        # deterime the augmentation interval
        self.aug_step = 360 / self.n_datapoints_per_trajectory

    def _center_pcl(self, pcl, center):
        centered_pcl = pcl - center
        centered_pcl = centered_pcl * 10
        return centered_pcl

    def _center_normalize_action(self, action, ctr):
        # center the action
        new_action = np.zeros(5)
        new_action[0:3] = action[0:3] - ctr
        new_action[3:5] = action[3:5]
        # normalize centered action
        mins = np.array([-0.15, -0.15, -0.05, -90, 0.005])
        maxs = np.array([0.15, 0.15, 0.05, 90, 0.05])
        norm_action = np.zeros(5)
        norm_action[0:5] = (new_action[0:5] - mins) / (maxs - mins)
        norm_action = norm_action * 2 - 1 # set to [-1, 1]
        return norm_action

    def _normalize_action(self, action):
        a_mins5d = np.array([0.56, -0.062, 0.125, -90, 0.005])
        a_maxs5d = np.array([0.7, 0.062, 0.165, 90, 0.05])
        norm_action = (action - a_mins5d) / (a_maxs5d - a_mins5d)
        norm_action = norm_action  * 2 - 1 # set to [-1, 1]
        return norm_action
    
    def _rotate_pcl(self, state, center, rot):
        '''
        Faster implementation of rotation augmentation to fix slow down issue
        '''
        state = state - center
        R = Rotation.from_euler('xyz', np.array([0, 0, rot]), degrees=True).as_matrix()
        state = R @ state.T
        pcl_aug = state.T + center
        return pcl_aug

    def _rotate_action(self, action, center, rot):
        unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
        rot_original = math.degrees(math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0]))
        unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
        rot_new =  rot_original + rot

        new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
        
        new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
        x = new_global_grasp[0]
        y = new_global_grasp[1]
        rz = action[3] + rot
        rz = self._wrap_rz(rz)
        action_aug = np.array([x, y, action[2], rz, action[4]])
        return action_aug
    
    def _wrap_rz(self, original_rz):
        wrapped_rz = (original_rz + 90) % 180 - 90
        return wrapped_rz
    
    def __len__(self):
        """
        Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
        """
        return self.n_datapoints

    def __getitem__(self, idx):
        raw_traj_idx = int(idx // self.n_datapoints_per_trajectory) 
        # determine the rotation augmentation to apply
        aug_rot = (idx % self.n_datapoints_per_trajectory) * self.aug_step
        traj_path = self.dataset_dir + '/Trajectory' + str(raw_traj_idx)

        states = []
        actions = []
        centers = []
        j = 0

        while exists(traj_path + '/state' + str(j) + '.npy'):  
            ctr = np.load(traj_path + '/pcl_center' + str(j) + '.npy')
            s = np.load(traj_path + '/state' + str(j) + '.npy')
            s_rot = self._rotate_pcl(s, ctr, aug_rot)
            s_rot_scaled = self._center_pcl(s_rot, ctr)
            states.append(s_rot_scaled)

            if j != 0:
                # load unnormalized action
                a = np.load(traj_path + '/action' + str(j-1) + '.npy')
                a_rot = self._rotate_action(a, ctr, aug_rot)
                if self.center_action:
                    a_scaled = self._center_normalize_action(a_rot, ctr)
                    centers.append(ctr)
                else:
                    a_scaled = self._normalize_action(a_rot)
                    centers.append(ctr)
                actions.append(a_scaled)
            j+=1

        episode_len = len(actions)
        start_ts = np.random.choice(episode_len)
        state = states[start_ts]
        
        # load uncentered goal
        g = np.load(traj_path + '/new_goal_unnormalized.npy')
        g_rot = self._rotate_pcl(g, centers[start_ts], aug_rot)
        goal = self._center_pcl(g_rot, centers[start_ts])

        action = actions[start_ts:]
        action = np.stack(action, axis=0)

        # add in termination token -1 continue, 1 stop
        stop_token = -1 * np.ones((self.pred_horizon, 1))
        stop_token[-1] = 1
        action = c
        
        action_len = episode_len - start_ts

        if start_ts != 0:
            obs_pos = actions[start_ts-1]
        else:
            if self.center_action:
                obs_pos = self._center_normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.05]), centers[start_ts])
            else:
                obs_pos = self._normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.05]))

        if action_len < self.pred_horizon:
            padded_action = np.zeros((self.pred_horizon, 5))
            padded_action[:action_len] = action
            for i in range(action_len, self.pred_horizon):
                padded_action[i] = action[-1]
        else:
            padded_action = action[:self.pred_horizon]

        # construct observations
        state_data = torch.from_numpy(state)
        goal_data = torch.from_numpy(goal).float()
        action_data = torch.from_numpy(padded_action).float()
        obs_pos_data = torch.from_numpy(obs_pos).float()

        nsample = dict()
        nsample['pointcloud'] = state_data
        nsample['goal'] = goal_data
        nsample['action'] = action_data
        nsample['agent_pos'] = obs_pos_data
        return nsample