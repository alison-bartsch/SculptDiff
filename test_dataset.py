import math
import torch
import json
import numpy as np
import open3d as o3d 
from os.path import exists
from PIL import Image
from scipy.spatial.transform import Rotation

# NOTE: updating for 7D actions with new fingertip tool for variable pot creation (final pcl is the goal)
    # for this initial test - do no rotation augmentations
    # for this initial test - do not wrap the rotations, just normalize based on global min/max
    # 

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
        # TODO: update the mins and maxs to be the correct range for the dataset!
        mins = np.array([-0.15, -0.15, -0.05, -90, 0.005])
        maxs = np.array([0.15, 0.15, 0.05, 90, 0.05])
        norm_action = np.zeros(5)
        norm_action[0:5] = (new_action[0:5] - mins) / (maxs - mins)
        norm_action = norm_action * 2 - 1 # set to [-1, 1]
        return norm_action

    def _normalize_action(self, action):
        # mins = [0.5413, -0.04232, 0.1300, -360, -15, -90, 0.0005]
        # maxs = [0.6700, 0.08500, 0.1560, 360, 130, 90, 0.005]

        # # -------- min/max values for 7 demos from \Mar24_Bowl_Demos_Soft_Finger ------
        # a_mins7d = np.array([0.5413, -0.04232, 0.1300, -45, -15, -90, 0.0005])
        # a_maxs7d = np.array([0.6700, 0.08500, 0.1560, 45, 13, 90, 0.005])


        # # ------- min/max values for 20 concave/convex demos from \June18_Human_Demos -----
        # a_mins7d = np.array([0.5340, -0.0549, 0.1272, -360, -10.10, -180, 0.008])
        # a_maxs7d = np.array([0.6749, 0.0871, 0.1600, 360, 11.68, 180, 0.016])
        

        a_mins7d = np.array([0.5340, -0.0549, 0.1272, -360, -11.68, -180, 0.008])
        a_maxs7d = np.array([0.6749, 0.0871, 0.1600, 360, 10.10, 180, 0.016])

        norm_action = (action - a_mins7d) / (a_maxs7d - a_mins7d)
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

    # def _rotate_action(self, action, center, rot):
    #     unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
    #     rot_original = math.degrees(math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0]))
    #     unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
    #     rot_new =  rot_original + rot

    #     new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
        
    #     new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
    #     x = new_global_grasp[0]
    #     y = new_global_grasp[1]
    #     rz = action[5] + rot
    #     rz_new = (rz + 90) % 180 - 90 # wrap rz

    #     # convert to radians
    #     r_x = math.radians(action[3])
    #     r_y = math.radians(action[4])
    #     z_rotation_change = math.radians(rot)

    #     # calculate the new pitch and roll
    #     rx_new = math.asin(math.cos(z_rotation_change)*math.sin(r_x) + math.sin(z_rotation_change)*math.cos(r_x)*math.sin(r_y))
    #     ry_new = math.asin(math.cos(r_x)*math.sin(r_y))

    #     # convert back to degrees
    #     rx_new = math.degrees(rx_new)
    #     ry_new = math.degrees(ry_new)

    #     action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same
    #     return action_aug

    def _rotate_action(self, action, center, rot):
        pose_6d = action[:6]
        position = pose_6d[:3]
        orientation = pose_6d[3:]

        cx, cy, cz = center
        px, py, pz = position

        # Step 1: Translate point to origin (centered at cx, cy)
        dx = px - cx
        dy = py - cy

        # Step 2: Rotate in XY plane
        cos_theta = np.cos(math.radians(rot))
        sin_theta = np.sin(math.radians(rot))

        rotated_x = cos_theta * dx - sin_theta * dy + cx
        rotated_y = sin_theta * dx + cos_theta * dy + cy
        rotated_z = pz  # unchanged

        new_position = np.array([rotated_x, rotated_y, rotated_z])

        # Step 3: Update orientation
        # Apply additional rotation about the Z-axis (pre-multiplied)
        original_rot = Rotation.from_euler('xyz', orientation, degrees=True)
        z_rotation = Rotation.from_euler('z', rot, degrees=True)
        new_rot =  original_rot * z_rotation
        new_orientation = new_rot.as_euler('xyz', degrees=True)

        return np.concatenate([new_position, new_orientation, action[6:]])
    
    def _wrap_rz(self, original_rz):
        wrapped_rz = (original_rz + 90) % 180 - 90
        return wrapped_rz

    def _fix_action_rotations(self, action7d):
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

        return action7d
    
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

        while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
            ctr = np.load(traj_path + '/pcl_center' + str(j) + '.npy')
            s = np.load(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy')
            s_rot = self._rotate_pcl(s, ctr, aug_rot)
            s_rot_scaled = self._center_pcl(s_rot, ctr)
            states.append(s_rot_scaled)

            if j != 0:
                # load unnormalized action
                a = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
                # # fix the r_x scaling
                # # a[3] = self._wrap_rz(a[3])
                # # NOTE: need to go through and verify the action is correct (i.e. wrapping rz is flipping rx, etc.)
                # a_rot = self._rotate_action(a, ctr, aug_rot)

                a = self._fix_action_rotations(a)
                a_rot = self._rotate_action(a, ctr, aug_rot)
                a_rot = self._fix_action_rotations(a_rot)
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
        g = np.load(traj_path + '/unnormalized_pointcloud' + str(j-1) + '.npy') # set the goal point cloud to be the last pcl in demo trajectory
        g_rot = self._rotate_pcl(g, centers[start_ts], aug_rot)
        goal = self._center_pcl(g_rot, centers[start_ts])

        action = actions[start_ts:]
        action = np.stack(action, axis=0)

        # add in termination token -1 continue, 1 stop
        stop_token = -1 * np.ones((action.shape[0], 1))
        stop_token[-1] = 1
        action = np.concatenate((action, stop_token), axis=1)
        
        action_len = episode_len - start_ts

        if start_ts != 0:
            obs_pos = actions[start_ts-1]
        else:
            if self.center_action:
                obs_pos = self._center_normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]), centers[start_ts])
            else:
                obs_pos = self._normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]))
        
        # add padding to obs_pos of one 0 vector to make 8d
        obs_pos = np.concatenate((obs_pos, -1 * np.ones((1))), axis=0)

        if action_len < self.pred_horizon:
            padded_action = np.zeros((self.pred_horizon, 8))
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

class ClayDatasetForwardBackward(torch.utils.data.Dataset):
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
        # TODO: update the mins and maxs to be the correct range for the dataset!
        mins = np.array([-0.15, -0.15, -0.05, -90, 0.005])
        maxs = np.array([0.15, 0.15, 0.05, 90, 0.05])
        norm_action = np.zeros(5)
        norm_action[0:5] = (new_action[0:5] - mins) / (maxs - mins)
        norm_action = norm_action * 2 - 1 # set to [-1, 1]
        return norm_action

    def _normalize_action(self, action):
        # mins = [0.5413, -0.04232, 0.1300, -360, -15, -90, 0.0005]
        # maxs = [0.6700, 0.08500, 0.1560, 360, 130, 90, 0.005]

        # # -------- min/max values for 7 demos from \Mar24_Bowl_Demos_Soft_Finger ------
        # a_mins7d = np.array([0.5413, -0.04232, 0.1300, -45, -15, -90, 0.0005])
        # a_maxs7d = np.array([0.6700, 0.08500, 0.1560, 45, 13, 90, 0.005])


        # # ------- min/max values for 20 concave/convex demos from \June18_Human_Demos -----
        # a_mins7d = np.array([0.5340, -0.0549, 0.1272, -360, -10.10, -180, 0.008])
        # a_maxs7d = np.array([0.6749, 0.0871, 0.1600, 360, 11.68, 180, 0.016])

        a_mins7d = np.array([0.5340, -0.0549, 0.1272, -360, -11.68, -180, 0.008])
        a_maxs7d = np.array([0.6749, 0.0871, 0.1600, 360, 10.10, 180, 0.016])
        

        norm_action = (action - a_mins7d) / (a_maxs7d - a_mins7d)
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

    # def _rotate_action(self, action, center, rot):
    #     unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
    #     rot_original = math.degrees(math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0]))
    #     unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
    #     rot_new =  rot_original + rot

    #     new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
        
    #     new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
    #     x = new_global_grasp[0]
    #     y = new_global_grasp[1]
    #     rz = action[5] + rot
    #     rz_new = (rz + 90) % 180 - 90 # wrap rz

    #     # convert to radians
    #     r_x = math.radians(action[3])
    #     r_y = math.radians(action[4])
    #     z_rotation_change = math.radians(rot)

    #     # calculate the new pitch and roll
    #     rx_new = math.asin(math.cos(z_rotation_change)*math.sin(r_x) + math.sin(z_rotation_change)*math.cos(r_x)*math.sin(r_y))
    #     ry_new = math.asin(math.cos(r_x)*math.sin(r_y))

    #     # convert back to degrees
    #     rx_new = math.degrees(rx_new)
    #     ry_new = math.degrees(ry_new)

    #     action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same
    #     return action_aug
    
    def _rotate_action(self, action, center, rot):
        pose_6d = action[:6]
        position = pose_6d[:3]
        orientation = pose_6d[3:]

        cx, cy, cz = center
        px, py, pz = position

        # Step 1: Translate point to origin (centered at cx, cy)
        dx = px - cx
        dy = py - cy

        # Step 2: Rotate in XY plane
        cos_theta = np.cos(math.radians(rot))
        sin_theta = np.sin(math.radians(rot))

        rotated_x = cos_theta * dx - sin_theta * dy + cx
        rotated_y = sin_theta * dx + cos_theta * dy + cy
        rotated_z = pz  # unchanged

        new_position = np.array([rotated_x, rotated_y, rotated_z])

        # Step 3: Update orientation
        # Apply additional rotation about the Z-axis (pre-multiplied)
        original_rot = Rotation.from_euler('xyz', orientation, degrees=True)
        z_rotation = Rotation.from_euler('z', rot, degrees=True)
        new_rot =  original_rot * z_rotation
        new_orientation = new_rot.as_euler('xyz', degrees=True)

        return np.concatenate([new_position, new_orientation, action[6:]])
    
    def _wrap_rz(self, original_rz):
        wrapped_rz = (original_rz + 90) % 180 - 90
        return wrapped_rz

    def _fix_action_rotations(self, action7d):
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

        return action7d
    
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

        while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
            ctr = np.load(traj_path + '/pcl_center' + str(j) + '.npy')
            s = np.load(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy')
            s_rot = self._rotate_pcl(s, ctr, aug_rot)
            s_rot_scaled = self._center_pcl(s_rot, ctr)
            states.append(s_rot_scaled)

            if j != 0:
                # load unnormalized action
                a = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
                # fix the r_x scaling
                # a[3] = self._wrap_rz(a[3])
                # NOTE: need to go through and verify the action is correct (i.e. wrapping rz is flipping rx, etc.)
                # a_rot = self._rotate_action(a, ctr, aug_rot)

                a = self._fix_action_rotations(a)
                a_rot = self._rotate_action(a, ctr, aug_rot)
                a_rot = self._fix_action_rotations(a_rot)
                if self.center_action:
                    a_scaled = self._center_normalize_action(a_rot, ctr)
                    centers.append(ctr)
                else:
                    a_scaled = self._normalize_action(a_rot)
                    centers.append(ctr)
                actions.append(a_scaled)
            j+=1

        # TODO: then will need to pad the previous action to also be the pred_horizon length
            # NOTE: instead of zero padding, we will pad with obs pos (normalized)

        episode_len = len(actions)
        start_ts = np.random.choice(episode_len)
        state = states[start_ts]
        
        # load uncentered goal
        g = np.load(traj_path + '/unnormalized_pointcloud' + str(j-1) + '.npy') # set the goal point cloud to be the last pcl in demo trajectory
        g_rot = self._rotate_pcl(g, centers[start_ts], aug_rot)
        goal = self._center_pcl(g_rot, centers[start_ts])

        action = actions[start_ts:]
        action = np.stack(action, axis=0)

        # add in termination token -1 continue, 1 stop
        stop_token = -1 * np.ones((action.shape[0], 1))
        stop_token[-1] = 1
        action = np.concatenate((action, stop_token), axis=1)
        
        action_len = episode_len - start_ts

        if start_ts != 0:
            obs_pos = actions[start_ts-1]
        else:
            if self.center_action:
                obs_pos = self._center_normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]), centers[start_ts])
            else:
                obs_pos = self._normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]))
        
        # add padding to obs_pos of one 0 vector to make 8d
        obs_pos = np.concatenate((obs_pos, -1 * np.ones((1))), axis=0)

        if action_len < self.pred_horizon:
            padded_action = np.zeros((self.pred_horizon, 8))
            padded_action[:action_len] = action
            for i in range(action_len, self.pred_horizon):
                padded_action[i] = action[-1]
        else:
            padded_action = action[:self.pred_horizon]

        # get previous actions
        prev_actions = actions[0:start_ts]
        # reverse the previous actions to get the backward trajectory
        prev_actions = prev_actions[::-1] # this way we get most recent previous action first
        prev_actions = np.stack(prev_actions, axis=0)
        prev_stop_tokens = -1 * np.ones((prev_actions.shape[0], 1))
        prev_actions = np.concatenate((prev_actions, prev_stop_tokens), axis=1)
        prev_action_len = start_ts

        if prev_action_len < self.pred_horizon:
            padded_prev_action = np.zeros((self.pred_horizon, 8))
            padded_prev_action[:prev_action_len] = prev_actions
            for i in range(prev_action_len, self.pred_horizon):
                padded_prev_action[i] = self._normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04])) # pad with obs pos
        else:
            padded_prev_action = prev_actions[:self.pred_horizon]

        # reverse the padded previous actions again 
        padded_prev_action = padded_prev_action[::-1]

        # combine the prev actions and actions to get continuous action sequence
        full_padded_action = np.concatenate((padded_prev_action, padded_action), axis=0)

        # construct observations
        state_data = torch.from_numpy(state)
        goal_data = torch.from_numpy(goal).float()
        action_data = torch.from_numpy(full_padded_action).float()
        obs_pos_data = torch.from_numpy(obs_pos).float()

        nsample = dict()
        nsample['pointcloud'] = state_data
        nsample['goal'] = goal_data
        nsample['action'] = action_data
        nsample['agent_pos'] = obs_pos_data
        return nsample
    
class SubGoalClayDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pred_horizon, n_datapoints, n_raw_trajectories, center_action, subgoal_stepsize):
        """
        The Dataloader for the clay sculpting dataset at the Trajectory level (compatible with ACT and Diffusion Policy). 

        :param episode_idxs: list of indices of the episodes to load
        :param dataset_dir: directory where the dataset is stored
        :param n_datapoints: number of datapoints (i.e. desired number of final trajectories after augmentation)
        :param n_raw_trajectories: number of raw trajectories in the dataset
        :param center_action: whether to center the action before normalizing
        """
        super(SubGoalClayDataset).__init__()
        self.dataset_dir = dataset_dir
        self.pred_horizon = pred_horizon
        self.n_datapoints = n_datapoints
        self.n_raw_trajectories = n_raw_trajectories
        self.center_action = center_action
        self.subgoal_stepsize = subgoal_stepsize
        self.center = np.array([0.628, 0.000, 0.104])

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
        # TODO: update the mins and maxs to be the correct range for the dataset!
        mins = np.array([-0.15, -0.15, -0.05, -90, 0.005])
        maxs = np.array([0.15, 0.15, 0.05, 90, 0.05])
        norm_action = np.zeros(5)
        norm_action[0:5] = (new_action[0:5] - mins) / (maxs - mins)
        norm_action = norm_action * 2 - 1 # set to [-1, 1]
        return norm_action

    def _normalize_action(self, action):
        # mins = [0.5413, -0.04232, 0.1300, -360, -15, -90, 0.0005]
        # maxs = [0.6700, 0.08500, 0.1560, 360, 130, 90, 0.005]

        # # -------- min/max values for 7 demos from \Mar24_Bowl_Demos_Soft_Finger ------
        # a_mins7d = np.array([0.5413, -0.04232, 0.1300, -45, -15, -90, 0.0005])
        # a_maxs7d = np.array([0.6700, 0.08500, 0.1560, 45, 13, 90, 0.005])


        # # ------- min/max values for 20 concave/convex demos from \June18_Human_Demos -----
        # a_mins7d = np.array([0.5340, -0.0549, 0.1272, -360, -10.10, -180, 0.008])
        # a_maxs7d = np.array([0.6749, 0.0871, 0.1600, 360, 11.68, 180, 0.016])

        a_mins7d = np.array([0.5340, -0.0549, 0.1272, -360, -11.68, -180, 0.008])
        a_maxs7d = np.array([0.6749, 0.0871, 0.1600, 360, 10.10, 180, 0.016])

        norm_action = (action - a_mins7d) / (a_maxs7d - a_mins7d)
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

    # def _rotate_action(self, action, center, rot):
    #     unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
    #     rot_original = math.degrees(math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0]))
    #     unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
    #     rot_new =  rot_original + rot

    #     new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
        
    #     new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
    #     x = new_global_grasp[0]
    #     y = new_global_grasp[1]
    #     rz = action[5] + rot
    #     rz_new = (rz + 90) % 180 - 90 # wrap rz

    #     # convert to radians
    #     r_x = math.radians(action[3])
    #     r_y = math.radians(action[4])
    #     z_rotation_change = math.radians(rot)

    #     # calculate the new pitch and roll
    #     rx_new = math.asin(math.cos(z_rotation_change)*math.sin(r_x) + math.sin(z_rotation_change)*math.cos(r_x)*math.sin(r_y))
    #     ry_new = math.asin(math.cos(r_x)*math.sin(r_y))

    #     # convert back to degrees
    #     rx_new = math.degrees(rx_new)
    #     ry_new = math.degrees(ry_new)

    #     action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same
    #     return action_aug

    def _rotate_action(self, action, center, rot):
        pose_6d = action[:6]
        position = pose_6d[:3]
        orientation = pose_6d[3:]

        cx, cy, cz = center
        px, py, pz = position

        # Step 1: Translate point to origin (centered at cx, cy)
        dx = px - cx
        dy = py - cy

        # Step 2: Rotate in XY plane
        cos_theta = np.cos(math.radians(rot))
        sin_theta = np.sin(math.radians(rot))

        rotated_x = cos_theta * dx - sin_theta * dy + cx
        rotated_y = sin_theta * dx + cos_theta * dy + cy
        rotated_z = pz  # unchanged

        new_position = np.array([rotated_x, rotated_y, rotated_z])

        # Step 3: Update orientation
        # Apply additional rotation about the Z-axis (pre-multiplied)
        original_rot = Rotation.from_euler('xyz', orientation, degrees=True)
        z_rotation = Rotation.from_euler('z', rot, degrees=True)
        new_rot =  original_rot * z_rotation
        new_orientation = new_rot.as_euler('xyz', degrees=True)

        return np.concatenate([new_position, new_orientation, action[6:]])
    
    def _wrap_rz(self, original_rz):
        wrapped_rz = (original_rz + 90) % 180 - 90
        return wrapped_rz

    def _fix_action_rotations(self, action7d):
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

        return action7d
    
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

        # print("Traj path: ", traj_path)

        states = []
        actions = []
        centers = []
        j = 0

        while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
            # print("exists")
            s = np.load(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy')
            s_rot = self._rotate_pcl(s, self.center, aug_rot)
            s_rot_scaled = self._center_pcl(s_rot, self.center)
            states.append(s_rot_scaled)

            if j != 0:
                # load unnormalized action
                a = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
                a = self._fix_action_rotations(a)
                a_rot = self._rotate_action(a, self.center, aug_rot)
                a_rot = self._fix_action_rotations(a_rot)
                if self.center_action:
                    a_scaled = self._center_normalize_action(a_rot, self.center)
                    centers.append(self.center)
                else:
                    a_scaled = self._normalize_action(a_rot)
                    centers.append(self.center)
                actions.append(a_scaled)
            j+=1

        # episode_len = len(actions)
        full_episode_len = len(actions)
        start_ts = np.random.choice(full_episode_len - self.subgoal_stepsize - 1)
        full_action_len = full_episode_len - start_ts

        # action = actions[start_ts:]
        if full_action_len >= self.pred_horizon:
            action = actions[start_ts:start_ts + self.pred_horizon]
            state_list = states[start_ts:(start_ts + self.pred_horizon + self.subgoal_stepsize):self.subgoal_stepsize]
        else:
            action = actions[start_ts:]
            state_list = states[start_ts::self.subgoal_stepsize]

        action_len = len(action)
        action = np.stack(action, axis=0)

        # add in termination token -1 continue, 1 stop
        stop_token = -1 * np.ones((action.shape[0], 1))
        stop_token[-1] = 1
        action = np.concatenate((action, stop_token), axis=1)
        
        if start_ts != 0:
            obs_pos = actions[start_ts-1]
        else:
            if self.center_action:
                obs_pos = self._center_normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]), centers[start_ts])
            else:
                obs_pos = self._normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]))
        
        # add padding to obs_pos of one 0 vector to make 8d
        obs_pos = np.concatenate((obs_pos, -1 * np.ones((1))), axis=0)

        states_seq_size = int((self.pred_horizon + self.subgoal_stepsize) / self.subgoal_stepsize)

        if action_len < self.pred_horizon:
            padded_action = np.zeros((self.pred_horizon, 8))
            padded_action[:action_len] = action
            for i in range(action_len, self.pred_horizon):
                padded_action[i] = action[-1]

            padded_states = np.zeros((states_seq_size, 2048, 3))
            padded_states[:len(state_list)] = np.stack(state_list, axis=0)
        else:
            padded_action = action[:self.pred_horizon]
            padded_states = np.stack(state_list, axis=0)

        # construct observations
        padded_states_data = torch.from_numpy(padded_states).float()
        action_data = torch.from_numpy(padded_action).float()
        obs_pos_data = torch.from_numpy(obs_pos).float()

        nsample = dict()
        nsample['pcl_seq'] = padded_states_data
        nsample['action'] = action_data
        nsample['agent_pos'] = obs_pos_data
        return nsample