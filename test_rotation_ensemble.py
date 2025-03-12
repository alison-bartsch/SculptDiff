import os
import cv2
import time
import math
import torch
import queue
import threading
import numpy as np
import open3d as o3d
from pcl_utils import *
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def create_gripper_rectangle(action7d):
    # create dense 3D rectangle array
    rectangle = np.array([[x, y, z] for x in np.linspace(-0.025, 0.025, 10) for y in np.linspace(-0.05, 0.05, 10) for z in np.linspace(-0.01, 0.01, 10)])

    # set the rectangle rotation
    R = Rotation.from_euler('xyz', np.array([action7d[3], action7d[4], action7d[5]]), degrees=True).as_matrix()
    # R = Rotation.from_euler('xyz', np.array([0, 0, action7d[5]]), degrees=True).as_matrix()
    rectangle = rectangle 
    rectangle = R @ rectangle.T
    rectangle = rectangle.T

    # set rectangle x,y,z position
    tiled = np.tile(action7d[0:3], (len(rectangle),1))
    rectangle = rectangle + tiled
    return rectangle

def unnormalize_action(action):
    a_mins7d = np.array([0.5413, -0.04232, 0.1300, -45, -15, -90, 0.0005])
    a_maxs7d = np.array([0.6700, 0.08500, 0.1560, 45, 13, 90, 0.005])
    action = (action + 1) / 2
    action = action * (a_maxs7d - a_mins7d) + a_mins7d
    return action

def rotate_pcl(state, center, rot):
    '''
    Faster implementation of rotation augmentation to fix slow down issue
    '''
    state = state - center
    R = Rotation.from_euler('xyz', np.array([0, 0, rot]), degrees=True).as_matrix()
    state = R @ state.T
    pcl_aug = state.T + center
    return pcl_aug

def center_pcl(pcl, center):
    centered_pcl = pcl - center
    centered_pcl = centered_pcl * 10
    return centered_pcl

def rotate_action(action, center, rot):
    unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
    rot_original = math.degrees(math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0]))
    unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
    rot_new =  rot_original + rot

    new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
    
    new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
    x = new_global_grasp[0]
    y = new_global_grasp[1]
    rz = action[5] + rot
    rz_new = (rz + 90) % 180 - 90 # wrap rz

    # convert to radians
    r_x = math.radians(action[3])
    r_y = math.radians(action[4])
    z_rotation_change = math.radians(rot)

    # calculate the new pitch and roll
    rx_new = math.asin(math.cos(z_rotation_change)*math.sin(r_x) + math.sin(z_rotation_change)*math.cos(r_x)*math.sin(r_y))
    ry_new = math.asin(math.cos(r_x)*math.sin(r_y))

    # convert back to degrees
    rx_new = math.degrees(rx_new)
    ry_new = math.degrees(ry_new)

    action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same
    return action_aug

# define model checkpoint directory
ckpt_dir = '/home/alison/Documents/GitHub/SculptDiff/checkpoints/pottery_12pred_with_augs'

# define diffusion parameters
obs_horizon = 1
B = 1
pred_horizon = 12 
action_dim = 8
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# define the device
device = torch.device('cuda')

# define the action space limits for unnormalization
a_mins7d = np.array([0.5413, -0.04232, 0.1300, -360, -15, -90, 0.0005])
a_maxs7d = np.array([0.6700, 0.08500, 0.1560, 360, 130, 90, 0.005])

# define qpos for the current state
qpos = np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04])
qpos = (qpos - a_mins7d) / (a_maxs7d - a_mins7d)
qpos = qpos * 2.0 - 1.0
qpos = np.concatenate((qpos, np.array([-1.])), axis=0)
nagent_pos = torch.from_numpy(qpos).to(torch.float32).unsqueeze(axis=0).unsqueeze(axis=0).to(device)

# initialize the pointbert model
testconfig = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
testmodel_config = testconfig.model
pointbert = builder.model_builder(testmodel_config)
testweights_path = ckpt_dir + '/pointbert_statedict' 
pointbert.load_state_dict(torch.load(testweights_path))
pointbert.to(device)

# load projection head from ckpt_dir
enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint', map_location=torch.device('cpu')) 
projection_head = enc_checkpoint['encoder_head'].to(device)

# load noise_pred_net from ckpt_dir
noise_checkpoint = torch.load(ckpt_dir + '/noise_pred_best_checkpoint', map_location=torch.device('cpu')) 
noise_pred_net = noise_checkpoint['noise_pred_net'].to(device)

# load in the goal
raw_goal = np.load('/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory5/unnormalized_pointcloud22.npy')

# load in the current state observation
raw_state = np.load('/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory5/unnormalized_pointcloud9.npy')

# load in the current state center
center = np.load('/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory5/pcl_center9.npy')
starting_action_idx = 9

if starting_action_idx != 0:
    prev_action = np.load('/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory5/action7d_unnormalized' + str(starting_action_idx - 2) + '.npy')
    prev_action[3] = (prev_action[3] + 90) % 180 - 90
    prev_action = (prev_action - a_mins7d) / (a_maxs7d - a_mins7d)
    prev_action = prev_action * 2.0 - 1.0
    prev_action = np.concatenate((prev_action, np.array([-1.])), axis=0)
    nagent_pos = torch.from_numpy(prev_action).to(torch.float32).unsqueeze(axis=0).unsqueeze(axis=0).to(device)

generated_actions = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6 : [], 7 : [], 8 : [], 9 : [], 10 : [], 11 : []}
for rot in range(0, 360, 60):
    # process state 
    rot_state = rotate_pcl(raw_state, center, rot)
    centered_state = center_pcl(rot_state, center)

    # process goal 
    rot_goal = rotate_pcl(raw_goal, center, rot)
    centered_goal = center_pcl(rot_goal, center)

    # generate action from model
    state = torch.from_numpy(centered_state).to(torch.float32)
    state = state.contiguous()
    states = torch.unsqueeze(state, 0).to(device)
    tokenized_states = pointbert(states)
    pcl_embed = projection_head(tokenized_states)
    pointcloud_features = pcl_embed.unsqueeze(1).repeat(1, obs_horizon, 1)

    # pass the goal cloud through Point-BERT and projection head
    goal = torch.from_numpy(centered_goal).to(torch.float32)
    goal = goal.contiguous()
    goals = torch.unsqueeze(goal, 0).to(device)
    tokenized_goals = pointbert(goals)
    goal_embed = projection_head(tokenized_goals)
    goalcloud_features = goal_embed.unsqueeze(1).repeat(1, obs_horizon, 1)

    # concatenate vision feature and low-dim obs
    obs_features = torch.cat([pointcloud_features, nagent_pos, goalcloud_features],dim=-1)
    obs_cond = obs_features.flatten(start_dim=1)

    # generate actions 10x to see variance better
    for run in range(10):
        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, pred_horizon, action_dim), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        pred_action = naction[0]

        for i in range(pred_horizon):
            action = pred_action[i,0:7]
            unnorm_action = unnormalize_action(action)
            unrotated_action = rotate_action(unnorm_action, center, -rot)
            generated_actions[i].append(unrotated_action)

for k in range(pred_horizon):
    # compare the variance between generated actions with different rotation augmentations applied!
    # load in the g.t. action
    raw_action = np.load('/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory5/action7d_unnormalized' + str(starting_action_idx - 1 + k) + '.npy')
    raw_action[3] = (raw_action[3] + 90) % 180 - 90
    print("\n\n\n--------- k = ", k, "---------")
    print("\nGround Truth Action: ", raw_action)
    print("\nMean Generated Action: ", np.mean(generated_actions[k], axis=0))
    print("Variance Generated Action: ", np.var(generated_actions[k], axis=0))
    print("Stdev Generated Action: ", np.std(generated_actions[k], axis=0))

    # get axes with high variance (var > 0.1)
    variances = np.var(generated_actions[k], axis=0)
    high_var_axes = np.where(variances > 20)[0]

    # visualize the ground truth action in green
    pcl_rot_o3d = o3d.geometry.PointCloud()
    pcl_rot_o3d.points = o3d.utility.Vector3dVector(raw_state)
    pcl_rot_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(raw_state),1)))
    rectangle = create_gripper_rectangle(raw_action)
    rectangle_o3d = o3d.geometry.PointCloud()
    rectangle_o3d.points = o3d.utility.Vector3dVector(rectangle)
    rectangle_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,1,0]), (len(rectangle),1)))
    action_vis_list = [rectangle_o3d, pcl_rot_o3d]
    for gen_action in generated_actions[k]:
        rectangle_gen = create_gripper_rectangle(gen_action)
        rectangle_gen_o3d = o3d.geometry.PointCloud()
        rectangle_gen_o3d.points = o3d.utility.Vector3dVector(rectangle_gen)
        rectangle_gen_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(rectangle_gen),1)))
        action_vis_list.append(rectangle_gen_o3d)
    o3d.visualization.draw_geometries(action_vis_list)


    for axes in high_var_axes:
        print("\nHigh Variance Axis: ", axes)
        # round the values to 3 decimal places
        axes_vals = [round(action[axes], 3) for action in generated_actions[k]]
        print("Axes Vals: ", axes_vals)