# import os
# import cv2
# import time
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

# TODO: visualization script iterating through pottery demonstrations while visualizing each grasp action overlaid
dataset_dir = '/home/acar/Clay_Demos/Pottery_Mar24/'

for i in range(32):
    state_path = dataset_dir + '/Trajectory0/unnormalized_pointcloud' + str(i) + '.npy'
    s1_pcl = np.load(state_path)

    # state 2 for pcl difference visualization
    s2_pcl = np.load(dataset_dir + '/Trajectory0/unnormalized_pointcloud' + str(i+1) + '.npy')

    raw_action = np.load(dataset_dir + '/Trajectory0/action7d_unnormalized' + str(i) + '.npy') # this is the original action
    rectangle = create_gripper_rectangle(raw_action)
    raw_action[3] = (raw_action[3] + 90) % 180 - 90
    rectangle_o3d = o3d.geometry.PointCloud()
    rectangle_o3d.points = o3d.utility.Vector3dVector(rectangle)
    rectangle_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(rectangle),1)))

    # visualize the original point clouds
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(s1_pcl)
    pcd2.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]]*s1_pcl.shape[0]))
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(s2_pcl)
    pcd3.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]*s2_pcl.shape[0]))
    o3d.visualization.draw_geometries([pcd2, pcd3, rectangle_o3d])