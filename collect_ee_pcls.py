import os
import time
import random
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
# from robot_utils import *
from frankapy import FrankaArm
from scipy.spatial.transform import Rotation
import robomail.vision as vis
from skimage.color import rgb2lab

def goto_grasp(fa, x, y, z, rx, ry, rz, d):
    """
    Parameterize a grasp action by the position [x,y,z] Euler angle rotation [rx,ry,rz], and width [d] of the gripper.
    This function was designed to be used for clay moulding, but in practice can be applied to any task.

    :param fa:  franka robot class instantiation
    """
    pose = fa.get_pose()
    starting_rot = pose.rotation
    orig = Rotation.from_matrix(starting_rot)
    orig_euler = orig.as_euler('xyz', degrees=True)
    rot_vec = np.array([rx, ry, rz])
    new_euler = orig_euler + rot_vec
    r = Rotation.from_euler('xyz', new_euler, degrees=True)
    pose.rotation = r.as_matrix()
    pose.translation = np.array([x, y, z])

    fa.goto_pose(pose)
    # fa.goto_gripper(d, force=60.0)
    time.sleep(3)


# initialize the robot and reset joints
fa = FrankaArm()

# initialize the cameras
cam1 = vis.CameraClass(1)  # ee camera
cam2 = vis.CameraClass(2)
cam3 = vis.CameraClass(3)
cam4 = vis.CameraClass(4)
cam5 = vis.CameraClass(5)

# initialize the 3D vision code
pcl_vis = vis.Vision3D()

# create a list of poses 
action_list = ['ee_pcls/action0.npy',
               'ee_pcls/action1.npy',
               'ee_pcls/action2.npy']

# get the camera extrinsics for each of the cameras
ext1 = cam1.get_cam_extrinsics()
ext2 = cam2.get_cam_extrinsics()
ext3 = cam3.get_cam_extrinsics()
ext4 = cam4.get_cam_extrinsics()
ext5 = cam5.get_cam_extrinsics()

# reset franka to its home joints
fa.goto_gripper(0.04)
fa.reset_joints()
original_pose = fa.get_pose()

# define the original euler rotation
orig_rot = Rotation.from_matrix(original_pose.rotation)
orig_euler = orig_rot.as_euler('xyz', degrees=True) 

# define hovering pose
overhead_pose = original_pose.copy()
overhead_pose.translation = np.array([0.625, 0, 0.325]) 
fa.goto_pose(overhead_pose)

for i in range(len(action_list)):
    # load in the action
    action = np.load(action_list[i])

    # execute the action
    goto_grasp(fa, action[0], action[1], action[2] + 0.1, action[3], action[4], action[5], action[6])

    # get point clouds
    # _, _, pc1, _ = cam1._get_next_frame()
    _, _, pc2, _ = cam2._get_next_frame()
    _, _, pc3, _ = cam3._get_next_frame()
    _, _, pc4, _ = cam4._get_next_frame()
    _, _, pc5, _ = cam5._get_next_frame()

    # combine the point clouds into world coordinate frame
    cur_pose = fa.get_pose()
    ee_pos = cur_pose.translation
    ee_rot = cur_pose.rotation
    pose_transform = np.eye(4)
    pose_transform[:3,:3] = ee_rot
    pose_transform[0:3,3] = ee_pos
    # transform each cloud to world frame
    # pc1.transform(pose_transform).transform(self.camera_transforms[1])
    # pc1.transform(ext1).transform(pose_transform)
    # # handle calibration offsets
    # pc1.translate((-0.05,-0.025,-0.01))
    pc2.transform(ext2)
    pc3.transform(ext3)
    pc4.transform(ext4)
    pc5.transform(ext5)

    # combine the point clouds
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = pc5.points
    pointcloud.colors = pc5.colors
    # pointcloud.points.extend(pc1.points)
    # pointcloud.colors.extend(pc1.colors)
    pointcloud.points.extend(pc2.points)
    pointcloud.colors.extend(pc2.colors)
    pointcloud.points.extend(pc3.points)
    pointcloud.colors.extend(pc3.colors)
    pointcloud.points.extend(pc4.points)
    pointcloud.colors.extend(pc4.colors)

    # crop point cloud
    pointcloud, ind = pointcloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    # o3d.visualization.draw_geometries([pointcloud])

    # based on the action, crop the point clouds to isolate the end-effector region
    minz = action[2] + 0.1 - 0.1
    maxz = action[2] + 0.1 + 0.1
    minx = action[0] - 0.1
    maxx = action[0] + 0.1
    miny = action[1] - 0.1
    maxy = action[1] + 0.1
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)
    ind_x = np.where((points[:, 0] > minx) & (points[:, 0] < maxx))
    ind_y = np.where((points[:, 1] > miny) & (points[:, 1] < maxy))
    ind_z = np.where((points[:, 2] > minz) & (points[:, 2] < maxz))
    ind_combined = np.intersect1d(np.intersect1d(ind_x, ind_y), ind_z)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[ind_combined])
    pcd.colors = o3d.utility.Vector3dVector(colors[ind_combined])
    # visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    # keep only the white points
    cropped_pts = np.asarray(pcd.points)
    cropped_clrs = np.asarray(pcd.colors)
    # remove points that are too dark (i.e. black)
    lab_colors = rgb2lab(cropped_clrs.reshape(-1, 1, 3))  # Convert to LAB color space
    print("lab colors shape: ", lab_colors.shape)
    lightness = lab_colors[:, 0]  # Extract the lightness channel
    # threshold for lightness
    lightness_threshold = 70  # Adjust this threshold as needed
    light_points = np.where(lightness > lightness_threshold)
    # filter the points and colors
    filtered_points = cropped_pts[light_points]
    filtered_colors = cropped_clrs[light_points]
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    cropped_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    o3d.visualization.draw_geometries([cropped_pcd])
    # assert False
    
    # move to observation pose
    fa.goto_pose(overhead_pose)

    # # save the ee point cloud
    # np.save('ee_pcls/finger_pcl' + str(i) + '.npy', np.asarray(pcd.points))