import os
import math
import numpy as np
import open3d as o3d
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

def rotate_pcl(state, center, rot):
    '''
    Faster implementation of rotation augmentation to fix slow down issue
    '''
    state = state - center
    R = Rotation.from_euler('xyz', np.array([0, 0, rot]), degrees=True).as_matrix()
    state = R @ state.T
    pcl_aug = state.T + center
    return pcl_aug

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

# NOTE: as we are changing the z rotation, we want to preserve the pitch and roll, but w.r.t the new z rotation

def preserve_pitch_roll(r_x, r_y, r_z, z_rotation_change):
    '''
    Given z_rotation change, we add that rotation change to r_z (update the yaw).
    However, we would like to update r_x and r_y to preserve the pitch and roll w.r.t the new yaw.
    '''
    # convert to radians
    r_x = math.radians(r_x)
    r_y = math.radians(r_y)
    r_z = math.radians(r_z)
    z_rotation_change = math.radians(z_rotation_change)

    # calculate the new pitch and roll
    r_x_new = math.asin(math.cos(z_rotation_change)*math.sin(r_x) + math.sin(z_rotation_change)*math.cos(r_x)*math.sin(r_y))
    r_y_new = math.asin(math.cos(r_x)*math.sin(r_y))
    r_z_new = r_z + z_rotation_change

    # convert back to degrees
    r_x_new = math.degrees(r_x_new)
    r_y_new = math.degrees(r_y_new)
    r_z_new = math.degrees(r_z_new)

    return r_x_new, r_y_new, r_z_new

for i in range(6):
    j = 1
    r_idx = 0
    traj_path = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory' + str(i) 
    # traj_path = '/home/alison/Documents/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory' + str(i)
    # traj_path = '/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory' + str(i)

    while os.path.exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
        print("Traj: ", i, "action: ", j-1)
        # load unnormalized action
        action7d = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')

        # load state
        pcl_arr = np.load(traj_path + '/unnormalized_pointcloud' + str(j-1) + '.npy')
        pcl_o3d = o3d.geometry.PointCloud()
        pcl_o3d.points = o3d.utility.Vector3dVector(pcl_arr)
        pcl_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(pcl_arr),1)))

        # # find the points in pcl_arr that are max in the x direction (sanity check that the x axis arrow is correct)
        # max_x = np.max(pcl_arr[:,0])
        # max_x_points = pcl_arr[pcl_arr[:,0] == max_x]
        # pcl_x = o3d.geometry.PointCloud()
        # pcl_x.points = o3d.utility.Vector3dVector(max_x_points)
        # pcl_x.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(max_x_points),1)))

        # load in the center
        ctr = np.load(traj_path + '/pcl_center' + str(j-1) + '.npy')

        # create an arrow in open3d along the x axis
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.05, cone_height=0.02)
        arrow.paint_uniform_color([1, 0, 0]) # red
        arrow.rotate(Rotation.from_euler('xyz', np.array([0, 90, 0]), degrees=True).as_matrix(), center=(0, 0, 0))
        arrow.translate((ctr[0], ctr[1], ctr[2]))

        # visualize the action
        rectangle = create_gripper_rectangle(action7d)
        rectangle_o3d = o3d.geometry.PointCloud()
        rectangle_o3d.points = o3d.utility.Vector3dVector(rectangle)
        rectangle_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,1,0]), (len(rectangle),1)))
        o3d.visualization.draw_geometries([rectangle_o3d, pcl_o3d, arrow])

        # if r_idx > 0:
        #     np.save(traj_path + '/action7d_unnormalized' + str(j-1-r_idx) + '.npy', action7d)
        #     np.save(traj_path + '/unnormalized_pointcloud' + str(j-1-r_idx) + '.npy', pcl_arr)
        #     np.save(traj_path + '/pcl_center' + str(j-1-r_idx) + '.npy', ctr)

        # if yes, flip the rx to be negative
        user_response = input("Is the action flipped? (y/n) Is the action bad? (r): ")
        if user_response.lower() == 'y':
            action7d[3] = -action7d[3]
            np.save(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy', action7d)
            print("Updated action7d_unnormalized" + str(j-1) + ".npy")

            # load unnormalized action
            action7d = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')

            # visualize the action
            rectangle = create_gripper_rectangle(action7d)
            rectangle_o3d = o3d.geometry.PointCloud()
            rectangle_o3d.points = o3d.utility.Vector3dVector(rectangle)
            rectangle_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,1,0]), (len(rectangle),1)))
            o3d.visualization.draw_geometries([rectangle_o3d, pcl_o3d, arrow])

        # TODO: have r trigger a remove an action (increment the removal idx theshold)
        # elif user_response.lower() == 'r':
        #     r_idx += 1
        #     print("Removing Trajectory " + str(j-1) + " and saving to action7d_unnormalized" + str(j-1-r_idx) + ".npy")

        # else, do nothing
        j+=1