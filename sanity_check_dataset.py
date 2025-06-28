import os
import cv2
import time
import math
import copy
import numpy as np
import open3d as o3d
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from test_collision_checker import create_grippers
from scipy.spatial.transform import Rotation

def rotate_pcl(state, center, rot):
    '''
    Faster implementation of rotation augmentation to fix slow down issue
    '''
    state = state - center
    R = Rotation.from_euler('xyz', np.array([0, 0, rot]), degrees=True).as_matrix()
    state = R @ state.T
    pcl_aug = state.T + center
    return pcl_aug

# def rotate_action(action, center, rot):
#     unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
#     rot_original = math.degrees(math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0]))
#     unit_circle_radius = math.sqrt(unit_circle_og_grasp[0]**2 + unit_circle_og_grasp[1]**2)
#     rot_new =  rot_original + rot

#     new_unit_circle_grasp = (unit_circle_radius*math.cos(math.radians(rot_new)), unit_circle_radius*math.sin(math.radians(rot_new)))
    
#     new_global_grasp = (center[0] + new_unit_circle_grasp[0], center[1] + new_unit_circle_grasp[1])
#     x = new_global_grasp[0]
#     y = new_global_grasp[1]
#     rz_new = action[5] + rot
#     # ensure rz_new is within [-180, 180]
#     rz_new = (rz_new + 180) % 360 - 180 # wrap rz

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

#     # if np.abs(rz_new) > 90:
#     #     rx_new = -rx_new
#     # #     ry_new = -ry_new

#     # if np.abs(rz_new) < 90:
#     #     ry_new = -ry_new # flip the roll if rz_new is less than 90 degrees

#     action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same
#     return action_aug

def rotate_action(action, center, rot):
    # convert rotation to radians
    # angle_rad = math.radians(rot)
    
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


    # if new_orientation[2] < -90:
    #     if new_orientation[0] < 0:
    #         new_orientation[1] = -new_orientation[1]
    #     else:
    #         new_orientation[0] = -new_orientation[0]
    #         new_orientation[1] = -new_orientation[1]

    # if new_orientation[2] > 90:
    #     new_orientation[1] = -new_orientation[1]

    # if np.abs(new_orientation[2]) < 90:
    #     new_orientation[1] = -new_orientation[1]

    return np.concatenate([new_position, new_orientation])

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

if __name__ == "__main__":

    for i in range(19,20):
        j = 10
        r_idx = 0
        traj_path = '/home/alison/Documents/June18_Human_Demos_Train/Trajectory' + str(i)

        # initialize rectangle labels
        elem = np.array([[x, y, z] for x in np.linspace(-0.025, 0.025, 10) for y in np.linspace(-0.05, 0.05, 10) for z in np.linspace(-0.01, 0.01, 10)])
        # all the points with positive y values assign label 1, negative y values assign label 0
        labels = np.where(elem[:, 1] > 0, 1, 0)

        while os.path.exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
            print("Traj: ", i, "action: ", j-1)
            # load unnormalized action
            action7d = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')

            # # flip the rotation once 
            # if np.abs(action7d[5]) > 90:
            #     action7d[3] = -action7d[3]




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

            # print action7d rounded to 3 decimal places
            print("Action7D: ", np.round(action7d, 3))

            # load state
            pcl_arr = np.load(traj_path + '/unnormalized_pointcloud' + str(j-1) + '.npy')
            pcl_o3d = o3d.geometry.PointCloud()
            pcl_o3d.points = o3d.utility.Vector3dVector(pcl_arr)
            pcl_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(pcl_arr),1)))

            # load in the center
            ctr = np.load(traj_path + '/pcl_center' + str(j-1) + '.npy')

            # create an arrow in open3d along the x axis
            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.05, cone_height=0.02)
            arrow.paint_uniform_color([1, 0, 0]) # red
            arrow.rotate(Rotation.from_euler('xyz', np.array([0, 90, 0]), degrees=True).as_matrix(), center=(0, 0, 0))
            arrow.translate((ctr[0], ctr[1], ctr[2]))

            # create a green arrow along the y axis
            arrow_y = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.01, cylinder_height=0.05, cone_height=0.02)
            arrow_y.paint_uniform_color([0, 1, 0]) # green
            arrow_y.rotate(Rotation.from_euler('xyz', np.array([90, 0, 0]), degrees=True).as_matrix(), center=(0, 0, 0))
            arrow_y.translate((ctr[0], ctr[1], ctr[2]))

            # visualize the action
            vis_action = copy.deepcopy(action7d)
            # make scaling adjustments
            # vis_action[0] += 0.03
            # vis_action[1] -= 0.025
            vis_action[2] -= 0.04
            vis_action[5] += 90
            
            # if np.abs(action7d[5]) > 90:
            #     vis_action[3] = -vis_action[3] # flip the x rotation
            
            c1_og, c2_og = create_grippers(vis_action, color=[1,0,0]) #create_gripper_rectangle(action7d)
            o3d.visualization.draw_geometries([c1_og, c2_og, pcl_o3d, arrow, arrow_y])

            # # visualize a modified action with flipped x rotation
            # vis_action = copy.deepcopy(action7d)
            # vis_action[2] -= 0.04
            # vis_action[5] += 90
            # vis_action[3] = -vis_action[3] # flip the x rotation
            # c1_x, c2_x = create_grippers(vis_action, color=[0,1,0]) #create_gripper_rectangle(action7d)
            # # o3d.visualization.draw_geometries([c1, c2, pcl_o3d, arrow, arrow_y])

            # # visualize a modified action with flipped y rotation
            # vis_action = copy.deepcopy(action7d)
            # vis_action[2] -= 0.04
            # vis_action[5] += 90
            # vis_action[4] = -vis_action[4] # flip the x rotation
            # c1_y, c2_y = create_grippers(vis_action, color=[0,0,1]) #create_gripper_rectangle(action7d)
            
            # # visualize a modified action with flipped y rotation
            # vis_action = copy.deepcopy(action7d)
            # vis_action[2] -= 0.04
            # vis_action[5] += 90
            # vis_action[4] = -vis_action[4] # flip the x rotation
            # vis_action[3] = -vis_action[3]
            # c1_xy, c2_xy = create_grippers(vis_action, color=[1,0,1]) #create_gripper_rectangle(action7d)
            # o3d.visualization.draw_geometries([c1_og, c2_og, c1_x, c2_x, c1_y, c2_y, c1_xy, c2_xy, pcl_o3d, arrow, arrow_y])

            vis_list = [c1_og, c2_og, pcl_o3d, arrow, arrow_y]

            for k in range(0,360,30):
                rotated_action = rotate_action(action7d, ctr, k)
                rotated_pcl = rotate_pcl(pcl_arr, ctr, k)

                
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
                
                
                
                vis_rot_action = copy.deepcopy(rotated_action)
                # make scaling adjustments
                # vis_rot_action[0] += 0.03
                # vis_rot_action[1] -= 0.025
                vis_rot_action[2] -= 0.04
                vis_rot_action[5] += 90

                # if np.abs(rotated_action[5]) > 90:
                #     vis_rot_action[3] = -vis_rot_action[3] # flip the x rotation

                c1_rot, c2_rot = create_grippers(vis_rot_action) # create_gripper_rectangle(rotated_action)
                rotated_pcl_o3d = o3d.geometry.PointCloud()
                rotated_pcl_o3d.points = o3d.utility.Vector3dVector(rotated_pcl)
                rotated_pcl_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(rotated_pcl),1)))
                # o3d.visualization.draw_geometries([c1_rot, c2_rot, rotated_pcl_o3d, arrow, arrow_y])

                vis_list.append(c1_rot)
                vis_list.append(c2_rot)
            
            o3d.visualization.draw_geometries(vis_list)

            j+=1