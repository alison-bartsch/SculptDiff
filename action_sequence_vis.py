import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from test_collision_checker import create_grippers

def vis_gripper_sequence(action_seq, n_actions=16):
    '''
    Given an array of shape (n_actions, 7), visualize the gripper sequence.
    Each action is a 7D vector representing the gripper state.
    '''
    # create a list of n_actions discrete colors on the viridis colormap scale
    color_list = plt.cm.rainbow(np.linspace(0, 1, n_actions))[:, :3]  # RGB colors

    gripper_meshes = []
    for a in range(n_actions):
        action = action_seq[a]

        # make scaling adjustments
        action[0] += 0.03
        action[1] -= 0.025
        action[2] -= 0.04
        action[5] += 90

        elem_color = color_list[a]

        cylinder1, cylinder2 = create_grippers(action, color=elem_color) # TODO: add ee_dist parameter to visualize , ee_dist=action[6]
        gripper_meshes.append(cylinder1)
        gripper_meshes.append(cylinder2)
    return gripper_meshes
    

if __name__ == "__main__":
    # load in a point cloud
    pcl = np.load('/home/alison/Documents/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory2/unnormalized_pointcloud14.npy')

    # iterate through 16 actions to populate the action sequence
    action_seq = np.zeros((16, 7))  # Initialize an array for 16 actions, each with 7 dimensions
    for i in range(16):
        action_seq[i] = np.load('/home/alison/Documents/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory2/action7d_unnormalized' + str(i+14) + '.npy')
    print("Action sequence shape:", action_seq.shape)

    # get the gripper meshes for each action
    gripper_meshes = vis_gripper_sequence(action_seq)

    # visualize the gripper meshes with the point cloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcl)
    pointcloud.colors = o3d.utility.Vector3dVector(np.array([[0.25, 0.25, .25]] * pcl.shape[0]))  # Blue color for point cloud
    o3d.visualization.draw_geometries([pointcloud] + gripper_meshes)