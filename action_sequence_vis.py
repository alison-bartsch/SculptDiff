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
    color_list = plt.cm.gist_rainbow(np.linspace(0, 1, n_actions))[:, :3]  # RGB colors

    gripper_meshes = []
    for a in range(n_actions):
        action = action_seq[a]

        # make scaling adjustments
        action[0] += 0.01 # 0.03
        # action[1] -= 0.025
        action[2] -= 0.04
        action[5] += 90

        elem_color = color_list[a]

        # create a list of 5 colors each morphing elem_color more towards white
        elem_color_list = [elem_color * (1 - i * 0.2) + np.array([1, 1, 1]) * (i * 0.2) for i in range(5)]
        # reverse elem_color_list to start with the original color and end with white
        elem_color_list.reverse()
        ees = np.linspace(action[6], 0.04, 5)  # create a list of 5 end-effector distances from action[6] to 0.05
        
        # create 5x meshes to show the squeeze path
        for i in range(5):
            # create a gripper mesh for each color in the list
            cylinder1, cylinder2 = create_grippers(action, color=elem_color_list[i], ee_dist=ees[i], cylinder_radius=0.003)
            gripper_meshes.append(cylinder1)
            gripper_meshes.append(cylinder2)

        # cylinder1, cylinder2 = create_grippers(action, color=elem_color) # TODO: add ee_dist parameter to visualize , ee_dist=action[6]
        # gripper_meshes.append(cylinder1)
        # gripper_meshes.append(cylinder2)
    return gripper_meshes
    

if __name__ == "__main__":
    # load in a point cloud
    starting_idx = 14
    pcl = np.load('/home/alison/Documents/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory2/unnormalized_pointcloud' + str(starting_idx) + '.npy')

    # create a list of size 16 in colorspace from black to white
    bw_colorlist = np.linspace(0, 1, 16)[:, None] * np.array([[1, 1, 1]])  # RGB colors from black to white

    # iterate through 16 actions to populate the action sequence
    action_seq = np.zeros((16, 7))  # Initialize an array for 16 actions, each with 7 dimensions
    for i in range(16):
        action_seq[i] = np.load('/home/alison/Documents/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory2/action7d_unnormalized' + str(i+starting_idx) + '.npy')
    print("Action sequence shape:", action_seq.shape)

    # get the gripper meshes for each action
    gripper_meshes = vis_gripper_sequence(action_seq)

    # make pcl slightly larger for visualization

    pointcloud_list = []
    for j in reversed(range(16)):
        pcl = np.load('/home/alison/Documents/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory2/unnormalized_pointcloud' + str(starting_idx + j) + '.npy')
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(pcl)
        pointcloud.colors = o3d.utility.Vector3dVector(np.array([bw_colorlist[j]] * pcl.shape[0]))  
        pointcloud_list.append(pointcloud)
        # o3d.visualization.draw_geometries([pointcloud])

    # visualize the gripper meshes with the point cloud
    # pointcloud = o3d.geometry.PointCloud()
    # pointcloud.points = o3d.utility.Vector3dVector(pcl)
    # pointcloud.colors = o3d.utility.Vector3dVector(np.array([[0.25, 0.25, .25]] * pcl.shape[0])) 
    # o3d.visualization.draw_geometries([pointcloud] + gripper_meshes)
    o3d.visualization.draw_geometries(pointcloud_list + gripper_meshes)