import os
import numpy as np
import open3d as o3d

traj_path = '/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory1'
for i in range(19,23):
    action = np.load(traj_path + '/action7d_unnormalized' + str(i) + '.npy')
    np.save(traj_path + '/action7d_unnormalized' + str(i-1) + '.npy', action)

    action5d = np.load(traj_path + '/action5d_unnormalized' + str(i) + '.npy')
    np.save(traj_path + '/action5d_unnormalized' + str(i-1) + '.npy', action5d)

    ctr = np.load(traj_path + '/pcl_center' + str(i) + '.npy')
    np.save(traj_path + '/pcl_center' + str(i-1) + '.npy', ctr)

    pcl = np.load(traj_path + '/unnormalized_pointcloud' + str(i) + '.npy')
    np.save(traj_path + '/unnormalized_pointcloud' + str(i-1) + '.npy', pcl)

# for i in range(1):
#     j = 1
#     r_idx = 0
#     # traj_path = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory' + str(i) 
#     traj_path = '/home/alison/Documents/Apr17_Human_Demos_Difficult_Shapes/pottery/Trajectory' + str(i)
#     # traj_path = '/home/alison/Documents/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory' + str(i)

#     while os.path.exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
#         j+=1

#     # load state
#     print("Traj: ", i)
#     pcl_arr = np.load(traj_path + '/unnormalized_pointcloud' + str(j-1) + '.npy')
#     pcl_o3d = o3d.geometry.PointCloud()
#     pcl_o3d.points = o3d.utility.Vector3dVector(pcl_arr)
#     pcl_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(pcl_arr),1)))
#     o3d.visualization.draw_geometries([pcl_o3d])