import numpy as np

base_path = '/home/alison/Documents/June18_Human_Demos_Train/Trajectory17/'

for i in range(1,30):
    # ctr = np.load(base_path + 'pcl_center' + str(i) + '.npy')
    # pcl = np.load(base_path + 'unnormalized_pointcloud' + str(i) + '.npy')
    action = np.load(base_path + 'action7d_unnormalized' + str(i-1) + '.npy')
    print("Action: ", action)
    # np.save(base_path + 'pcl_center' + str(i-1) + '.npy', ctr)
    # np.save(base_path + 'unnormalized_pointcloud' + str(i-1) + '.npy', pcl)
    # np.save(base_path + 'action7d_unnormalized' + str(i-2) + '.npy', action)