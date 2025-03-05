import numpy as np
from os.path import exists

j = 1
traj_path = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory1' 

while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
    # load unnormalized action
    a = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
    print("\nAction Rotations ", j, ": ", a[3:6])
    j+=1