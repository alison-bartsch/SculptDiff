import numpy as np
from os.path import exists

mins = np.ones(7) * 1000
maxs = np.ones(7) * -1000

for i in range(6):
    traj_path = '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/Trajectory' + str(i)
    j = 0
    while exists(traj_path + '/action7d_unnormalized' + str(j) + '.npy'):  
        action = np.load(traj_path + '/action7d_unnormalized' + str(j) + '.npy')
        mins = np.minimum(mins, action)
        maxs = np.maximum(maxs, action)
        j+=1
print('\nmins', mins)
print('\nmaxs', maxs)

# mins = [0.5413, -0.04232, 0.1300, -360, -15, -90, 0.0005]
# maxs = [0.6700, 0.08500, 0.1560, 360, 130, 90, 0.005]