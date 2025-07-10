import copy
import numpy as np
import open3d as o3d
from pcl_utils import *
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error

# start with importing the final pcl from a single experiment to visualize and sanity check everything!
# then build into a dictionary of experiments for each measurement/final 3d goal

# # pointbert forward
# method_dict = {8 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory6/unnormalized_pointcloud29.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp3/pcl31.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp6/pcl29.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp7/pcl35.npy'],
#                     },
#                10 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory3/unnormalized_pointcloud28.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp26/pcl27.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp8/pcl27.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp9/pcl30.npy'],
#                     },
#                12 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory9/unnormalized_pointcloud22.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp18/pcl30.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp19/pcl27.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp20/pcl30.npy'],
#                     }}

# # pointbert subgoal
# method_dict = {8 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory6/unnormalized_pointcloud29.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp17/pcl40.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp19/pcl40.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp20/pcl40.npy'],
#                     },
#                10 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory3/unnormalized_pointcloud28.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp13/pcl40.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp15/pcl40.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp24/pcl32.npy'],
#                     },
#                12 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory9/unnormalized_pointcloud22.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp21/pcl32.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp22/pcl32.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Subgoal_Exp23/pcl26.npy'],
#                     }}

# # pointbert cont guidance
# method_dict = {8 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory6/unnormalized_pointcloud29.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp35/pcl58.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp36/pcl38.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp37/pcl18.npy'],
#                     },
#                10 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory3/unnormalized_pointcloud28.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp38/pcl31.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp39/pcl30.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp40/pcl31.npy'], 
#                     },
#                12 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory9/unnormalized_pointcloud22.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp41/pcl35.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp42/pcl28.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/Exp43/pcl30.npy'],
#                     }}

# # pointnet forward
# method_dict = {8 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory6/unnormalized_pointcloud29.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp5/pcl80.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp6/pcl54.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp7/pcl78.npy'],
#                     },
#                10 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory3/unnormalized_pointcloud28.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp2/pcl40.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp3/pcl30.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp4/pcl30.npy'],
#                     },
#                12 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory9/unnormalized_pointcloud22.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp8/pcl48.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp10/pcl48.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp11/pcl63.npy'],
#                     }}

# # pointnet subgoal
# method_dict = {8 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory6/unnormalized_pointcloud29.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp10/pcl40.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp11/pcl40.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp12/pcl40.npy'],
#                     },
#                10 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory3/unnormalized_pointcloud28.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp13/pcl32.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp14/pcl40.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp15/pcl40.npy'],
#                     },
#                12 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory9/unnormalized_pointcloud22.npy',
#                     'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp16/pcl32.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp17/pcl32.npy',
#                                        '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Subgoal_Exp18/pcl32.npy'],
#                     }}

# pointnet continual guidance
method_dict = {8 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory6/unnormalized_pointcloud29.npy',
                    'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp12/pcl84.npy',
                                       '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp13/pcl75.npy',
                                       '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp14/pcl60.npy'],
                    },
               10 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory3/unnormalized_pointcloud28.npy',
                    'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp15/pcl32.npy',
                                       '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp16/pcl56.npy',
                                       '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp17/pcl32.npy'],
                    },
               12 : {'goal_path' : '/home/alison/Clay_Data/June18_Human_Demos/pottery/Train/Trajectory9/unnormalized_pointcloud22.npy',
                    'exp_path_list' : ['/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp18/pcl64.npy',
                                       '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp19/pcl55.npy',
                                       '/home/alison/Documents/GitHub/SculptDiff/Experiments/PN_Exp20/pcl64.npy'],
                    }}

# # template
# method_dict = {8 : {'goal_path' : '',
#                     'exp_path_list' : ['',
#                                        '',
#                                        ''],
#                     },
#                10 : {'goal_path' : '',
#                     'exp_path_list' : ['',
#                                        '',
#                                        ''],
#                     },
#                12 : {'goal_path' : '',
#                     'exp_path_list' : ['',
#                                        '',
#                                        ''],
#                     }}

# define global pcl center
pcl_center = np.array([0.630, -0.0054, 0.074])

for diam in method_dict:

    cd_list = []
    emd_list = []
    mse_list = []

    # iterate through the experiments
    for i in range(len(method_dict[diam]['exp_path_list'])):


        # load in a pointcloud
        exp_pcl = np.load(method_dict[diam]['exp_path_list'][i])
        exp_pcl = (exp_pcl / 10.0) 
        # print("Mean exp pcl: ", np.mean(exp_pcl, axis=0))
        goal_pcl = np.load(method_dict[diam]['goal_path'])
        goal_pcl = goal_pcl - pcl_center
        # print("mean goal pcl: ", np.mean(goal_pcl, axis=0))

        # visualize the point clouds
        exp_pointcloud = o3d.geometry.PointCloud()
        exp_pointcloud.points = o3d.utility.Vector3dVector(exp_pcl)
        exp_pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(exp_pcl),1)))
        goal_pointcloud = o3d.geometry.PointCloud()
        goal_pointcloud.points = o3d.utility.Vector3dVector(goal_pcl)
        goal_pointcloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(goal_pcl),1)))
        goal_pointcloud, ind = goal_pointcloud.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
        # o3d.visualization.draw_geometries([exp_pointcloud, goal_pointcloud])

        # dist_metrics = {'CD': chamfer(exp_pcl, goal_pcl),
        #                     'EMD': emd(exp_pcl, goal_pcl)}

        # print("Dists before ICP: ", dist_metrics)


        # ICP tuning of the alignment to account for variation in clay placement
        target = copy.deepcopy(exp_pointcloud)
        source = copy.deepcopy(goal_pointcloud)

        # get icp aligmnet
        threshold = 0.01 #1 # 07 # 05
        trans_init = np.asarray([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])

        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, 
                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2500))
        source.transform(reg_p2p.transformation)

        # add a final combination of the point clouds and remove outliers one more time after combined
        calibrated_pcl = o3d.geometry.PointCloud()
        calibrated_pcl.points = target.points
        calibrated_pcl.colors = target.colors
        calibrated_pcl.points.extend(source.points)
        calibrated_pcl.colors.extend(source.colors)
     #    o3d.visualization.draw_geometries([calibrated_pcl])

        dist_metrics = {'CD': chamfer(np.asarray(source.points), np.asarray(target.points)),
                        'EMD': emd(np.asarray(source.points), np.asarray(target.points))}

        print("exp: ", method_dict[diam]['exp_path_list'][i])
        print("CD: ", dist_metrics['CD'])
        print("EMD: ", dist_metrics['EMD'])

        cd_list.append(dist_metrics['CD'])
        emd_list.append(dist_metrics['EMD'])


        # predict mean radius (i.e. radius of clay where ~97% of points lie within the circle)
        pcl_mins = np.min(exp_pcl, axis=0) 
        pcl_maxs = np.max(exp_pcl, axis=0) 
        minx = pcl_mins[0]
        maxx = pcl_maxs[0] 
        miny = pcl_mins[1]
        maxy = pcl_maxs[1]
        theta = np.linspace(0, 2 * np.pi, 100)
        pcl_xy = exp_pcl[:, :2]  # take only x and y components
        distances = pairwise_distances(pcl_xy, metric='euclidean')
        # find the radius such that 95% of points are within that radius
        radius = np.percentile(distances, 97.5) / 2.0
        # print("Diameter: ", 2*radius)
        # create green point cloud of points along the circle with that radius
        goal_x = (radius * np.cos(theta) + (maxx + minx) / 2) 
        goal_y = (radius * np.sin(theta) + (maxy + miny) / 2) 
        goal_z = np.zeros_like(goal_x) + 0.045
        circle_pcl = o3d.geometry.PointCloud()
        circle_pcl.points = o3d.utility.Vector3dVector(np.column_stack((goal_x, goal_y, goal_z)))
        circle_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,1,0]), (len(goal_x),1)))
        # o3d.visualization.draw_geometries([exp_pointcloud, goal_pcl])

        # get the g.t. diameter of the goal
        #  predict mean radius (i.e. radius of clay where ~97% of points lie within the circle)
        goal_pcl_mins = np.min(goal_pcl, axis=0) 
        goal_pcl_maxs = np.max(goal_pcl, axis=0) 
        minx = goal_pcl_mins[0]
        maxx = goal_pcl_maxs[0] 
        miny = goal_pcl_mins[1]
        maxy = goal_pcl_maxs[1]
        theta = np.linspace(0, 2 * np.pi, 100)
        pcl_xy = goal_pcl[:, :2]  # take only x and y components
        distances = pairwise_distances(pcl_xy, metric='euclidean')
        # find the radius such that 95% of points are within that radius
        actual_radius = np.percentile(distances, 97.5) / 2.0

        # calculate mean squared error between measured diameter and diam goal
        print("Diameter: ", 2 * radius * 100)
        print("Actual Diameter: ", 2  * actual_radius * 100)
        mse = mean_squared_error([2*radius*100], [2*actual_radius*100])
        mse_list.append(mse)
    print("\n\n\n---------DIAM: ", diam, " ----------")
    print("Mean CD: ", np.mean(cd_list))
    print("Std Dev CD: ", np.std(cd_list))
    print("Mean EMD: ", np.mean(emd_list))
    print("Std Dev EMD: ", np.std(emd_list))
    print("Mean MSE: ", np.mean(mse_list))
    print("Std Dev MSE: ", np.std(mse_list))
