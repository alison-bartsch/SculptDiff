import math
import copy
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import robomail.vision as vis
from frankapy import FrankaArm
from pcl_utils import *

pcl_center = np.array([0.630, -0.0054, 0.074])
ee_center = np.array([0.608, 0.014, 0.125])

# initialize the robot and reset joints
fa = FrankaArm()
# fa.reset_joints()
# fa.open_gripper()
# fa.goto_gripper(0.04)

# initialize the cameras
cam1 = vis.CameraClass(1)
cam2 = vis.CameraClass(2)
cam3 = vis.CameraClass(3)
cam4 = vis.CameraClass(4)
cam5 = vis.CameraClass(5) 

# initialize the 3D vision code
pcl_vis = vis.Vision3D()    

# define observation pose
observation_pose = fa.get_pose()
observation_translation = np.array([0.625, 0, 0.325]) # np.array([0.6, 0, 0.325])
observation_pose.translation = observation_translation
# fa.goto_pose(observation_pose)

# define initial joint rotation
joints = fa.get_joints()
ee_joint_pos = joints[6]


# get the observation state
rgb1, _, pc1, _ = cam1._get_next_frame()
rgb2, _, pc2, _ = cam2._get_next_frame()
rgb3, _, pc3, _ = cam3._get_next_frame()
rgb4, _, pc4, _ = cam4._get_next_frame()
rgb5, _, pc5, _ = cam5._get_next_frame()

# unnorm_pcl, ctr = pcl_vis.unnormalize_fuse_point_clouds_no_base(pc2, pc3, pc4, pc5, color="Orange")

# for 5x cameras, we need to get the ee pose
cur_pose = fa.get_pose()
translation = cur_pose.translation
rotation = cur_pose.rotation
_, _, _, _, _, unnorm_pcl, ctr = pcl_vis.crop_point_clouds_separately(pc1, pc2, pc3, pc4, pc5, color="Orange", ee_pos=translation, ee_rot=rotation, icp=True)
            
# center and scale pointcloud
pointcloud = (np.copy(unnorm_pcl) - pcl_center) * 10

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(pointcloud)
pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(pointcloud),1)))


# get the min and max x and y components of the pointcloud
pcl_copy = copy.deepcopy(pointcloud)
pcl_copy = pcl_copy / 10.0
pcl_copy = pcl_copy - pcl_center
pcl_mins = np.min(pcl_copy, axis=0) 
pcl_maxs = np.max(pcl_copy, axis=0) 
minx = pcl_mins[0]
maxx = pcl_maxs[0] 
miny = pcl_mins[1]
maxy = pcl_maxs[1]

# NOTE: if this is not a reliable way to find good radius constraint (i.e. too much noise)
# then instead project all points into x,y plane and do a few optimization steps to find
# best circle fit to minimize radius, but fit most ~95% of points inside

# get the mean radius constraint
r = np.mean([maxx-minx, maxy-miny]) / 2.0 # - 0.03
print("\nR: ", r)

# create red point cloud of points along the circle with that radius
theta = np.linspace(0, 2 * np.pi, 100)
goal_x = ((r * np.cos(theta) + (maxx + minx) / 2) + pcl_center[0]) * 10.0
goal_y = ((r * np.sin(theta) + (maxy + miny) / 2) + pcl_center[1]) * 10.0
goal_z = np.zeros_like(goal_x) + (pcl_center[2] - 0.03) # slightly below the center of the point cloud
goal_pcl = o3d.geometry.PointCloud()
goal_pcl.points = o3d.utility.Vector3dVector(np.column_stack((goal_x, goal_y, goal_z)))
goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(goal_x),1)))

o3d.visualization.draw_geometries([pcl, goal_pcl])


# alternatively, project all points in pointcloud into the x,y plane and fit a circle to those points
# project points into x,y plane
pcl_xy = pcl_copy[:, :2]  # take only x and y components
from sklearn.metrics import pairwise_distances
# calculate pairwise distances
distances = pairwise_distances(pcl_xy, metric='euclidean')
# find the radius such that 95% of points are within that radius
radius = np.percentile(distances, 95) / 2.0
print("Radius constraint: ", radius)
# create green point cloud of points along the circle with that radius
goal_x = ((radius * np.cos(theta) + (maxx + minx) / 2) + pcl_center[0]) * 10.0
goal_y = ((radius * np.sin(theta) + (maxy + miny) / 2) + pcl_center[1]) * 10.0
goal_z = np.zeros_like(goal_x) + (pcl_center[2] - 0.03) # slightly below the center of the point cloud
goal_pcl = o3d.geometry.PointCloud()
goal_pcl.points = o3d.utility.Vector3dVector(np.column_stack((goal_x, goal_y, goal_z)))
goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,1,0]), (len(goal_x),1)))
o3d.visualization.draw_geometries([pcl, goal_pcl])

# instead of using a circle constraint, we can use an ellipse constraint
# create ellipse constraint
ellipse_a = ((maxx - minx) / 2.0) - 0.007
ellipse_b = ((maxy - miny) / 2.0) - 0.007
print("Rx: ", ellipse_a)
print("Ry: ", ellipse_b)
# create red point cloud of points along the ellipse with that radius
goal_x = ((ellipse_a * np.cos(theta) + (maxx + minx) / 2) + pcl_center[0]) * 10.0
goal_y = ((ellipse_b * np.sin(theta) + (maxy + miny) / 2) + pcl_center[1]) * 10.0
goal_z = np.zeros_like(goal_x) + (pcl_center[2] - 0.03) # slightly below the center of the point cloud
goal_pcl = o3d.geometry.PointCloud()
goal_pcl.points = o3d.utility.Vector3dVector(np.column_stack((goal_x, goal_y, goal_z)))
goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(goal_x),1)))
o3d.visualization.draw_geometries([pcl, goal_pcl])