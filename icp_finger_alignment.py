import numpy as np
import open3d as o3d
from test_collision_checker import create_grippers

# load in the real-world point cloud of the fingers
finger_pcl = np.load('/home/alison/Documents/GitHub/subgoal_diffusion/real_world_data/finger_pcl.npy')

# load in the corresponding end effector action
action = np.load('/home/alison/Documents/GitHub/subgoal_diffusion/real_world_data/finger_action.npy')

# create the synthetic end-effector point cloud based on the action
finger_mesh = create_grippers(action, ee_dist=0.04, color=[0.8, 0.8, 0.8], cylinder_radius=0.004)
# Convert the mesh to a point cloud
finger_mesh_combined = finger_mesh[0] + finger_mesh[1]  # Combine the two gripper meshes
synthetic_finger_pcl = np.asarray(finger_mesh_combined.points)

# ICP alignment to get the transformation matric to transform the synthetic point cloud to the real-world point cloud
def icp_alignment(source, target, threshold=0.02, max_iterations=100):
    """
    Perform ICP alignment between source and target point clouds.
    
    :param source: Source point cloud (numpy array).
    :param target: Target point cloud (numpy array).
    :param threshold: Distance threshold for convergence.
    :param max_iterations: Maximum number of iterations.
    :return: Transformation matrix.
    """
    # for i in range(max_iterations):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)

    reg_icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
        # # Check if the registration converged
        # if reg_icp.inlier_rmse < threshold:
        #     print(f"ICP converged after {i+1} iterations.")
        #     return reg_icp.transformation
    return reg_icp.transformation

# Perform ICP alignment
transformation_matrix = icp_alignment(synthetic_finger_pcl, finger_pcl)

# Apply the transformation to the synthetic point cloud
new_synthetic_finger_pcl = o3d.geometry.PointCloud()
new_synthetic_finger_pcl.points = o3d.utility.Vector3dVector(
    np.dot(synthetic_finger_pcl, transformation_matrix[:3, :3].T) + transformation_matrix[:3, 3]
)
# Visualize the aligned point clouds
real_finger_pcd = o3d.geometry.PointCloud()
real_finger_pcd.points = o3d.utility.Vector3dVector(finger_pcl)
real_finger_pcd.paint_uniform_color([1, 0, 0])  # Red for real finger

synthetic_finger_pcd = o3d.geometry.PointCloud()
synthetic_finger_pcd.points = o3d.utility.Vector3dVector(new_synthetic_finger_pcl.points)
synthetic_finger_pcd.paint_uniform_color([0, 1, 0])  # Green for synthetic finger

o3d.visualization.draw_geometries([real_finger_pcd, synthetic_finger_pcd])