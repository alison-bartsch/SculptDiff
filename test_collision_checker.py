import numpy as np
import open3d as o3d

def check_finger_collision(action7d, pcl, vis=False):
    '''
    Initialize the default gripper fingers mesh of two cylinders 5mm in diameter and 45 mm in 
    length with a 40mm gap between them. This function creates the gripper mesh, and then 
    applies the translational and rotation transformations to the mesh based on the action7d input.
    '''
    # Create two cylinders for the gripper fingers
    cylinder_radius = 0.004 # 4  # 8mm diameter
    cylinder_length = 0.045  # 45mm length
    cylinder1 = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cylinder_length)
    cylinder1.compute_vertex_normals()
    cylinder1.paint_uniform_color([0.8, 0.8, 0.8])  # Set color to light gray
    cylinder1.translate([-cylinder_length / 2, 0, 0])  # Center the cylinder at the origin
    cylinder1.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi / 2)))  # Rotate to align with the y-axis
    cylinder2 = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cylinder_length)
    cylinder2.compute_vertex_normals()
    cylinder2.paint_uniform_color([0.8, 0.8, 0.8])  # Set color to light gray
    cylinder2.translate([-cylinder_length / 2, 0, 0])  # Center the cylinder at the origin
    cylinder2.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi / 2)))  # Rotate to align with the y-axis
    cylinder2.translate([0.04, 0, 0])  # Move the second cylinder to the right

    # visualize the gripper mesh
    if vis:
        o3d.visualization.draw_geometries([cylinder1, cylinder2, pcl])

    # Apply the action7d transformations
    translation = action7d[:3]  # x, y, z translation
    rotation = action7d[3:6]  # roll, pitch, yaw in degrees
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(rotation))

    cylinder1.rotate(rotation_matrix, center=(0, 0, 0))  # Apply rotation around the origin
    cylinder1.translate(translation)  # Apply translation
    cylinder2.rotate(rotation_matrix, center=(0, 0, 0))  # Apply rotation around the origin
    cylinder2.translate(translation)  # Apply translation

    # visualize the transformed gripper mesh
    if vis:
        o3d.visualization.draw_geometries([cylinder1, cylinder2, pcl])

    # check if the fingers collide with the point cloud
    c1_collision = check_collision(pcl, cylinder1)
    c2_collision = check_collision(pcl, cylinder2)

    # return True if either finger collides with the point cloud
    if c1_collision or c2_collision:
        return True
    else:
        return False

def check_collision(pcl, cylinder):
    '''
    Check for collision between the cylinder and the point cloud.
    A collision is defined if any point in the point cloud lies within
    the gripper mesh. Returns True if there is a collision, False otherwise.
    '''
    # Create a bounding box around the cylinder
    bounding_box = cylinder.get_axis_aligned_bounding_box()
    # Check if any point in the point cloud is within the bounding box
    points = np.asarray(pcl.points)
    collision = np.any(np.all((points >= bounding_box.min_bound) & (points <= bounding_box.max_bound), axis=1))
    return collision
    
if __name__ == "__main__":
    # Example action7d input
    action7d = np.array([0.6, 0.0, 0.1, 0.0, 0.0, 0.0, 0.001])  # Example action7d input
    np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory4/unnormalized_pointcloud22.npy') # 22.npy')  # Load a sample point cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(np_pcl)

    # create and visualize the gripper mesh
    collision = check_finger_collision(action7d, pcl, vis=True)
    if collision:
        print("Collision detected!")
    else:
        print("No collision detected.")
