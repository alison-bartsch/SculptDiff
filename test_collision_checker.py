import numpy as np
import open3d as o3d

def check_finger_collision(action7d, pcl, vis=False):
    '''
    Initialize the default gripper fingers mesh of two cylinders 5mm in diameter and 45 mm in 
    length with a 40mm gap between them. This function creates the gripper mesh, and then 
    applies the translational and rotation transformations to the mesh based on the action7d input.
    '''
    # Create two cylinders for the gripper fingers
    cylinder_radius = 0.006 #4 # 4  # 8mm diameter
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
        # # create a red point cloud that is a line in the x direction
        # line_points = np.zeros((100, 3))
        # line_points[:, 0] = np.linspace(-2, 2, 100)  # x-axis line from -0.02 to 0.02
        # line_pcl = o3d.geometry.PointCloud()
        # line_pcl.points = o3d.utility.Vector3dVector(line_points)
        # line_pcl.paint_uniform_color([1, 0, 0])  # Set color to red
        # # create a line in open3d along the y-axis
        # line_points_y = np.zeros((100, 3))
        # line_points_y[:, 1] = np.linspace(-2, 2, 100)  # y-axis line from -0.02 to 0.02
        # line_pcl_y = o3d.geometry.PointCloud()
        # line_pcl_y.points = o3d.utility.Vector3dVector(line_points_y)
        # line_pcl_y.paint_uniform_color([0, 1, 0])  # Set color to green
        # o3d.visualization.draw_geometries([cylinder1, cylinder2, pcl, line_pcl, line_pcl_y])
        o3d.visualization.draw_geometries([cylinder1, cylinder2, pcl])

    # check if the fingers collide with the point cloud
    c1_collision_count = check_collision(pcl, cylinder1)
    c2_collision_count = check_collision(pcl, cylinder2)

    collision_count = c1_collision_count + c2_collision_count

    # return True if either finger collides with the point cloud
    if collision_count > 10:  # threshold for collision detection
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
    # collision = np.any(np.all((points >= bounding_box.min_bound) & (points <= bounding_box.max_bound), axis=1))
    collision_count = np.sum(np.all((points >= bounding_box.min_bound) & (points <= bounding_box.max_bound), axis=1))
    return collision_count
    
if __name__ == "__main__":
    # Example action7d input
    action7d = np.array([0.6, 0.0, 0.1, 40.0, 20.0, 45.0, 0.001])  # Example action7d input
    np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory4/unnormalized_pointcloud22.npy') # 22.npy')  # Load a sample point cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(np_pcl)

    # create and visualize the gripper mesh
    collision = check_finger_collision(action7d, pcl, vis=True)
    if collision:
        print("Collision detected!")
    else:
        print("No collision detected.")

    for i in range(1,60):
        np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory0/unnormalized_pointcloud' + str(i) + '.npy') # 22.npy')  # Load a sample point cloud
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(np_pcl)

        action7d = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory0/action7d_unnormalized' + str(i-1) + '.npy')  # Load the corresponding action
        # add 0.03 offset to the x coordinate of the action
        action7d[0] += 0.025 # 0.025
        # add 0.01 offset to the y coordinate of the action
        action7d[1] -= 0.02 
        # change the z coordinate of the action to be - 0.045
        action7d[2] -= 0.04
        # change the rotation about the z-axis to add 90 degrees
        action7d[5] += 90
        collision = check_finger_collision(action7d, pcl, vis=True)
        if collision:
            print(f"Collision detected for point cloud {i-1}!")
        else:
            print(f"No collision detected for point cloud {i-1}.")