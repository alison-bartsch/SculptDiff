import numpy as np
import open3d as o3d
from tqdm import tqdm

def create_grippers(action7d, ee_dist=0.05):
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
    # cylinder2.translate([0.04, 0, 0])  # Move the second cylinder to the right
    cylinder2.translate([ee_dist, 0, 0])  # Move the second cylinder to the right

    # Apply the action7d transformations
    translation = action7d[:3]  # x, y, z translation
    rotation = action7d[3:6]  # roll, pitch, yaw in degrees
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(rotation))

    cylinder1.rotate(rotation_matrix, center=(0, 0, 0))  # Apply rotation around the origin
    cylinder1.translate(translation)  # Apply translation
    cylinder2.rotate(rotation_matrix, center=(0, 0, 0))  # Apply rotation around the origin
    cylinder2.translate(translation)  # Apply translation
    return cylinder1, cylinder2

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
    # cylinder2.translate([0.04, 0, 0])  # Move the second cylinder to the right
    cylinder2.translate([0.05, 0, 0])  # Move the second cylinder to the right

    # make scaling adjustments
    action7d[0] += 0.03
    action7d[1] -= 0.025
    action7d[2] -= 0.04
    action7d[5] += 90

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
        # create a red point cloud that is a line in the x direction
        line_points = np.zeros((100, 3))
        line_points[:, 0] = np.linspace(0.6, 2, 100)  # x-axis line from -0.02 to 0.02
        line_pcl = o3d.geometry.PointCloud()
        line_pcl.points = o3d.utility.Vector3dVector(line_points)
        line_pcl.paint_uniform_color([1, 0, 0])  # Set color to red
        # create a line in open3d along the y-axis
        line_points_y = np.zeros((100, 3))
        line_points_y[:, 1] = np.linspace(0, 2, 100)  # y-axis line from -0.02 to 0.02
        line_points_y[:, 0] = 0.6*np.ones(100)
        line_pcl_y = o3d.geometry.PointCloud()
        line_pcl_y.points = o3d.utility.Vector3dVector(line_points_y)
        line_pcl_y.paint_uniform_color([0, 1, 0])  # Set color to green
        o3d.visualization.draw_geometries([cylinder1, cylinder2, pcl, line_pcl, line_pcl_y])


        # o3d.visualization.draw_geometries([cylinder1, cylinder2, pcl])

    # check size of the point cloud, if more than 250 points, downsample it to 250 points
    if len(pcl.points) > 250:
        pcl.points = o3d.utility.Vector3dVector(np.asarray(pcl.points)[np.random.choice(len(pcl.points), 250, replace=False)])

    # check if the fingers collide with the point cloud
    c1_collision_count = check_collision(pcl, cylinder1)
    c2_collision_count = check_collision(pcl, cylinder2)

    collision_count = c1_collision_count + c2_collision_count

    # return True if either finger collides with the point cloud
    if collision_count > 5:  # threshold for collision detection
        print("Collision count: ", collision_count)
        return True
    else:
        return False

def check_bbox_collision(pcl, cylinder):
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

def check_collision_new_o3d_version(pcl, cylinder, epsilon=0.001):
    '''
    Check for collision between the cylinder and the point cloud.
    A collision is defined if any point in the point cloud lies within
    the gripper mesh. Returns True if there is a collision, False otherwise.
    '''
    # point-mesh epsilon intersection test
    # for all points perform a point-triangle distance check against all triangles in the cylinder mesh
    points = np.asarray(pcl.points)
    triangles = np.asarray(cylinder.triangles)
    vertices = np.asarray(cylinder.vertices)
    collision_count = 0
    
    for point in points:
        random_direction = np.random.rand(3)  # Random direction vector
        p1 = point
        p2 = point + random_direction*1e6
        dir = p2 - p1
        length = np.linalg.norm(dir)
        dir /= length  # Normalize the direction vector

        cylinder.compute_triangle_normals()  # Ensure normals are computed for the mesh

        scene = o3d.geometry.RaycastingScene()
        mesh_id = scene.add_triangles(cylinder)  # Add the cylinder mesh to the scene
        query = scene.cast_rays(o3d.core.Tensor([{'origin': p1, 'direction': dir}]))  # Cast a ray from the point in the direction
        intersections = scene.compute_scene_intersection(p1, dir, max_distance=length)  # Compute intersections with the mesh
        ray_mesh_intersections = intersections.shape[0] if intersections is not None else 0

        if len(ray_mesh_intersections) % 2 == 1:  # odd number of intersections means point is inside the mesh
            collision_count += 1
    return collision_count

def intersect_segment_triangle(p0, p1, v0, v1, v2):
    EPSILON = 1e-8
    dir = p1 - p0
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(dir, edge2)
    a = np.dot(edge1, h)
    if -EPSILON < a < EPSILON:
        return False  # Ray is parallel to triangle
    f = 1.0 / a
    s = p0 - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * np.dot(dir, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(edge2, q)
    if t < 0.0 or t > 1.0:
        return False  # Outside the segment
    return True  # Intersection found


def check_collision(pcl, cylinder, epsilon=0.001):
    '''
    Check for collision between the cylinder and the point cloud.
    A collision is defined if any point in the point cloud lies within
    the gripper mesh. Returns True if there is a collision, False otherwise.
    '''
    # point-mesh epsilon intersection test
    # for all points perform a point-triangle distance check against all triangles in the cylinder mesh
    points = np.asarray(pcl.points)
    triangles = np.asarray(cylinder.triangles)
    vertices = np.asarray(cylinder.vertices)
    collision_count = 0
    
    for point in tqdm(points):
        random_direction = np.random.rand(3)  # Random direction vector
        p1 = point
        p2 = point + random_direction*1e6

        # get intersections
        ray_mesh_intersections = 0
        for tri in triangles:
            v0, v1, v2 = vertices[tri]
            if intersect_segment_triangle(p1, p2, v0, v1, v2):
                ray_mesh_intersections += 1

        if ray_mesh_intersections % 2 == 1:  # odd number of intersections means point is inside the mesh
            collision_count += 1
    return collision_count
    
    
if __name__ == "__main__":
    # # Example action7d input
    # action7d = np.array([0.6, 0.0, 0.1, 40.0, 20.0, 45.0, 0.001])  # Example action7d input
    # np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory4/unnormalized_pointcloud22.npy') # 22.npy')  # Load a sample point cloud
    # pcl = o3d.geometry.PointCloud()
    # pcl.points = o3d.utility.Vector3dVector(np_pcl)

    # # create and visualize the gripper mesh
    # collision = check_finger_collision(action7d, pcl, vis=True)
    # if collision:
    #     print("Collision detected!")
    # else:
    #     print("No collision detected.")

    # np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory0/unnormalized_pointcloud2.npy') # 22.npy')  # Load a sample point cloud
    # pcl = o3d.geometry.PointCloud()
    # pcl.points = o3d.utility.Vector3dVector(np_pcl)

    # action7d = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory0/action7d_unnormalized1.npy')  # Load the corresponding action
    # action7d[2] -= 0.04
    # action7d[5] += 90
    # collision = check_finger_collision(action7d, pcl, vis=True)

    # a2 = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory0/action7d_unnormalized1.npy')  # Load the corresponding action
    # a2[0] -= 0.05
    # a2[1] -= 0.025
    # a2[2] -= 0.04
    # # a2[5] += 90
    # collision = check_finger_collision(a2, pcl, vis=True)

    # a3 = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory0/action7d_unnormalized1.npy')  # Load the corresponding action
    # a3[0] += 0.03
    # a3[1] += 0.025
    # a3[2] -= 0.04
    # a3[5] += 90
    # collision = check_finger_collision(a3, pcl, vis=True)


    for i in range(1,60):
        np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory0/unnormalized_pointcloud' + str(i) + '.npy') # 22.npy')  # Load a sample point cloud
        # downsample the point cloud to 1000 points
        np_pcl = np_pcl[np.random.choice(np_pcl.shape[0], 250, replace=False), :]
        
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(np_pcl)

        action7d = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory0/action7d_unnormalized' + str(i-1) + '.npy')  # Load the corresponding action
    

        # add offsets
        action7d[0] += 0.03
        action7d[1] -= 0.025
        action7d[2] -= 0.04
        action7d[5] += 90

        collision = check_finger_collision(action7d, pcl, vis=True)
        if collision:
            print(f"Collision detected for point cloud {i-1}!")
        else:
            print(f"No collision detected for point cloud {i-1}.")