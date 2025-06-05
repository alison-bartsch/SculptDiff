import cv2
import time
import numpy as np
import open3d as o3d
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from test_collision_checker import create_grippers

def create_lineset_from_point_pairs(p1, p2, color=[1, 1, 1]):
    '''
    Given the points p1 and p2 being the start and end of a line segment,
    create an Open3D LineSet object representing that line segment.
    Lineset has the attributes, points, lines, and colors.
    '''
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([p1, p2])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([color, color])  # Set color to white
    return line

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

def animate_point_cloud(traj_idx, traj_len, view='isometric', pltmap='viridis'):
    '''
    This function takes a point cloud and generates an animated gif with the point cloud
    rotating around the z-axis. The camera remains in a fixed observation pose.
    '''
    img_sequence = []
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    # set the point size
    ro = vis.get_render_option()
    ro.point_size = 10.0

    # set the camera view
    ctr = vis.get_view_control()
    base_path = "/home/alison/Documents/GitHub/subgoal_diffusion/open3d_configs"
    if view == 'isometric':
        path = base_path + "/isometric_view.json"
        print("Using isometric view.")
    elif view == 'side':
        path = base_path + "/side_on_view.json"
    elif view == 'top':
        path = base_path + "/top_down_view.json"
    else:
        raise ValueError("Invalid view type. Choose 'isometric', 'side', or 'top'.")
    print("Loading camera parameters from:", path)
    parameters = o3d.io.read_pinhole_camera_parameters(path)

    # add in the geometry
    np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud1.npy') # 22.npy')  # Load a sample point cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(np_pcl)

    action7d = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory' + str(traj_idx) + '/action7d_unnormalized0.npy')  # Load the corresponding action
    # add offsets
    action7d[0] += 0.03
    action7d[1] -= 0.025
    action7d[2] -= 0.04
    action7d[5] += 90
    # print("Action 7D ee dist:", action7d[6])
    cylinder1, cylinder2 = create_grippers(action7d) #, ee_dist=action7d[6])
    # create a line in open3d from one point into infinity
    line = create_lineset_from_point_pairs(pcl.points[0], pcl.points[0] + np.array([0, 0, 1]) * 1e6, color=[1, 1, 1])
    vis.add_geometry(pcl)
    vis.add_geometry(cylinder1)
    vis.add_geometry(cylinder2)
    vis.add_geometry(line)

    img = vis.capture_screen_float_buffer()
    img_sequence.append(img)

    ctr.convert_from_pinhole_camera_parameters(parameters, True)
    ctr.set_zoom(1.15)
    time.sleep(0.25)


    for i in range(1,20,10):
        np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(i) + '.npy') # 22.npy')  # Load a sample point cloud
        # pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(np_pcl)

        action7d = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory' + str(traj_idx) + '/action7d_unnormalized' + str(i-1) + '.npy')  # Load the corresponding action
        # add offsets
        action7d[0] += 0.03
        action7d[1] -= 0.025
        action7d[2] -= 0.04
        action7d[5] += 90
        # print("Action 7D ee dist:", action7d[6])
        c1, c2 = create_grippers(action7d) #, ee_dist=action7d[6])

        # iterate through the collision checker
        points = np.asarray(pcl.points)
        # downsample points to 250 
        if len(points) > 100:
            indices = np.random.choice(len(points), 100, replace=False)
            points = points[indices]
        c1_triangles = np.asarray(c1.triangles)
        c1_vertices = np.asarray(c1.vertices)
        c2_triangles = np.asarray(c2.triangles)
        c2_vertices = np.asarray(c2.vertices)
        collision_count = 0
        
        for point in points:
            random_direction = np.random.rand(3)  # Random direction vector
            p1 = point
            p2 = point + random_direction*1e6

            # get intersections
            ray_mesh_intersections = 0
            for tri in c1_triangles:
                v0, v1, v2 = c1_vertices[tri]
                if intersect_segment_triangle(p1, p2, v0, v1, v2):
                    ray_mesh_intersections += 1

            if ray_mesh_intersections % 2 == 1:  # odd number of intersections means point is inside the mesh
                collision_count += 1

                # create a line in open3d from one point into infinity
                new_line = create_lineset_from_point_pairs(p1, p2, color=[0, 0, 0])

                # update the gripper geometries
                cylinder1.vertices = c1.vertices
                cylinder1.triangles = c1.triangles
                cylinder1.paint_uniform_color([1, 0, 0])
                cylinder2.vertices = c2.vertices
                cylinder2.triangles = c2.triangles
                line.points = new_line.points
                line.colors = new_line.colors

                vis.update_geometry(pcl)
                vis.update_geometry(cylinder1)
                vis.update_geometry(cylinder2)
                vis.update_geometry(line)
                vis.poll_events()
                vis.update_renderer()

                # Capture the screen
                img = vis.capture_screen_float_buffer()
                for i in range(5):
                    img_sequence.append(img)

                time.sleep(0.025)

            
            else:
                # create a line in open3d from one point into infinity
                new_line = create_lineset_from_point_pairs(p1, p2, color=[0, 0, 0])

                # update the gripper geometries
                cylinder1.vertices = c1.vertices
                cylinder1.triangles = c1.triangles
                cylinder1.paint_uniform_color([0.8, 0.8, 0.8])
                cylinder2.vertices = c2.vertices
                cylinder2.triangles = c2.triangles
                line.points = new_line.points
                line.colors = new_line.colors

                vis.update_geometry(pcl)
                vis.update_geometry(cylinder1)
                vis.update_geometry(cylinder2)
                vis.update_geometry(line)
                vis.poll_events()
                vis.update_renderer()

                # Capture the screen
                img = vis.capture_screen_float_buffer()
                img_sequence.append(img)

                time.sleep(0.00025)


            # get intersections
            ray_mesh_intersections = 0
            for tri in c2_triangles:
                v0, v1, v2 = c2_vertices[tri]
                if intersect_segment_triangle(p1, p2, v0, v1, v2):
                    ray_mesh_intersections += 1

            if ray_mesh_intersections % 2 == 1:  # odd number of intersections means point is inside the mesh
                collision_count += 1

                # create a line in open3d from one point into infinity
                new_line = create_lineset_from_point_pairs(p1, p2, color=[0, 0, 0])

                # update the gripper geometries
                cylinder1.vertices = c1.vertices
                cylinder1.triangles = c1.triangles
                cylinder1.paint_uniform_color([0.8, 0.8, 0.8])
                cylinder2.vertices = c2.vertices
                cylinder2.triangles = c2.triangles
                cylinder2.paint_uniform_color([1, 0, 0])
                line.points = new_line.points
                line.colors = new_line.colors

                vis.update_geometry(pcl)
                vis.update_geometry(cylinder1)
                vis.update_geometry(cylinder2)
                vis.update_geometry(line)
                vis.poll_events()
                vis.update_renderer()

                # Capture the screen
                img = vis.capture_screen_float_buffer()
                for i in range(5):
                    img_sequence.append(img)

                time.sleep(0.025)

            else:
                # create a line in open3d from one point into infinity
                new_line = create_lineset_from_point_pairs(p1, p2, color=[0, 0, 0])

                # update the gripper geometries
                cylinder1.vertices = c1.vertices
                cylinder1.triangles = c1.triangles
                cylinder1.paint_uniform_color([0.8, 0.8, 0.8])
                cylinder2.vertices = c2.vertices
                cylinder2.triangles = c2.triangles
                cylinder2.paint_uniform_color([0.8, 0.8, 0.8])
                line.points = new_line.points
                line.colors = new_line.colors

                vis.update_geometry(pcl)
                vis.update_geometry(cylinder1)
                vis.update_geometry(cylinder2)
                vis.update_geometry(line)
                vis.poll_events()
                vis.update_renderer()

                # Capture the screen
                img = vis.capture_screen_float_buffer()
                img_sequence.append(img)

                time.sleep(0.00025)


        
    
    # close the visualizer
    vis.destroy_window()
    return img_sequence

def set_camera_to_orthographic(vis):
    # get current camra settings
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    # modify the intrinsic parameters to achieve orthographic projection
    param.intrinsic.set_intrinsics(
        width=1920,
        height=1080,
        fx=1.0,
        fy=1.0,
        cx=960,
        cy=540
    )
    ctr.convert_from_pinhole_camera_parameters(param)

def generate_colormap(pcl, pltmap='viridis'):
    '''
    This function takes a point cloud and generates a colormap based on the z-coordinate of each point.
    The colormap is then applied to the point cloud and visualized.
    '''
    # Normalize the z-coordinates to the range [0, 1]
    z = pcl[:, 2]
    z_min = np.min(z)
    z_max = np.max(z)
    z_normalized = (z - z_min) / (z_max - z_min)

    # Create a colormap
    colormap = plt.get_cmap(pltmap)
    colors = colormap(z_normalized)

    return colors[:, :3]  # Ignore the alpha channel

def vis_fov_point_cloud(pcl):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    # setr the point size
    ro = vis.get_render_option()
    ro.point_size = 10.0

    # set the camera view
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("/home/alison/Documents/GitHub/subgoal_diffusion/open3d_configs/isometric_view.json")

    # add in the geometry
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(pcl)
    geometry.colors = o3d.utility.Vector3dVector(generate_colormap(pcl))
    vis.add_geometry(geometry) #, reset_bounding_box=False)

    ctr.convert_from_pinhole_camera_parameters(parameters, True)

    vis.run()
    vis.destroy_window()

def make_video(img_sequence, filename='point_cloud_animation.mp4', fps=5):
    '''
    This function takes a list of images and creates a video from them.
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = np.array(img_sequence[0]).shape
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for img in img_sequence:
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        out.write(img)

    out.release()

def make_gif(img_sequence, filename='point_cloud_animation.gif', duration=100):
    '''
    This function takes a list of images and creates a gif from them.
    '''
    images = []
    for img in img_sequence:
        img = np.array(img)
        img = img[::-1, :, :]  # flip the colors from BGR to RGB <--- this is close, but flips the image upside down
        # flip the image upside down
        img = np.flipud(img)
        img = (img * 255).astype(np.uint8)
        images.append(Image.fromarray(img))
    images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

if __name__ == "__main__":
    traj_idx = 5 # 0, 1, 2, 3, 4, 5, 6
    traj_len = 20 # 64, 22, 33, 32, 22, 20, 33  # Number of frames in the trajectory
    
    img_list = animate_point_cloud(traj_idx, traj_len, view='isometric')
    print("Generated {} images for the animation.".format(len(img_list)))
    make_gif(img_list, filename='collision_traj2' + str(traj_idx) + '.gif', duration=50)
    # vis_fov_point_cloud(pcl)


    # # visualize final point cloud for all trajectories
    # traj_len = [64, 22, 33, 32, 22, 20, 33]  # Number of frames in each trajectory
    # for traj_idx in range(7):
    #     np_pcl = np.load('/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(traj_len[traj_idx]-1) + '.npy')  # Load a sample point cloud
    #     pcl = o3d.geometry.PointCloud()
    #     pcl.points = o3d.utility.Vector3dVector(np_pcl)
    #     o3d.visualization.draw_geometries([pcl])

