import time
import torch
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from pcl_utils import *
from vis_trajectory_with_collision import make_gif
from action_sequence_vis import vis_gripper_sequence
from test_collision_checker import check_finger_collision
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# parameters defined
ckpt_dir = '/home/alison/Documents/GitHub/SculptDiff/checkpoints/pottery_long_epochs_16pred_7datasetfixed_with_augs'
traj_goal_pairs = [(0, 64), (1, 22), (2, 33), (3, 32), (4, 22), (5, 24), (6, 33)]
traj_idx = 6 # (0, 64) (1, 22) (2, 33) (3, 32) (4, 22) (5, 24) (6, 33)
goal_idx = 33 
goal_path = '/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery/Trajectory' + str(traj_idx) + '/unnormalized_pointcloud' + str(goal_idx) + '.npy'
centered_action = False
pred_horizon = 16 
execute_horizon = 16 
collision_check = False
view = 'isometric'  # 'isometric', 'side', or 'top'

# define diffusion parameters
obs_horizon = 1
B = 1
action_dim = 8
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# define the device
device = torch.device('cuda')

# define the action space limits for unnormalization
if centered_action:
    a_mins7d = np.array([-0.15, -0.15, -0.05, -90, 0.005])
    a_maxs7d = np.array([0.15, 0.15, 0.05, 90, 0.05])
else:
    a_mins7d = np.array([0.5413, -0.04232, 0.1300, -45, -15, -90, 0.0005])
    a_maxs7d = np.array([0.6700, 0.08500, 0.1560, 45, 13, 90, 0.005])

qpos = np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04])
qpos = (qpos - a_mins7d) / (a_maxs7d - a_mins7d)
qpos = qpos * 2.0 - 1.0
qpos = np.concatenate((qpos, np.array([-1.])), axis=0)
nagent_pos = torch.from_numpy(qpos).to(torch.float32).unsqueeze(axis=0).unsqueeze(axis=0).to(device)

# initialize the pointbert model
testconfig = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
testmodel_config = testconfig.model
pointbert = builder.model_builder(testmodel_config)
testweights_path = ckpt_dir + '/pointbert_statedict' 
pointbert.load_state_dict(torch.load(testweights_path))
pointbert.to(device)

# load projection head from ckpt_dir
enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint', map_location=torch.device('cpu')) 
projection_head = enc_checkpoint['encoder_head'].to(device)

# load noise_pred_net from ckpt_dir
noise_checkpoint = torch.load(ckpt_dir + '/noise_pred_best_checkpoint', map_location=torch.device('cpu')) 
noise_pred_net = noise_checkpoint['noise_pred_net'].to(device)

# load in the goal
raw_goal = np.load(goal_path)


# iterate through the observations from the goal trajectory with a step size of execute_horizon
for i in range(0,goal_idx, execute_horizon):
    # load in the goal observation point cloud
    unnorm_pcl = np.load('/home/alison/Documents/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory0/unnormalized_pointcloud' + str(i) + '.npy')
    ctr = np.mean(raw_goal, axis=0)
    pointcloud = (unnorm_pcl.copy() - ctr) * 10
    numpy_goal = (raw_goal.copy() - ctr) * 10.0

    B = 1
    with torch.inference_mode():
        # pass the point cloud through Point-BERT to get the latent representation
        state = torch.from_numpy(pointcloud).to(torch.float32)
        states = torch.unsqueeze(state, 0).to(device)
        tokenized_states = pointbert(states)
        pcl_embed = projection_head(tokenized_states)
        pointcloud_features = pcl_embed.unsqueeze(1).repeat(1, obs_horizon, 1)

        # pass the goal cloud through Point-BERT and projection head
        goal = numpy_goal.copy()
        goal = torch.from_numpy(goal).to(torch.float32)
        goals = torch.unsqueeze(goal, 0).to(device)
        tokenized_goals = pointbert(goals)
        goal_embed = projection_head(tokenized_goals)
        goalcloud_features = goal_embed.unsqueeze(1).repeat(1, obs_horizon, 1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([pointcloud_features, nagent_pos, goalcloud_features],dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, pred_horizon, action_dim), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            
            # predict noise
            noise_pred = noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

            # intermediate actions unnormalize
            intermediate_pred_action = naction.detach().to('cpu').numpy()[0]
            intermediate_action_pred = (intermediate_pred_action[:,0:7] + 1.0) / 2.0
            intermediate_action_pred = intermediate_action_pred * (a_maxs7d - a_mins7d) + a_mins7d

            # visualize the intermediate action sequence
            
            
            # o3d.visualization.draw_geometries([pointcloud] + gripper_list)

            if k == 99 and i == 0:
                # initialize the visualizer
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

                gripper_list = vis_gripper_sequence(intermediate_action_pred)

                o3d_pointcloud = o3d.geometry.PointCloud()
                o3d_pointcloud.points = o3d.utility.Vector3dVector(unnorm_pcl)

                # assign colors to each point in the point cloud based on the z-height
                # specifically the binary colormap in matplotlib
                z_heights = unnorm_pcl[:, 2]
                z_heights = (z_heights - np.min(z_heights)) / (np.max(z_heights) - np.min(z_heights))
                colormap = plt.get_cmap('binary')
                colors = colormap(z_heights)[:, :3]  # Get RGB values
                o3d_pointcloud.colors = o3d.utility.Vector3dVector(colors)


                vis.add_geometry(o3d_pointcloud)
                
                for elem in gripper_list:
                    vis.add_geometry(elem)

                ctr.convert_from_pinhole_camera_parameters(parameters, True)
                ctr.set_zoom(1.15)

                # Capture the screen
                img = vis.capture_screen_float_buffer()
                
                time.sleep(0.025)
            else:
                # update the geometries to modify
                o3d_pointcloud.points = o3d.utility.Vector3dVector(unnorm_pcl)

                z_heights = unnorm_pcl[:, 2]
                z_heights = (z_heights - np.min(z_heights)) / (np.max(z_heights) - np.min(z_heights))
                colormap = plt.get_cmap('binary')
                colors = colormap(z_heights)[:, :3]  # Get RGB values
                o3d_pointcloud.colors = o3d.utility.Vector3dVector(colors)

                new_gripper_list = vis_gripper_sequence(intermediate_action_pred)

                vis.update_geometry(o3d_pointcloud)
                for i in range(len(gripper_list)):
                    elem = gripper_list[i]
                    elem.vertices = new_gripper_list[i].vertices
                    elem.triangles = new_gripper_list[i].triangles
                    # cylinder2.vertices = c2.vertices
                    # cylinder2.triangles = c2.triangles
                    vis.update_geometry(elem)
                vis.poll_events()
                vis.update_renderer()

                # Capture the screen
                img = vis.capture_screen_float_buffer()
                img_sequence.append(img)

                if k == 0:
                    for _ in range(10):
                        img_sequence.append(img)

                time.sleep(0.025)

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()


    time.sleep(0.5)

    # update nagent pos
    pred_action = naction[0]
    termination_pred = pred_action[:,7]
    action_pred = (pred_action[:,0:7] + 1.0) / 2.0
    action_pred = action_pred * (a_maxs7d - a_mins7d) + a_mins7d
    for j in range(execute_horizon):
        unnorm_a = action_pred[j,:]
        terminate = termination_pred[j]

        # update nagent_pos to be the new position
        nagent_pos = torch.from_numpy(pred_action[j]).to(torch.float32).unsqueeze(axis=0).unsqueeze(axis=0).to(device)

# close the visualizer
vis.destroy_window()

# save the images to a gifs
print("Generated {} images for the animation.".format(len(img_sequence)))
make_gif(img_sequence, filename='gifs/sculptdiff_denoising_traj' + str(traj_idx) + '.gif', duration=100)