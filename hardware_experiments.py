import os
import cv2
import time
import torch
import queue
import threading
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import robomail.vision as vis
from frankapy import FrankaArm
from pcl_utils import *
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def goto_grasp(fa, x, y, z, rx, ry, rz, d):
	"""
	Parameterize a grasp action by the position [x,y,z] Euler angle rotation [rx,ry,rz], and width [d] of the gripper.
	This function was designed to be used for clay moulding, but in practice can be applied to any task.

	:param fa:  franka robot class instantiation
	"""
	pose = fa.get_pose()
	starting_rot = pose.rotation
	orig = Rotation.from_matrix(starting_rot)
	orig_euler = orig.as_euler('xyz', degrees=True)
	rot_vec = np.array([rx, ry, rz])
	new_euler = orig_euler + rot_vec
	r = Rotation.from_euler('xyz', new_euler, degrees=True)
	pose.rotation = r.as_matrix()
	pose.translation = np.array([x, y, z])

	fa.goto_pose(pose)
	fa.goto_gripper(d, force=60.0)
	time.sleep(3)

def experiment_loop(fa, cam2, cam3, cam4, cam5, pcl_vis, save_path, goal_str, ckpt_dir, done_queue, centered_action):
    '''
    '''
    # define diffusion parameters
    obs_horizon = 1
    B = 1
    pred_horizon = 4 
    action_dim = 6
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
        a_mins5d = np.array([-0.15, -0.15, -0.05, -90, 0.005])
        a_maxs5d = np.array([0.15, 0.15, 0.05, 90, 0.05])
    else:
        a_mins5d = np.array([0.56, -0.062, 0.125, -90, 0.005])
        a_maxs5d = np.array([0.7, 0.062, 0.165, 90, 0.05])

    qpos = np.array([0.6, 0.0, 0.165, 0.0, 0.05])
    qpos = (qpos - a_mins5d) / (a_maxs5d - a_mins5d)
    qpos = qpos * 2.0 - 1.0
    qpos = np.concatenate((qpos, np.array([-1.])), axis=0)
    nagent_pos = torch.from_numpy(qpos).to(torch.float32).unsqueeze(axis=0).unsqueeze(axis=0).to(device)

    # initialize the pointbert model
    testconfig = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
    testmodel_config = testconfig.model
    pointbert = builder.model_builder(testmodel_config)
    testweights_path = ckpt_dir + '/pointbert_statedict.zip' 
    pointbert.load_state_dict(torch.load(testweights_path))
    pointbert.to(device)

    # load projection head from ckpt_dir
    enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint.zip', map_location=torch.device('cpu')) 
    projection_head = enc_checkpoint['encoder_head'].to(device)

    # load noise_pred_net from ckpt_dir
    noise_checkpoint = torch.load(ckpt_dir + '/noise_pred_best_checkpoint.zip', map_location=torch.device('cpu')) 
    noise_pred_net = noise_checkpoint['noise_pred_net'].to(device)

    # load in the goal
    raw_goal = np.load('goals/' + goal_str + '.npy')

    # define observation pose
    pose = fa.get_pose()
    observation_pose = np.array([0.6, 0, 0.325])
    pose.translation = observation_pose
    fa.goto_pose(pose)
    
    # initialize the n_actions counter
    n_action = 0

    # establish the list tracking how long the system takes to plan
    planning_time_list = []

    # get the observation state
    rgb2, _, pc2, _ = cam2._get_next_frame()
    rgb3, _, pc3, _ = cam3._get_next_frame()
    rgb4, _, pc4, _ = cam4._get_next_frame()
    rgb5, _, pc5, _ = cam5._get_next_frame()

    unnorm_pcl, ctr = pcl_vis.unnormalize_fuse_point_clouds(pc2, pc3, pc4, pc5)
    # center and scale pointcloud
    pointcloud = (unnorm_pcl - ctr) * 10

    # save the point clouds from each camera
    o3d.io.write_point_cloud(save_path + '/cam2_pcl0.ply', pc2)
    o3d.io.write_point_cloud(save_path + '/cam3_pcl0.ply', pc3)
    o3d.io.write_point_cloud(save_path + '/cam4_pcl0.ply', pc4)
    o3d.io.write_point_cloud(save_path + '/cam5_pcl0.ply', pc5)

    # center the goal based on the goal center
    numpy_goal = (raw_goal - ctr) * 10.0
    # scale distance metric goal differently 
    dist_goal = numpy_goal.copy()

    # visualize observation vs goal cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pointcloud)
    pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(pointcloud),1)))
    goal_pcl = o3d.geometry.PointCloud()
    goal_pcl.points = o3d.utility.Vector3dVector(dist_goal)
    goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(dist_goal),1)))
    o3d.visualization.draw_geometries([pcl, goal_pcl])

    # save observation
    np.save(save_path + '/pcl0.npy', pointcloud)
    np.save(save_path + '/center0.npy', ctr)
    cv2.imwrite(save_path + '/rgb2_state0.jpg', rgb2)
    cv2.imwrite(save_path + '/rgb3_state0.jpg', rgb3)
    cv2.imwrite(save_path + '/rgb4_state0.jpg', rgb4)
    cv2.imwrite(save_path + '/rgb5_state0.jpg', rgb5)

    # get the distance metrics between the point cloud and goal
    dist_metrics = {'CD': chamfer(unnorm_pcl, raw_goal),
                    'EMD': emd(unnorm_pcl, raw_goal),
                    'HAUSDORFF': hausdorff(unnorm_pcl, raw_goal)}

    print("\nDists: ", dist_metrics)
    with open(save_path + '/dist_metrics_0.txt', 'w') as f:
        f.write(str(dist_metrics))

    in_progress = True
    while in_progress:
        with torch.inference_mode():
            start = time.time()
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

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        end = time.time()
        planning_time_list.append(end-start)

        # execute 4 actions before replanning
        pred_action = naction[0]
        termination_pred = pred_action[:,5]
        action_pred = (pred_action[:,0:5] + 1.0) / 2.0
        action_pred = action_pred * (a_maxs5d - a_mins5d) + a_mins5d
        
        for j in range(action_pred.shape[0]):
            unnorm_a = action_pred[j,:]
            terminate = termination_pred[j]

            if centered_action:
                unnorm_a[0:3] = unnorm_a[0:3] + ctr
            
            goto_grasp(fa, unnorm_a[0], unnorm_a[1], unnorm_a[2], 0, 0, unnorm_a[3], unnorm_a[4])
            n_action+=1

            # wait here
            time.sleep(3)

            # open the gripper
            fa.open_gripper(block=True)

            # move to observation pose
            pose.translation = observation_pose
            fa.goto_pose(pose)

            # get the observation state
            rgb2, _, pc2, _ = cam2._get_next_frame()
            rgb3, _, pc3, _ = cam3._get_next_frame()
            rgb4, _, pc4, _ = cam4._get_next_frame()
            rgb5, _, pc5, _ = cam5._get_next_frame()
            unnorm_pcl, ctr = pcl_vis.unnormalize_fuse_point_clouds(pc2, pc3, pc4, pc5)
            # center and scale pointcloud
            pointcloud = (unnorm_pcl - ctr) * 10

            # save the point clouds from each camera
            o3d.io.write_point_cloud(save_path + '/cam2_pcl' + str(i*4 + j + 1) + '.ply', pc2)
            o3d.io.write_point_cloud(save_path + '/cam3_pcl' + str(i*4 + j + 1) + '.ply', pc3)
            o3d.io.write_point_cloud(save_path + '/cam4_pcl' + str(i*4 + j + 1) + '.ply', pc4)
            o3d.io.write_point_cloud(save_path + '/cam5_pcl' + str(i*4 + j + 1) + '.ply', pc5)

            # center the goal based on the point cloud center
            numpy_goal = (raw_goal - ctr) * 10.0
            # scale distance metric goal differently 
            dist_goal = numpy_goal.copy()

            # visualize observation vs goal cloud
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(pointcloud)
            pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(pointcloud),1)))
            goal_pcl = o3d.geometry.PointCloud()
            goal_pcl.points = o3d.utility.Vector3dVector(dist_goal)
            goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(dist_goal),1)))
            o3d.visualization.draw_geometries([pcl, goal_pcl])

            # save observation
            np.save(save_path + '/pcl' + str(i*4 + j + 1) + '.npy', pointcloud)
            np.save(save_path + '/center' + str(i*4 + j + 1) + '.npy', ctr)
            cv2.imwrite(save_path + '/rgb2_state' + str(i*4 + j + 1) + '.jpg', rgb2)
            cv2.imwrite(save_path + '/rgb3_state' + str(i*4 + j + 1) + '.jpg', rgb3)
            cv2.imwrite(save_path + '/rgb4_state' + str(i*4 + j + 1) + '.jpg', rgb4)
            cv2.imwrite(save_path + '/rgb5_state' + str(i*4 + j + 1) + '.jpg', rgb5)

            # get the distance metrics between the point cloud and goal
            dist_metrics = {'CD': chamfer(unnorm_pcl, raw_goal),
                            'EMD': emd(unnorm_pcl, raw_goal),
                            'HAUSDORFF': hausdorff(unnorm_pcl, raw_goal)}

            print("\nDists: ", dist_metrics)
            with open(save_path + '/dist_metrics_' + str(i+1) + '.txt', 'w') as f:
                f.write(str(dist_metrics))
            
            # if that action was predicted to be the final action, then terminate the experiment
            if terminate > 0:
                in_progress = False
                break
            
    # completed the experiment, send the message to the video recording loop
    done_queue.put("Done!")
    
    # create and save a dictionary of the experiment results
    results_dict = {'n_actions': n_action, 'avg planning time': np.mean(planning_time_list), 'chamfer_distance': dist_metrics['CD'], 'earth_movers_distance': dist_metrics['EMD']}
    with open(save_path + '/results.txt', 'w') as f:
        f.write(str(results_dict))

# VIDEO THREAD
def video_loop(cam_pipeline, save_path, done_queue):
    '''
    '''
    forcc = cv2.VideoWriter_fourcc(*'XVID')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    out = cv2.VideoWriter(save_path + '/video.avi', forcc, 30.0, (1280, 800))

    frame_save_counter = 0
    # record until main loop is complete
    while done_queue.empty():
        frames = cam_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # crop and rotate the image to just show elevated stage area
        cropped_image = color_image[320:520,430:690,:]
        rotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # save frame approx. every 100 frames
        if frame_save_counter % 100 == 0:
            cv2.imwrite(save_path + '/external_rgb' + str(frame_save_counter) + '.jpg', rotated_image)
        frame_save_counter += 1
        out.write(color_image)
    
    cam_pipeline.stop()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # -------------------------------------------------------------------
    # ---------------- Experimental Parameters to Define ----------------
    # -------------------------------------------------------------------
    exp_num = 11
    goal_shape = 'Line' 
    model_path = '/checkpoints/...' 
    centered_action = False
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    exp_save = 'Experiments/Exp' + str(exp_num)

    # check to make sure the experiment number is not already in use, if it is, increment the number to ensure no save overwrites
    while os.path.exists(exp_save):
        exp_num += 1
        exp_save = 'Experiments/Exp' + str(exp_num)

    # make the experiment folder
    os.mkdir(exp_save)

    # make the experiment dictionary with important information for the experiment run
    exp_dict = {'goal: ', goal_shape,
                'model: ', model_path,
                'centered_action: ', centered_action}
    
    with open(exp_save + '/experiment_params.txt', 'w') as f:
        f.write(str(exp_dict))

    # initialize the robot and reset joints
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()

    # initialize the cameras
    cam2 = vis.CameraClass(2)
    cam3 = vis.CameraClass(3)
    cam4 = vis.CameraClass(4)
    cam5 = vis.CameraClass(5) 

    # initialize camera 6 pipeline
    W = 1280
    H = 800
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('152522250441')
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipeline.start(config)

    # initialize the 3D vision code
    pcl_vis = vis.Vision3D()    

    # load in the goal and save to the experiment folder
    goal = np.load('goals/' + goal_shape + '.npy')
    # center goal
    goal = (goal - np.mean(goal, axis=0)) * 10.0
    np.save(exp_save + '/goal.npy', goal)

    # initialize the threads
    done_queue = queue.Queue()

    main_thread = threading.Thread(target=experiment_loop, args=(fa, cam2, cam3, cam4, cam5, pcl_vis, exp_save, goal_shape, model_path, done_queue, centered_action))
    video_thread = threading.Thread(target=video_loop, args=(pipeline, exp_save, done_queue))

    main_thread.start()
    video_thread.start()