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
from test_collision_checker import check_finger_collision
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def calculate_intermediate_pose(final_pose, dist=0.05):
    """
    Calculate an intermediate pose for the robot to move before executing the grasp.
    Specifically, find the position of the gripper at a distance of [dist] from the final pose
    in the direction of the final_pose rotation.
    """
    # NOTE: may want to add a positional offset towards the center of the clay???? to prevent wall sliding when too close
    final_position = final_pose.translation
    final_rotation = final_pose.rotation
    final_rotation = Rotation.from_matrix(final_rotation)
    final_rotation = final_rotation.as_matrix()
    final_rotation = np.array(final_rotation)
    final_rotation = final_rotation[:, 2]
    intermediate_position = final_position - dist * final_rotation
    intermediate_pose = final_pose
    intermediate_pose.translation = intermediate_position
    return intermediate_pose


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

    intermediate_pose = calculate_intermediate_pose(pose.copy())
    fa.goto_pose(intermediate_pose)

    fa.goto_pose(pose)
    fa.goto_gripper(d, force=60.0)
    time.sleep(3)
    return intermediate_pose

def regression_generate_actions(pointbert, projection_head, pred_net, pointcloud, numpy_goal, nagent_pos, obs_horizon, action_dim, device):
    B = 1
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
        print("\nNagent pos: ", nagent_pos)
        obs_features = torch.cat([pointcloud_features, nagent_pos, goalcloud_features],dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, pred_horizon, action_dim), device=device)
        
        # predict the action
        predicted_actions = pred_net(
            noisy_action, 
            global_cond=obs_cond)

        # unnormalize action
        naction = predicted_actions.detach().to('cpu').numpy()
        end = time.time()
    return naction, end - start


def experiment_loop(fa, cam2, cam3, cam4, cam5, pcl_vis, save_path, goal_str, ckpt_dir, done_queue, centered_action, pred_horizon, execute_horizon, goal_path, collision_check):
    '''
    '''
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
    testweights_path = ckpt_dir + '/pointbert_statedict.zip' 
    pointbert.load_state_dict(torch.load(testweights_path))
    pointbert.to(device)

    # load projection head from ckpt_dir
    enc_checkpoint = torch.load(ckpt_dir + '/encoder_best_checkpoint.zip', map_location=torch.device('cpu')) 
    projection_head = enc_checkpoint['encoder_head'].to(device)

    # # load noise_pred_net from ckpt_dir
    # noise_checkpoint = torch.load(ckpt_dir + '/noise_pred_best_checkpoint.zip', map_location=torch.device('cpu')) 
    # noise_pred_net = noise_checkpoint['noise_pred_net'].to(device)

    # load pred_net from ckpt_dir
    pred_checkpoint = torch.load(ckpt_dir + '/action_pred_best_checkpoint.zip', map_location=torch.device('cpu'))
    pred_net = pred_checkpoint['pred_net']

    # load in the goal
    raw_goal = np.load(goal_path)

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

    unnorm_pcl, ctr = pcl_vis.unnormalize_fuse_point_clouds_no_base(pc2, pc3, pc4, pc5, color="Orange")
    # center and scale pointcloud
    pointcloud = (np.copy(unnorm_pcl) - ctr) * 10

    # save the point clouds from each camera
    o3d.io.write_point_cloud(save_path + '/cam2_pcl0.ply', pc2)
    o3d.io.write_point_cloud(save_path + '/cam3_pcl0.ply', pc3)
    o3d.io.write_point_cloud(save_path + '/cam4_pcl0.ply', pc4)
    o3d.io.write_point_cloud(save_path + '/cam5_pcl0.ply', pc5)

    # center the goal based on the goal center
    numpy_goal = (np.copy(raw_goal) - ctr) * 10.0
    # scale distance metric goal differently 
    dist_goal = np.copy(numpy_goal)

    # visualize observation vs goal cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(unnorm_pcl)
    pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(unnorm_pcl),1)))
    goal_pcl = o3d.geometry.PointCloud()
    goal_pcl.points = o3d.utility.Vector3dVector(dist_goal)
    goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(dist_goal),1)))

    # save observation
    np.save(save_path + '/pcl0.npy', pointcloud)
    np.save(save_path + '/center0.npy', ctr)
    cv2.imwrite(save_path + '/rgb2_state0.jpg', rgb2)
    cv2.imwrite(save_path + '/rgb3_state0.jpg', rgb3)
    cv2.imwrite(save_path + '/rgb4_state0.jpg', rgb4)
    cv2.imwrite(save_path + '/rgb5_state0.jpg', rgb5)

    print("\nMax unnorm pcl: ", np.max(unnorm_pcl, axis=0))
    print("\nMax raw goal: ", np.max(raw_goal, axis=0))

    # get the distance metrics between the point cloud and goal
    dist_metrics = {'CD': chamfer(unnorm_pcl, raw_goal),
                    'EMD': emd(unnorm_pcl, raw_goal),
                    'HAUSDORFF': hausdorff(unnorm_pcl, raw_goal)}

    print("\nDists: ", dist_metrics)

    # calculate the dist metrics if the pcls are centered
    ctr_pcl = unnorm_pcl.copy() - np.mean(unnorm_pcl, axis=0)
    ctr_goal = raw_goal.copy() - np.mean(raw_goal, axis=0)
    centered_dists = {'CD': chamfer(ctr_pcl, ctr_goal),
                    'EMD': emd(ctr_pcl, ctr_goal),
                    'HAUSDORFF': hausdorff(ctr_pcl, ctr_goal)}
    print("\nCentered Dists: ", centered_dists)

    with open(save_path + '/dist_metrics_0.txt', 'w') as f:
        f.write(str(dist_metrics))

    iter = 1
    in_progress = True
    while in_progress:
        naction, total_time = regression_generate_actions(pointbert, projection_head, pred_net, pointcloud, numpy_goal, nagent_pos, obs_horizon, action_dim, device)
        og_nagent_pos = nagent_pos.detach().clone()
        og_pointcloud = pointcloud.copy()
        planning_time_list.append(total_time)

        # execute N actions before replanning
        pred_action = naction[0]
        print("\nPredicted action sequence: ", pred_action)
        termination_pred = pred_action[:,7]
        print("\nTermination prediction: ", termination_pred)
        action_pred = (pred_action[:,0:7] + 1.0) / 2.0
        action_pred = action_pred * (a_maxs7d - a_mins7d) + a_mins7d
        
        # for j in range(action_pred.shape[0]):
        for j in range(execute_horizon):
            unnorm_a = action_pred[j,:]
            print("\nSingle-step action: ", unnorm_a)
            terminate = termination_pred[j]

            # check for collision with the point cloud if the initial piercing actions have been executed
            if iter > 6 and collision_check:
                collision = check_finger_collision(unnorm_a, pcl, vis=False)
                n_checks = 0
                while collision and n_checks < 10:
                    n_checks += 1
                    print("\nCollision detected, replanning...")
                    naction, total_time = regression_generate_actions(pointbert, projection_head,  pred_net, og_pointcloud, numpy_goal, og_nagent_pos, obs_horizon, action_dim, device)
                    pred_action = naction[0]
                    termination_pred = pred_action[:,7]
                    action_pred = (pred_action[:,0:7] + 1.0) / 2.0
                    action_pred = action_pred * (a_maxs7d - a_mins7d) + a_mins7d
                    unnorm_a = action_pred[j,:]
                    print("unnorm a new: ", unnorm_a)
                    terminate = termination_pred[j]
                    collision = check_finger_collision(unnorm_a, pcl, vis=False)

            if centered_action:
                unnorm_a[0:3] = unnorm_a[0:3] + ctr

            # update nagent_pos to be the new position
            nagent_pos = torch.from_numpy(pred_action[j]).to(torch.float32).unsqueeze(axis=0).unsqueeze(axis=0).to(device)
            
            # assert False
            intermediate_pose = goto_grasp(fa, unnorm_a[0], unnorm_a[1], unnorm_a[2], unnorm_a[3], unnorm_a[4], unnorm_a[5], unnorm_a[6])
            n_action+=1

            # wait here
            time.sleep(3)

            # open the gripper
            fa.goto_gripper(0.04, block=True)

            # move to intermediate_pose
            fa.goto_pose(intermediate_pose)

            # move to observation pose
            pose.translation = observation_pose
            fa.goto_pose(pose)

            # get the observation state
            rgb2, _, pc2, _ = cam2._get_next_frame()
            rgb3, _, pc3, _ = cam3._get_next_frame()
            rgb4, _, pc4, _ = cam4._get_next_frame()
            rgb5, _, pc5, _ = cam5._get_next_frame()
            unnorm_pcl, ctr = pcl_vis.unnormalize_fuse_point_clouds_no_base(pc2, pc3, pc4, pc5, color="Orange")
            # center and scale pointcloud
            pointcloud = (np.copy(unnorm_pcl) - ctr) * 10

            # save the point clouds from each camera
            o3d.io.write_point_cloud(save_path + '/cam2_pcl' + str(iter) + '.ply', pc2)
            o3d.io.write_point_cloud(save_path + '/cam3_pcl' + str(iter) + '.ply', pc3)
            o3d.io.write_point_cloud(save_path + '/cam4_pcl' + str(iter) + '.ply', pc4)
            o3d.io.write_point_cloud(save_path + '/cam5_pcl' + str(iter) + '.ply', pc5)

            # center the goal based on the point cloud center
            numpy_goal = (np.copy(raw_goal) - ctr) * 10.0
            # scale distance metric goal differently 
            dist_goal = np.copy(numpy_goal)

            # visualize observation vs goal cloud
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(unnorm_pcl)
            pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]), (len(unnorm_pcl),1)))
            goal_pcl = o3d.geometry.PointCloud()
            goal_pcl.points = o3d.utility.Vector3dVector(dist_goal)
            goal_pcl.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]), (len(dist_goal),1)))
            # o3d.visualization.draw_geometries([pcl, goal_pcl])

            # save observation
            np.save(save_path + '/pcl' + str(iter) + '.npy', pointcloud)
            np.save(save_path + '/center' + str(iter) + '.npy', ctr)
            cv2.imwrite(save_path + '/rgb2_state' + str(iter) + '.jpg', rgb2)
            cv2.imwrite(save_path + '/rgb3_state' + str(iter) + '.jpg', rgb3)
            cv2.imwrite(save_path + '/rgb4_state' + str(iter) + '.jpg', rgb4)
            cv2.imwrite(save_path + '/rgb5_state' + str(iter) + '.jpg', rgb5)

            # get the distance metrics between the point cloud and goal
            dist_metrics = {'CD': chamfer(unnorm_pcl, raw_goal),
                            'EMD': emd(unnorm_pcl, raw_goal),
                            'HAUSDORFF': hausdorff(unnorm_pcl, raw_goal)}

            print("\nDists: ", dist_metrics)
            with open(save_path + '/dist_metrics_' + str(iter) + '.txt', 'w') as f:
                f.write(str(dist_metrics))

            # calculate the dist metrics if the pcls are centered
            ctr_pcl = unnorm_pcl.copy() - np.mean(unnorm_pcl, axis=0)
            ctr_goal = raw_goal.copy() - np.mean(raw_goal, axis=0)
            centered_dists = {'CD': chamfer(ctr_pcl, ctr_goal),
                            'EMD': emd(ctr_pcl, ctr_goal),
                            'HAUSDORFF': hausdorff(ctr_pcl, ctr_goal)}
            print("\nCentered Dists: ", centered_dists)
            
            # if that action was predicted to be the final action, then terminate the experiment
            if terminate > 0:
                in_progress = False
                break

            iter += 1
            
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
    exp_num = 1
    goal_shape = 'pottery' 
    model_path = '/home/alison/Documents/GitHub/SculptDiff/checkpoints/regression_16pred_7datasetfixed_with_augs'
    goal_path = '/home/alison/Clay_Data/Mar24_Human_Demos_Raw_Thick_Cast_Soft/pottery/Trajectory2/unnormalized_pointcloud33.npy'
    centered_action = False
    pred_horizon = 16 
    execute_horizon = 16 
    collision_check = False
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

    # make the experiment folder for the video save
    os.mkdir('/home/alison/Documents/SculptDiff_experiment_videos/Exp' + str(exp_num))
    video_save_path = '/home/alison/Documents/SculptDiff_experiment_videos/Exp' + str(exp_num)

    # make the experiment dictionary with important information for the experiment run
    exp_dict = {'goal: ', goal_shape,
                'model: ', model_path,
                'goal: ', goal_path,
                'centered_action: ', centered_action,
                'pred_horizon: ', pred_horizon,
                'execute_horizon: ', execute_horizon,
                'collision_check: ', collision_check}
    
    with open(exp_save + '/experiment_params.txt', 'w') as f:
        f.write(str(exp_dict))

    # initialize the robot and reset joints
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    fa.goto_gripper(0.04)

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

    # initialize the threads
    done_queue = queue.Queue()

    main_thread = threading.Thread(target=experiment_loop, args=(fa, cam2, cam3, cam4, cam5, pcl_vis, exp_save, goal_shape, model_path, done_queue, centered_action, pred_horizon, execute_horizon, goal_path, collision_check))
    video_thread = threading.Thread(target=video_loop, args=(pipeline, video_save_path, done_queue))

    main_thread.start()
    video_thread.start()