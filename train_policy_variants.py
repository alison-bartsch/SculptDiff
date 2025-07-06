from policy import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from pointnet.models.model_dp3_pytorch import PointNetEncoderXYZ
from embeddings import EncoderHead, GoalMeasureHead
from test_dataset import ClayDataset, SubGoalClayDataset, ClayDatasetForwardBackward, ClayDatasetContinualGuidance, ClayDatasetGoalMeasure
from os.path import join
import os
import numpy as np
import torch

def train_diffusion_policy(ckpt_dir, training_params):
    device = torch.device('cuda')

    print("\n\n\n\nModel: ", ckpt_dir)
    
    if training_params['pretrained'] == True:
        if training_params['embedding'] == 'pointbert':
            config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
            model_config = config.model
            encoder = builder.model_builder(model_config)
            weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
            encoder.load_model_from_ckpt(weights_path)
            encoder.to(device)

        elif training_params['embedding'] == 'pointnet':
            encoder = PointNetEncoderXYZ().to(device)
            checkpoint_path = "/home/alison/Documents/GitHub/SculptDiff/pointnet/weights/best_model_epoch_181.pth"
            state_dict = torch.load(checkpoint_path, map_location=device)
            # Load only the encoder weights
            encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')})

        else:
            raise ValueError("Invalid embedding type. Choose 'pointbert' or 'pointnet'.")

    else:
        if training_params['embedding'] == 'pointbert':
            config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
            model_config = config.model
            encoder = builder.model_builder(model_config)
            encoder.to(device)
        
        elif training_params['embedding'] == 'pointnet':
            encoder = PointNetEncoderXYZ().to(device)
        
        else:
            raise ValueError("Invalid embedding type. Choose 'pointbert' or 'pointnet'.")

    if training_params['measure_goal'] == True:
        # setup the goal measurement projection
        goal_measure_encoder = GoalMeasureHead(1, 512).to(device)

    # setup the projection head
    if training_params['embedding'] == 'pointbert':
    # load in pointbert encoder from pretrained weights
        encoded_dim = 768 
        latent_dim = 512
        projection_head = EncoderHead(encoded_dim, latent_dim).to(device)
    elif training_params['embedding'] == 'pointnet':
        # setup the projection head
        encoded_dim = 1024
        latent_dim = 512
        projection_head = EncoderHead(encoded_dim, latent_dim, is_pointBERT=False).to(device)

    # define the dataloader
    n_datapoints = 3600 
    n_raw_trajectories = 20 
    pred_horizon = 16 
    num_epochs = 1500 
    target_shape = "pottery" 
    dataset_path = '/home/alison/Documents/June18_Human_Demos_Train'
    center_actions = False
    discount_factor = 0.9 # if 1.0 then no discounting
    subgoal_stepsize = 8

    if training_params['subgoal']:
        dataset = SubGoalClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions, subgoal_stepsize=subgoal_stepsize)
    elif training_params['forward_backward']:
        dataset = ClayDatasetForwardBackward(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions)
    elif training_params['continual_guidance']:
        dataset = ClayDatasetContinualGuidance(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions)
    elif training_params['measure_goal']:
        dataset = ClayDatasetGoalMeasure(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions)
    else:
        dataset = ClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions, training_params['rotate_goal'])
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8, # 64
        num_workers=4, # 4
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True)

    min, max = dataset.get_dataset_min_max_stats()
    # save the min and max action values 
    np.save(ckpt_dir + '/action_mins.npy', min)
    np.save(ckpt_dir + '/action_maxs.npy', max)

    # save experiment parameters as a dictionary
    exp_params = {'exp_name': ckpt_dir.split('/')[-1],
                'n_datapoints': n_datapoints, 
                'n_raw_trajectories': n_raw_trajectories, 
                'pred_horizon': pred_horizon,
                'center_actions': center_actions,
                'n_epochs': num_epochs,
                'dataset': dataset_path}
    with open(ckpt_dir + '/experiment_params.txt', 'w') as f:
            f.write(str(exp_params))

    # define the noise scheduler
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

    # define parameters
    pcl_feature_dim = 512
    lowdim_obs_dim = 8 
    if training_params['subgoal']:
        obs_dim = int((pred_horizon + subgoal_stepsize) / subgoal_stepsize)*pcl_feature_dim + lowdim_obs_dim
    else:
        obs_dim = 2*pcl_feature_dim + lowdim_obs_dim
    action_dim = 8
    obs_horizon = 1

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    ).to(device)

    if training_params['measure_goal'] == True:
        nets = nn.ModuleDict({
            'encoder': encoder,
            'projection_head': projection_head,
            'noise_pred_net': noise_pred_net,
            'goal_measure_encoder': goal_measure_encoder
            })
    else:
        nets = nn.ModuleDict({
            'encoder': encoder,
            'projection_head': projection_head,
            'noise_pred_net': noise_pred_net
        })

    # Exponential Moving Average
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    best_loss = 1e3
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    if training_params['subgoal']:
                        nagent_pos = nbatch['agent_pos'].to(device).unsqueeze(axis=1)
                        naction = nbatch['action'].to(device)
                        B = nagent_pos.shape[0]

                        obs_features = [nagent_pos]
                        for i in range(nbatch['pcl_seq'].shape[1]):
                            pcl = nbatch['pcl_seq'][:, i, :, :].to(device).float()

                            if training_params['embedding'] == 'pointbert':
                                # embed point cloud using pointbert
                                pcl_features = nets['encoder'](pcl)
                                pcl_features = nets['projection_head'](pcl_features)
                            
                            elif training_params['embedding'] == 'pointnet':
                                # embed point cloud using pointnet
                                pcl_features, _ = nets['encoder'](pcl)
                                pcl_features = nets['projection_head'](pcl_features)
                            
                            else:
                                raise ValueError("Invalid embedding type. Choose 'pointbert' or 'pointnet'.")

                            # weight the pcl features based on order
                            pcl_features = discount_factor ** i * pcl_features
                            pcl_features = pcl_features.unsqueeze(1).repeat(1, obs_horizon, 1)
                            obs_features.append(pcl_features)
                        obs_features = torch.cat(obs_features, dim=-1)
                        
                    else:
                        pointcloud = nbatch['pointcloud'].to(device).float()
                        goalcloud = nbatch['goal'].to(device).float()
                        nagent_pos = nbatch['agent_pos'].to(device).unsqueeze(axis=1)
                        naction = nbatch['action'].to(device)
                        B = nagent_pos.shape[0]

                        # embed point cloud
                        if training_params['embedding'] == 'pointbert':
                            pointcloud_features = nets['encoder'](pointcloud)

                            if training_params['measure_goal']:
                                # embed goal cloud using goal measure encoder
                                goalcloud_features = nets['goal_measure_encoder'](goalcloud)
                            else:
                                goalcloud_features = nets['encoder'](goalcloud)

                        elif training_params['embedding'] == 'pointnet':
                            pointcloud_features, _ = nets['encoder'](pointcloud)

                            if training_params['measure_goal']:
                                # embed goal cloud using goal measure encoder
                                goalcloud_features = nets['goal_measure_encoder'](goalcloud)
                            else:
                                goalcloud_features, _ = nets['encoder'](goalcloud)

                        else:
                            raise ValueError("Invalid embedding type. Choose 'pointbert' or 'pointnet'.")
                        
                        # pointcloud_features = nets['encoder'](pointcloud)
                        pointcloud_features = nets['projection_head'](pointcloud_features)

                        # embed goal cloud
                        # goalcloud_features = nets['encoder'](goalcloud)
                        if training_params['measure_goal'] == False:
                            goalcloud_features = nets['projection_head'](goalcloud_features)

                        # stack pointcloud features for each obs horizon
                        pointcloud_features = pointcloud_features.unsqueeze(1).repeat(1, obs_horizon, 1)
                        goalcloud_features = goalcloud_features.unsqueeze(1).repeat(1, obs_horizon, 1)
                        obs_features = torch.cat([pointcloud_features, nagent_pos, goalcloud_features],dim=-1)

                    # concatenate vision feature and low-dim obs
                    obs_cond = obs_features.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean actions according to the noise magnitude at each diffusion iteration
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = nets['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                
                # save the model weights every 50 epochs
                mean_loss = np.mean(epoch_loss)
                if mean_loss < best_loss and epoch_idx % 10 == 0:
                    best_loss = mean_loss
                    print("\nSaving model weights with avg loss = ", mean_loss)

                    if training_params['embedding'] == 'pointbert':
                        # state dict pointbert
                        torch.save(nets['encoder'].state_dict(), join(ckpt_dir, 'pointbert_statedict'))
                    
                    elif training_params['embedding'] == 'pointnet':
                        pointnet_checkpoint = {'encoder': nets['encoder']}
                        torch.save(pointnet_checkpoint, join(ckpt_dir, 'pointnet_best_checkpoint.zip'))

                    if training_params['measure_goal'] == True:
                        # goal measure head
                        checkpoint = {'goal_measure_encoder': nets['goal_measure_encoder']}
                        torch.save(checkpoint, join(ckpt_dir, 'goal_measure_best_checkpoint'))
                    
                    # projection head
                    checkpoint = {'encoder_head': nets['projection_head']}
                    torch.save(checkpoint, join(ckpt_dir, 'projection_encoder_best_checkpoint'))

                    # noise_pred_net
                    noise_checkpoint = {'noise_pred_net': nets['noise_pred_net']}
                    torch.save(noise_checkpoint, join(ckpt_dir, 'noise_pred_best_checkpoint'))


            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())

# 'pointbert_pretrained_measured_goal' : {'embedding' : 'pointbert',
#                                                     'pretrained' : True,
#                                                     'subgoal' : False,
#                                                     'forward_backward' : False,
#                                                     'continual_guidance' : False,
#                                                     'measure_goal' : True,
#                                                     'rotate_goal' : True},
#                 'pointbert_pretrained_no_rotate_goal' : {'embedding' : 'pointbert',
#                                                     'pretrained' : True,
#                                                     'subgoal' : False,
#                                                     'forward_backward' : False,
#                                                     'continual_guidance' : False,
#                                                     'measure_goal' : False,
#                                                     'rotate_goal' : False},
#                 'pointbert_pretrained_subgoal' : {'embedding' : 'pointbert',
#                                                     'pretrained' : True,
#                                                     'subgoal' : True,
#                                                     'forward_backward' : False,
#                                                     'continual_guidance' : False,
#                                                     'measure_goal' : False,
#                                                     'rotate_goal' : True},


if __name__ == "__main__":
    train_dict = {'pointbert_pretrained_forward_backward' : {'embedding' : 'pointbert',
                                                    'pretrained' : True,
                                                    'subgoal' : False,
                                                    'forward_backward' : True,
                                                    'continual_guidance' : False,
                                                    'measure_goal' : False,
                                                    'rotate_goal' : True},
                'pointbert_pretrained_continual_guidance' : {'embedding' : 'pointbert',
                                                    'pretrained' : True,
                                                    'subgoal' : False,
                                                    'forward_backward' : False,
                                                    'continual_guidance' : True,
                                                    'measure_goal' : False,
                                                    'rotate_goal' : True},
                'pointnet_pretrained_forward' : {'embedding' : 'pointnet',
                                                    'pretrained' : True,
                                                    'subgoal' : False,
                                                    'forward_backward' : False,
                                                    'continual_guidance' : False,
                                                    'measure_goal' : False,
                                                    'rotate_goal' : True},
                'pointnet_pretrained_subgoal' : {'embedding' : 'pointnet',
                                                    'pretrained' : True,
                                                    'subgoal' : True,
                                                    'forward_backward' : False,
                                                    'continual_guidance' : False,
                                                    'measure_goal' : False,
                                                    'rotate_goal' : True},
                'pointnet_pretrained_forward_backward' : {'embedding' : 'pointnet',
                                                    'pretrained' : True,
                                                    'subgoal' : False,
                                                    'forward_backward' : True,
                                                    'continual_guidance' : False,
                                                    'measure_goal' : False,
                                                    'rotate_goal' : True},
                'pointnet_pretrained_continual_guidance' : {'embedding' : 'pointnet',
                                                    'pretrained' : True,
                                                    'subgoal' : False,
                                                    'forward_backward' : False,
                                                    'continual_guidance' : True,
                                                    'measure_goal' : False,
                                                    'rotate_goal' : True},
                'pointnet_pretrained_measured_goal' : {'embedding' : 'pointnet',
                                                    'pretrained' : True,
                                                    'subgoal' : False,
                                                    'forward_backward' : False,
                                                    'continual_guidance' : False,
                                                    'measure_goal' : True,
                                                    'rotate_goal' : True},
                'pointbert_untrained_forward' : {'embedding' : 'pointbert',
                                                    'pretrained' : False,
                                                    'subgoal' : False,
                                                    'forward_backward' : False,
                                                    'continual_guidance' : False,
                                                    'measure_goal' : False,
                                                    'rotate_goal' : True},
                'pointnet_untrained_forward' : {'embedding' : 'pointnet',
                                                    'pretrained' : False,
                                                    'subgoal' : False,
                                                    'forward_backward' : False,
                                                    'continual_guidance' : False,
                                                    'measure_goal' : False,
                                                    'rotate_goal' : True},}

    for train_name, train_params in train_dict.items():
        ckpt_dir = 'checkpoints/' + train_name
        # if ckpt_dir does not exist, create it
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        train_diffusion_policy(ckpt_dir, train_params)