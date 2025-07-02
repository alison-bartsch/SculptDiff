from policy import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from embeddings import EncoderHead
from test_dataset import ClayDataset, SubGoalClayDataset, ClayDatasetForwardBackward
from os.path import join
import os
import numpy as np
import torch

def train_diffusion_policy(ckpt_dir, training_params):
    device = torch.device('cuda')
    
    if training_params['pretrained'] == True:
        if training_params['embedding'] == 'pointbert':
            encoder = None
            pass # load in pointbert encoder

        elif training_params['embedding'] == 'pointnet':
            encoder = None
            pass # load in pointnet encoder

        else:
            raise ValueError("Invalid embedding type. Choose 'pointbert' or 'pointnet'.")

    else:
        if training_params['embedding'] == 'pointbert':
            encoder = None
            pass # initialize pointbert encoder from scratch
        elif training_params['embedding'] == 'pointnet':
            encoder = None
            pass # initialize pointnet encoder from scratch
        else:
            raise ValueError("Invalid embedding type. Choose 'pointbert' or 'pointnet'.")

    # setup the projection head
    encoded_dim = 768 
    latent_dim = 512
    projection_head = EncoderHead(encoded_dim, latent_dim).to(device)

    # define the dataloader
    n_datapoints = 7200 
    n_raw_trajectories = 20 
    pred_horizon = 16 
    num_epochs = 1000 
    target_shape = "pottery" 
    dataset_path = '/home/alison/Documents/June18_Human_Demos_Train'
    center_actions = False
    discount_factor = 0.9 # if 1.0 then no discounting

    if training_params['subgoal']:
        dataset = SubGoalClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions, subgoal_stepsize=4, global_centering=training_params['global_centering'])
    elif training_params['forward_backward']:
        dataset = ClayDatasetForwardBackward(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions, global_centering=training_params['global_centering'])
    else:
        dataset = ClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions, global_centering=training_params['global_centering'])
    
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
    obs_dim = 2*pcl_feature_dim + lowdim_obs_dim
    action_dim = 8
    obs_horizon = 1

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    ).to(device)

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
                    # print("point cloud shape: ", nbatch['pointcloud'].shape)
                    pointcloud = nbatch['pointcloud'].to(device).float()
                    goalcloud = nbatch['goal'].to(device).float()
                    nagent_pos = nbatch['agent_pos'].to(device).unsqueeze(axis=1)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # embed point cloud
                    if training_params['embedding'] == 'pointbert':
                        pointcloud_features = nets['encoder'](pointcloud)
                        goalcloud_features = nets['encoder'](goalcloud)

                    elif training_params['embedding'] == 'pointnet':
                        pointcloud_features, _ = nets['encoder'](pointcloud)
                        goalcloud_features = nets['encoder'](goalcloud)

                    else:
                        raise ValueError("Invalid embedding type. Choose 'pointbert' or 'pointnet'.")
                    
                    # pointcloud_features = nets['encoder'](pointcloud)
                    pointcloud_features = nets['projection_head'](pointcloud_features)

                    # embed goal cloud
                    # goalcloud_features = nets['encoder'](goalcloud)
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


if __name__ == "__main__":
    train_dict = {'pointbert_pretrained_global_centering' : {'embedding' : 'pointbert',
                                                            'pretrained' : True,
                                                            'global_centering' : True,
                                                            'subgoal' : False,
                                                            'forward_backward' : False},
                'pointnet_pretrained_global_centering' : {'embedding' : 'pointnet',
                                                            'pretrained' : True,
                                                            'global_centering' : True,
                                                            'subgoal' : False,
                                                            'forward_backward' : False}}

    for train_name, train_params in train_dict.items():
        ckpt_dir = 'checkpoints/' + train_name
        # if ckpt_dir does not exist, create it
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        train_diffusion_policy(ckpt_dir, train_params)