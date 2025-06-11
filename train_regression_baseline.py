from policy import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from embeddings import EncoderHead
from test_dataset import ClayDataset, SubGoalClayDataset
from os.path import join
import os
import numpy as np
import torch


# exp name
exp_name = 'pottery_long_epochs_16pred_7datasetfixed_with_augs' # 'pottery_12pred_7datasetfixed_with_augs' #'subgoal_horizon_5_test_global_center' # 'pottery_20pred_with_augs'
ckpt_dir = 'checkpoints/' + exp_name
# if ckpt_dir does not exist, create it
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# load in pointbert encoder from pretrained weights
device = torch.device('cuda')
config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
model_config = config.model
pointbert_encoder = builder.model_builder(model_config)
# pointbert_encoder2 = builder.model_builder(model_config)
weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
pointbert_encoder.load_model_from_ckpt(weights_path)
pointbert_encoder.to(device)
# pointbert_encoder2.load_model_from_ckpt(weights_path)
# pointbert_encoder2.to(device)

# setup the projection head
encoded_dim = 768 
latent_dim = 512
projection_head = EncoderHead(encoded_dim, latent_dim).to(device)
# projection_head2 = EncoderHead(encoded_dim, latent_dim).to(device)

# define the dataloader
n_datapoints = 2520 # 2*2*1800 # the desired numer of datapoints after augmentation
n_raw_trajectories = 7 # the number of raw datapoints
pred_horizon = 16 # 12 # 8 # 20
num_epochs = 2000 # 1500 # 750
target_shape = "pottery" # ["Line", "X", "Cone", or "All_Shapes"] # TODO: select what shape target you are training for
dataset_path = '/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery' # '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/'
# test_dataset_path = "ClayDemoDataset/" + str(target_shape) + "/Test" 
center_actions = False
dataset = ClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions)
# dataset = SubGoalClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions, subgoal_stepsize=5)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8, # 64
    num_workers=4, # 4
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=True)

# save experiment parameters as a dictionary
exp_params = {'exp_name': exp_name,
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

# create the regression network
pred_net = RegressionConditionalUnet1D(input_dim=action_dim,
                                       global_cond_dim=obs_dim*obs_horizon).to(device)

# Standard ADAM optimizer
optimizer = torch.optim.AdamW(
    params=list(pred_net.parameters()) +
           list(pointbert_encoder.parameters()) +
           list(projection_head.parameters()),
    lr=1e-4, weight_decay=1e-6)

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
                pointcloud_features = pointbert_encoder(pointcloud)
                pointcloud_features = projection_head(pointcloud_features)

                # embed goal cloud
                goalcloud_features = pointbert_encoder(goalcloud)
                goalcloud_features = projection_head(goalcloud_features)

                # stack pointcloud features for each obs horizon
                pointcloud_features = pointcloud_features.unsqueeze(1).repeat(1, obs_horizon, 1)
                goalcloud_features = goalcloud_features.unsqueeze(1).repeat(1, obs_horizon, 1)
                obs_features = torch.cat([pointcloud_features, nagent_pos, goalcloud_features],dim=-1)

                # concatenate vision feature and low-dim obs
                obs_cond = obs_features.flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                
                # predict the action
                predicted_actions = pred_net(
                    noisy_action, 
                    global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(predicted_actions, naction)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
            
            # save the model weights every 50 epochs
            mean_loss = np.mean(epoch_loss)
            if mean_loss < best_loss and epoch_idx % 50 == 0:
                best_loss = mean_loss
                print("\nSaving model weights with avg loss = ", mean_loss)

                # state dict pointbert
                torch.save(pointbert_encoder.state_dict(), join(ckpt_dir, 'pointbert_statedict'))
                # torch.save(nets['pointbert_encoder2'].state_dict(), join(ckpt_dir, 'pointbert2_statedict'))
                
                # projection head
                checkpoint = {'encoder_head': projection_head}
                torch.save(checkpoint, join(ckpt_dir, 'encoder_best_checkpoint'))
                # checkpoint = {'encoder_head2': nets['projection_head2']}
                # torch.save(checkpoint, join(ckpt_dir, 'encoder2_best_checkpoint'))

                # noise_pred_net
                noise_checkpoint = {'pred_net': pred_net}
                torch.save(noise_checkpoint, join(ckpt_dir, 'action_pred_best_checkpoint'))

        tglobal.set_postfix(loss=np.mean(epoch_loss))

# # Weights of the EMA model
# # is used for inference
# ema_nets = nets
# ema.copy_to(ema_nets.parameters())




# RegressionConditionalUnet1D