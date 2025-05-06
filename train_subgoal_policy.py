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
exp_name = 'subgoal_3_pcl_seq_12pred_7datasetfixed_with_augs' 
ckpt_dir = 'checkpoints/' + exp_name
# if ckpt_dir does not exist, create it
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# load in pointbert encoder from pretrained weights
device = torch.device('cuda')
config = cfg_from_yaml_file('pointBERT/cfgs/PointTransformer.yaml')
model_config = config.model
pointbert_encoder = builder.model_builder(model_config)
weights_path = 'pointBERT/point-BERT-weights/Point-BERT.pth'
pointbert_encoder.load_model_from_ckpt(weights_path)
pointbert_encoder.to(device)

# setup the projection head
encoded_dim = 768 
latent_dim = 512
projection_head = EncoderHead(encoded_dim, latent_dim).to(device)

# define the dataloader
n_datapoints = 2520 # 2*2*1800 # the desired numer of datapoints after augmentation
n_raw_trajectories = 7 # the number of raw datapoints
pred_horizon = 12 # 8 # 20
subgoal_stepsize = 3
num_epochs = 750
target_shape = "pottery" # ["Line", "X", "Cone", or "All_Shapes"] # TODO: select what shape target you are training for
dataset_path = '/home/alison/Documents/Mar24_Bowl_Demos_Soft_Finger/pottery' # '/home/alison/Documents/Feb26_Human_Demos_Raw/pottery/'
# test_dataset_path = "ClayDemoDataset/" + str(target_shape) + "/Test" 
center_actions = False
# dataset = ClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions)
dataset = SubGoalClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions, subgoal_stepsize)
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
obs_dim = int((pred_horizon + subgoal_stepsize) / subgoal_stepsize)*pcl_feature_dim + lowdim_obs_dim
action_dim = 8
obs_horizon = 1

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
).to(device)

nets = nn.ModuleDict({
    'pointbert_encoder': pointbert_encoder,
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
                nagent_pos = nbatch['agent_pos'].to(device).unsqueeze(axis=1)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]

                obs_features = [nagent_pos]
                for i in range(nbatch['pcl_seq'].shape[1]):
                    pcl = nbatch['pcl_seq'][:, i, :, :].to(device).float()
                    pcl_features = nets['pointbert_encoder'](pcl)
                    pcl_features = nets['projection_head'](pcl_features)
                    pcl_features = pcl_features.unsqueeze(1).repeat(1, obs_horizon, 1)
                    obs_features.append(pcl_features)
                obs_features = torch.cat(obs_features, dim=-1)

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
            if mean_loss < best_loss and epoch_idx % 50 == 0:
                best_loss = mean_loss
                print("\nSaving model weights with avg loss = ", mean_loss)

                # state dict pointbert
                torch.save(nets['pointbert_encoder'].state_dict(), join(ckpt_dir, 'pointbert_statedict'))
                
                # projection head
                checkpoint = {'encoder_head': nets['projection_head']}
                torch.save(checkpoint, join(ckpt_dir, 'encoder_best_checkpoint'))

                # noise_pred_net
                noise_checkpoint = {'noise_pred_net': nets['noise_pred_net']}
                torch.save(noise_checkpoint, join(ckpt_dir, 'noise_pred_best_checkpoint'))

        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_nets = nets
ema.copy_to(ema_nets.parameters())