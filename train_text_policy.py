from policy import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from embeddings import EncoderHead
from dataset import ClayDataset
from os.path import join
import os
import numpy as np
import torch
import random
from sentence_transformers import SentenceTransformer, util

# exp name
exp_name = 'Line_textgoal'
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
n_datapoints = 2880 # the desired numer of datapoints after augmentation
n_raw_trajectories = 7 # the number of raw datapoints
pred_horizon = 4 
num_epochs = 750
target_shape = "Line" # ["Line", "X", "Cone", or "All_Shapes"] # TODO: select what shape target you are training for
dataset_path = "ClayDemoDataset/" + str(target_shape) + "/Train" 
test_dataset_path = "ClayDemoDataset/" + str(target_shape) + "/Test" 
center_actions = False
dataset = ClayDataset(dataset_path, pred_horizon, n_datapoints, n_raw_trajectories, center_actions)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8, # 64
    num_workers=4, # 4
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=True)

# add in sentence embedding code for semantic goal prompting
prompts = [f"make an {target_shape}", 
           f"sculpt a {target_shape}", 
           f"please creat a {target_shape}", 
           f"I would like a {target_shape} sculpture",
           f"could you make a {target_shape} for me?",
           f"I need a {target_shape} sculpture",
           f"sculpt a {target_shape} for me",
           f"make a {target_shape} sculpture"]
text_model = SentenceTransformer("all-MiniLM-L6-v2")

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
lowdim_obs_dim = 5 
# obs_dim = 2*pcl_feature_dim + lowdim_obs_dim
obs_dim = pcl_feature_dim + lowdim_obs_dim + 384
action_dim = 5 
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
                pointcloud = nbatch['pointcloud'].to(device).float()
                # goalcloud = nbatch['goal'].to(device).float()
                nagent_pos = nbatch['agent_pos'].to(device).unsqueeze(axis=1)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]

                # randomly select a prompt and embed with sentence transformer
                goal_prompt = random.choice(prompts).format(clay_shape=target_shape)
                goal_embed = text_model.encode(goal_prompt)
                goal_embed = torch.from_numpy(goal_embed).to(torch.float32).unsqueeze(0).to(device)
                # tile goal_embed for B times
                goal_features = goal_embed.unsqueeze(0).repeat(B, 1, 1)

                # embed point cloud
                pointcloud_features = nets['pointbert_encoder'](pointcloud)
                pointcloud_features = nets['projection_head'](pointcloud_features)

                # embed goal cloud
                # goalcloud_features = nets['pointbert_encoder'](goalcloud)
                # goalcloud_features = nets['projection_head'](goalcloud_features)

                # stack pointcloud features for each obs horizon
                pointcloud_features = pointcloud_features.unsqueeze(1).repeat(1, obs_horizon, 1)
                # goalcloud_features = goalcloud_features.unsqueeze(1).repeat(1, obs_horizon, 1)
                obs_features = torch.cat([pointcloud_features, nagent_pos, goal_features],dim=-1)

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

            if epoch_idx % 100 == 0:
                print("\n\n\n\n\n----------------- PREDICTIONS -------------------")
                with torch.inference_mode():
                    nets['projection_head'].eval()
                    nets['pointbert_encoder'].eval()
                    nets['noise_pred_net'].eval()

                    # do this for test trajectories
                    trajs = [0,1]
                    start_state = [0,0]
                    for k in range(len(trajs)):
                        t = trajs[k]
                        s_idx = start_state[k]

                        # import the state, center and goal 
                        ctr = np.load(test_dataset_path + '/Discrete/Trajectory' + str(t) + '/pcl_center' + str(s_idx) + '.npy')
                        # goal = np.load(test_dataset_path + '/goal_unnormalized.npy')
                        state = np.load(test_dataset_path + '/Discrete/Trajectory' + str(t) + '/state' + str(s_idx) + '.npy')

                        # # center and scale goal
                        # goal = (goal - ctr) * 10.0
                        # goal = torch.from_numpy(goal).to(torch.float32)
                        # goals = torch.unsqueeze(goal, 0).to(device)
                        # tokenized_goals = nets['pointbert_encoder'](goals)
                        # goal_embed = nets['projection_head'](tokenized_goals)
                        # goal_features = goal_embed.unsqueeze(1).repeat(1, obs_horizon, 1)

                        # get goal features
                        goal_prompt = random.choice(prompts).format(clay_shape=target_shape)
                        goal_embed = text_model.encode(goal_prompt)
                        goal_embed = torch.from_numpy(goal_embed).to(torch.float32).unsqueeze(0).to(device)
                        goal_features = goal_embed.unsqueeze(0).repeat(1, 1, 1)

                        # center and scale state
                        state = (state - ctr) * 10.0
                        state = torch.from_numpy(state).to(torch.float32)
                        states = torch.unsqueeze(state, 0).to(device)
                        tokenized_states = nets['pointbert_encoder'](states)
                        pcl_embed = nets['projection_head'](tokenized_states) 
                        pointcloud_features = pcl_embed.unsqueeze(1).repeat(1, obs_horizon, 1)

                        # get the previous action
                        if s_idx == 0:
                            pos = np.array([0.6, 0.0, 0.165, 0.0, 0.05])
                        else:
                            pos = np.load(test_dataset_path + '/Discrete/Trajectory' + str(t) + '/action' + str(s_idx-1) + '.npy')
                        
                        # normalize and scale action
                        a_mins5d = np.array([0.56, -0.062, 0.125, -90, 0.005])
                        a_maxs5d = np.array([0.7, 0.062, 0.165, 90, 0.05])
                        pos = (pos - a_mins5d) / (a_maxs5d - a_mins5d)
                        pos = pos * 2.0 - 1.0
                        nagent_pos = torch.from_numpy(pos).to(torch.float32).unsqueeze(axis=0).unsqueeze(axis=0).to(device)

                        # generate conditioning vector
                        obs_features = torch.cat([pointcloud_features, nagent_pos, goal_features],dim=-1)

                        # concatenate vision feature and low-dim obs
                        # obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                        obs_cond = obs_features.flatten(start_dim=1)

                        # initialize action from Guassian noise NOTE: swapped batch to 1 for testing here
                        noisy_action = torch.randn(
                            (1, pred_horizon, action_dim), device=device)
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
                        # (B, pred_horizon, action_dim)
                        naction = naction[0]
                        print("\n\n\nNorm Action Prediction: ", naction)
                        action_pred = (naction + 1.0) / 2.0
                        action_pred = action_pred * (a_maxs5d - a_mins5d) + a_mins5d
                        
                        # only take action_horizon number of actions
                        action_horizon = 4
                        start = obs_horizon - 1
                        end = start + action_horizon
                        diff_action = action_pred[start:end,:] # (4, 5)
                        
                        

                        # get the ground truth 5 next actions
                        gt_actions = []
                        norm_actions = []
                        for i in range(action_horizon):
                            action = np.load(test_dataset_path + '/Discrete/Trajectory' + str(t) + '/action' + str(i+s_idx) + '.npy')
                            gt_actions.append(action)

                            # normalize and scale action
                            a = (action - a_mins5d) / (a_maxs5d - a_mins5d)
                            a = a * 2.0 - 1.0
                            norm_actions.append(a)
                        
                        print("\nGround Truth Norm Actions: ", np.array(norm_actions))
                        print("\n\nAction Sequence Prediction: ", diff_action)
                        print("\nGround Truth Actions: ", np.array(gt_actions))
                print("\n\n\n\n\n")

        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_nets = nets
ema.copy_to(ema_nets.parameters())