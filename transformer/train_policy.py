"""
Code adapted from diffusion_policy: https://github.com/real-stanford/diffusion_policy
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from constants import TASK_CONFIGS, HARDWARE_CONFIGS
from pointBERT.tools import builder
from pointBERT.utils.config import cfg_from_yaml_file
from embeddings import EncoderHead
from dataset import ClayDataset
from transformer.policy import TransformerForDiffusion

######### Experiment Setup ##################
CKPT_DIR = TASK_CONFIGS["ckpt_dir"]
TEST_DATASET_PATH = TASK_CONFIGS["test_dataset_path"]
# if CKPT_DIR does not exist, create it
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# Load PointBERT encoder from pretrained weights
device = torch.device("cuda")
config = cfg_from_yaml_file("pointBERT/cfgs/PointTransformer.yaml")
model_config = config.model
pointbert_encoder = builder.model_builder(model_config)
WEIGHTS_PATH = "pointBERT/point-BERT-weights/Point-BERT.pth"
pointbert_encoder.load_model_from_ckpt(WEIGHTS_PATH)
pointbert_encoder.to(device)

# Setup the projection head
ENCODED_DIM = 768
LATENT_DIM = 512
projection_head = EncoderHead(ENCODED_DIM, LATENT_DIM).to(device)

# Define the dataloader
CENTER_ACTIONS = False
dataset = ClayDataset(
    TASK_CONFIGS["dataset_path"],
    TASK_CONFIGS["pred_horizon"],
    TASK_CONFIGS["n_datapoints"],
    TASK_CONFIGS["n_raw_trajectories"],
    CENTER_ACTIONS,
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,  # 64
    num_workers=4,  # 4
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=True,
)

# Save experiment parameters as a dictionary
num_epochs = TASK_CONFIGS["num_epochs"]
exp_params = {
    "exp_name": TASK_CONFIGS["EXP_NAME"],
    "n_datapoints": TASK_CONFIGS["n_datapoints"],
    "n_raw_trajectories": TASK_CONFIGS["n_raw_trajectories"],
    "pred_horizon": TASK_CONFIGS["pred_horizon"],
    "center_actions": CENTER_ACTIONS,
    "n_epochs": num_epochs,
    "dataset": TASK_CONFIGS["dataset_path"],
}
with open(CKPT_DIR + "/experiment_params.txt", "w") as f:
    f.write(str(exp_params))

# Define the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=TASK_CONFIGS["num_diffusion_iters"],
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule="squaredcos_cap_v2",
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type="epsilon",
)

# Define parameters
OBS_DIM = 2 * TASK_CONFIGS["pcl_feature_dim"] + TASK_CONFIGS["lowdim_obs_dim"]
INPUT_DIM = TASK_CONFIGS["action_dim"]
OUTPUT_DIM = INPUT_DIM
OBS_HORIZON = 1
COND_DIM = OBS_DIM * OBS_HORIZON

######### Define the Network ##################
model = TransformerForDiffusion(
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    horizon=TASK_CONFIGS["pred_horizon"],
    n_obs_steps=OBS_HORIZON,
    cond_dim=COND_DIM,
).to(device)

nets = nn.ModuleDict(
    {
        "pointbert_encoder": pointbert_encoder,
        "projection_head": projection_head,
        "noise_pred_net": model,
    }
)

# Exponential Moving Average
ema = EMAModel(parameters=nets.parameters(), power=0.75)

# Standard ADAM optimizer
optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs,
)

######### Training Loop ##################
best_loss = 1e3
with tqdm(range(num_epochs), desc="Epoch") as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
            for nbatch in tepoch:
                pointcloud = nbatch["pointcloud"].to(device).float()
                goalcloud = nbatch["goal"].to(device).float()
                nagent_pos = nbatch["agent_pos"].to(device).unsqueeze(axis=1)
                naction = nbatch["action"].to(device)
                B = nagent_pos.shape[0]

                # embed point cloud
                pointcloud_features = nets["pointbert_encoder"](pointcloud)
                pointcloud_features = nets["projection_head"](pointcloud_features)

                # embed goal cloud
                goalcloud_features = nets["pointbert_encoder"](goalcloud)
                goalcloud_features = nets["projection_head"](goalcloud_features)

                # stack pointcloud features for each obs horizon
                pointcloud_features = pointcloud_features.unsqueeze(1).repeat(
                    1, OBS_HORIZON, 1
                )
                goalcloud_features = goalcloud_features.unsqueeze(1).repeat(
                    1, OBS_HORIZON, 1
                )
                obs_features = torch.cat(
                    [pointcloud_features, nagent_pos, goalcloud_features], dim=-1
                )

                # concatenate vision feature and low-dim obs
                obs_cond = obs_features.flatten(start_dim=1)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (B,), device=device
                ).long()

                # add noise to the clean actions according to the noise magnitude at each diffusion iteration
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                # compute loss
                # predict the noise residual
                noise_pred = nets["noise_pred_net"](noisy_actions, timesteps, obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # Optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # Update Exponential Moving Average of the model weights
                ema.step(nets.parameters())

                # Logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

            # Save the model weights every 50 epochs
            mean_loss = np.mean(epoch_loss)
            if mean_loss < best_loss and epoch_idx % 50 == 0:
                best_loss = mean_loss
                print("\nSaving model weights with avg loss = ", mean_loss)

                # state dict pointbert
                torch.save(
                    nets["pointbert_encoder"].state_dict(),
                    os.path.join(CKPT_DIR, "pointbert_statedict"),
                )

                # projection head
                checkpoint = {"encoder_head": nets["projection_head"]}
                torch.save(
                    checkpoint, os.path.join(CKPT_DIR, "encoder_best_checkpoint")
                )

                # noise_pred_net
                noise_checkpoint = {"noise_pred_net": nets["noise_pred_net"]}
                torch.save(
                    noise_checkpoint,
                    os.path.join(CKPT_DIR, "noise_pred_best_checkpoint"),
                )

            if epoch_idx % 100 == 0:
                print("\n\n\n\n\n----------------- PREDICTIONS -------------------")
                with torch.inference_mode():
                    nets["projection_head"].eval()
                    nets["pointbert_encoder"].eval()
                    nets["noise_pred_net"].eval()

                    trajs = [0, 1]
                    start_state = [0, 0]
                    for traj_idx, traj in enumerate(trajs):
                        s_idx = start_state[traj_idx]

                        # import the state, center and goal
                        ctr = np.load(
                            TEST_DATASET_PATH
                            + "/Discrete/Trajectory"
                            + str(traj)
                            + "/pcl_center"
                            + str(s_idx)
                            + ".npy"
                        )
                        goal = np.load(TEST_DATASET_PATH + "/goal_unnormalized.npy")
                        state = np.load(
                            TEST_DATASET_PATH
                            + "/Discrete/Trajectory"
                            + str(traj)
                            + "/state"
                            + str(s_idx)
                            + ".npy"
                        )

                        # center and scale goal
                        goal = (goal - ctr) * 10.0
                        goal = torch.from_numpy(goal).to(torch.float32)
                        goals = torch.unsqueeze(goal, 0).to(device)
                        tokenized_goals = nets["pointbert_encoder"](goals)
                        goal_embed = nets["projection_head"](tokenized_goals)
                        goal_features = goal_embed.unsqueeze(1).repeat(
                            1, OBS_HORIZON, 1
                        )

                        # center and scale state
                        state = (state - ctr) * 10.0
                        state = torch.from_numpy(state).to(torch.float32)
                        states = torch.unsqueeze(state, 0).to(device)
                        tokenized_states = nets["pointbert_encoder"](states)
                        pcl_embed = nets["projection_head"](tokenized_states)
                        pointcloud_features = pcl_embed.unsqueeze(1).repeat(
                            1, OBS_HORIZON, 1
                        )

                        # get the previous action
                        if s_idx == 0:
                            pos = np.array([0.6, 0.0, 0.165, 0.0, 0.05])
                        else:
                            pos = np.load(
                                TEST_DATASET_PATH
                                + "/Discrete/Trajectory"
                                + str(traj)
                                + "/action"
                                + str(s_idx - 1)
                                + ".npy"
                            )

                        # normalize and scale action
                        a_mins5d = HARDWARE_CONFIGS["a_mins5d"]
                        a_maxs5d = HARDWARE_CONFIGS["a_maxs5d"]
                        pos = (pos - a_mins5d) / (a_maxs5d - a_mins5d)
                        pos = pos * 2.0 - 1.0
                        nagent_pos = (
                            torch.from_numpy(pos)
                            .to(torch.float32)
                            .unsqueeze(axis=0)
                            .unsqueeze(axis=0)
                            .to(device)
                        )

                        # generate conditioning vector
                        obs_features = torch.cat(
                            [pointcloud_features, nagent_pos, goal_features], dim=-1
                        )

                        # concatenate vision feature and low-dim obs
                        # obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                        obs_cond = obs_features.flatten(start_dim=1)

                        # initialize action from Guassian noise NOTE: swapped batch to 1 for testing here
                        noisy_action = torch.randn(
                            (
                                1,
                                TASK_CONFIGS["pred_horizon"],
                                TASK_CONFIGS["action_dim"],
                            ),
                            device=device,
                        )
                        naction = noisy_action

                        # init scheduler
                        noise_scheduler.set_timesteps(
                            TASK_CONFIGS["num_diffusion_iters"]
                        )

                        for t in noise_scheduler.timesteps:
                            # predict noise
                            noise_pred = nets["noise_pred_net"](
                                sample=naction, timestep=t, global_cond=obs_cond
                            )

                            # inverse diffusion step (remove noise)
                            naction = noise_scheduler.step(
                                model_output=noise_pred, timestep=t, sample=naction
                            ).prev_sample

                        # unnormalize action
                        naction = naction.detach().to("cpu").numpy()
                        # (B, pred_horizon, action_dim)
                        naction = naction[0]
                        print("\n\n\nNorm Action Prediction: ", naction)
                        action_pred = (naction + 1.0) / 2.0
                        action_pred = action_pred * (a_maxs5d - a_mins5d) + a_mins5d

                        # only take action_horizon number of actions
                        action_horizon = TASK_CONFIGS["action_horizon"]
                        start = OBS_HORIZON - 1
                        end = start + action_horizon
                        diff_action = action_pred[start:end, :]  # (4, 5)

                        # get the ground truth 5 next actions
                        gt_actions = []
                        norm_actions = []
                        for i in range(action_horizon):
                            action = np.load(
                                TEST_DATASET_PATH
                                + "/Discrete/Trajectory"
                                + str(traj)
                                + "/action"
                                + str(i + s_idx)
                                + ".npy"
                            )
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
