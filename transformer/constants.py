import numpy as np

EXP_NAME = "Line_correctpointbert"
TARGET_SHAPE = "Line"  # Choose from ["Line", "X", "Cone", or "All_Shapes"]
TASK_CONFIGS = {
    "ckpt_dir": "checkpoints/" + EXP_NAME,
    "dataset_path": "ClayDemoDataset/" + str(TARGET_SHAPE) + "/Train",
    "test_dataset_path": "ClayDemoDataset/" + str(TARGET_SHAPE) + "/Test",
    "n_datapoints": 2880,  # the desired numer of datapoints after augmentation
    "n_raw_trajectories": 7,  # the number of raw datapoints
    "pred_horizon": 4,
    "num_epochs": 750,
    "num_diffusion_iters": 100,
    "pcl_feature_dim": 512,
    "lowdim_obs_dim": 5,
    "action_dim": 5,
    "action_horizon": 4,
}
HARDWARE_CONFIGS = {
    "a_mins5d": np.array([0.56, -0.062, 0.125, -90, 0.005]),
    "a_maxs5d": np.array([0.7, 0.062, 0.165, 90, 0.05]),
}
