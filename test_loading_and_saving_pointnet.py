import torch
from pointnet.models.model_dp3_pytorch import PointNetEncoderXYZ

ckpt_dir = '/home/alison/Documents/GitHub/SculptDiff/checkpoints/pointnet_16pred_7datasetfixed_with_augs'

# initialize the pointnet model
pointnet_encoder_state_dict = torch.load(ckpt_dir + '/pointnet_best_checkpoint.zip', map_location=torch.device('cpu'))
print("pointnet encoder: ", pointnet_encoder_state_dict.keys())
pointnet_encoder = pointnet_encoder_state_dict['encoder']
pointnet_encoder.eval()
print("Success!")

# device = torch.device('cuda')
# pointnet_encoder = PointNetEncoderXYZ().to(device)
# state_dict = torch.load(ckpt_dir + '/pointnet_encoder_best_checkpoint', map_location=torch.device('cpu'))
# pointnet_encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')})
# pointnet_encoder.eval()
# print("Success!")
