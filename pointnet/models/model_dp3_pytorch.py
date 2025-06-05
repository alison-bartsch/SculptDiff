import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class PointNetEncoderXYZ(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs):
        super().__init__()
        block_channel = [64, 128, 256]

        
        assert in_channels == 3, print(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()

    def forward(self, x):
        end_points = {}
        x = self.mlp(x).contiguous()
        end_points['point_features'] = x
        x = torch.max(x, 1)[0]
        embedding = self.final_projection(x)
        end_points['embedding'] = embedding
        return embedding, end_points

# FC Decoder
class PointNetFCDecoder(nn.Module):
    def __init__(self, num_point=1024, latent_dim=1024):
        super(PointNetFCDecoder, self).__init__()
        self.num_point = num_point
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(1024, num_point * 3)

    def forward(self, global_feat):
        net = self.fc1(global_feat)
        net = self.fc2(net)
        net = self.fc3(net)
        return net.reshape(global_feat.shape[0], self.num_point, 3)

# Combined AE
class PointNetAE(nn.Module):
    def __init__(self, num_point=1024, point_dim=3, latent_dim=1024):
        super(PointNetAE, self).__init__()
        self.encoder = PointNetEncoderXYZ(in_channels=point_dim, out_channels=latent_dim)
        self.decoder = PointNetFCDecoder(num_point, latent_dim)

    def forward(self, point_cloud):
        global_feat, end_points = self.encoder(point_cloud)
        reconstructed = self.decoder(global_feat)
        return reconstructed, end_points

# Chamfer distance
def chamfer_distance(pc1, pc2):
    pc1 = pc1.unsqueeze(2)
    pc2 = pc2.unsqueeze(1)
    dist = torch.sum((pc1 - pc2) ** 2, dim=-1)
    dist1 = torch.min(dist, dim=2)[0]
    dist2 = torch.min(dist, dim=1)[0]
    return torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

# Loss
def get_loss(pred, label, end_points=None):
    if end_points is None:
        end_points = {}
    chamfer_loss = chamfer_distance(pred, label)
    loss = torch.mean(chamfer_loss)
    end_points['pcloss'] = loss
    return loss * 100, end_points

# Test
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.rand((32, 1024, 3)).to(device)
    
    model = PointNetAE(num_point=1024).to(device)
    outputs, end_points = model(inputs)
    print(f"Output shape: {outputs.shape}")
    print(f"Embedding shape: {end_points['embedding'].shape}")
    
    loss, _ = get_loss(outputs, torch.rand((32, 1024, 3)).to(device), end_points)
    print(f"Loss: {loss.item()}")
