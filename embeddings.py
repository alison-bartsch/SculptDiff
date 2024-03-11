import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderHead(nn.Module):
    def __init__(self, encoded_dim, latent_dim):
        super(EncoderHead, self).__init__()
        self.encoded_dim = encoded_dim
        self.latent_dim = latent_dim
        
        self.encoder_head = nn.Sequential(
            nn.Linear(self.encoded_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def forward(self, encoded_pcl):
        # concatentation strategy from pointtransformer for downstream classification tasks
        x = torch.cat([encoded_pcl[:,0], encoded_pcl[:, 1:].max(1)[0]], dim = -1) 
        latent_state = self.encoder_head(x)
        return latent_state