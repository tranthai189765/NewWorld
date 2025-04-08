import torch
import torch.nn as nn
import torch.nn.functional as F
from perception import EncodeLinear
import mate
from filter import env_base_filter as eb_f
import numpy as np 

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.attn(query, key_value, key_value)
        return self.norm(query + attn_output)  # Residual + LayerNorm

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads):
        super().__init__()
        self.attn_cam = CrossAttention(embed_dim, num_heads)
        self.attn_target = CrossAttention(embed_dim, num_heads)
        self.attn_env = CrossAttention(embed_dim, num_heads)

        self.fusion = nn.Linear(embed_dim * 3, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, targets, cameras, env_base):
        out_cam = self.attn_cam(targets, cameras)
        out_target = self.attn_target(targets, targets)
        out_env = self.attn_env(targets, env_base)

        fused = self.fusion(torch.cat([out_cam, out_target, out_env], dim=-1))
        fused = self.ffn(fused)
        return self.norm(targets + fused)  # Residual + LayerNorm
    
class WorldModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, ff_dim=256, num_layers=3):
        super().__init__()
        # Encoders
        self.encoder_camera = EncodeLinear(45, embed_dim)
        self.encoder_target = EncodeLinear(20, embed_dim)
        self.encoder_obstacle = EncodeLinear(3, embed_dim)
        self.encode_env = EncodeLinear(12, embed_dim)

        # Multi-layer cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, ff_dim, num_heads) for _ in range(num_layers)
        ])

        env = mate.make('MATE-4v4-0-v0')
        env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
        env_base = env.reset()
        self.env_base = eb_f.collected_infos(env_base)

    def forward(self, targets, cameras):
        cameras = self.encoder_camera(cameras)
        targets = self.encoder_target(targets)

        batch_size = targets.shape[0]
        new_env_base = np.tile(self.env_base, (batch_size, 1))
        new_env_base = np.expand_dims(new_env_base, axis=1)
        new_env_base = self.encode_env(new_env_base)

        # Apply stacked cross-attention layers
        for layer in self.layers:
            targets = layer(targets, cameras, new_env_base)

        # Predict future goal
        future_states = self.prediction_head(targets)

        predicted_labels = future_states.argmax(dim=-1)
        future_states_one_hot = F.one_hot(predicted_labels, num_classes=5).float()
        return future_states, future_states_one_hot
