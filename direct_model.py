import torch
import torch.nn as nn
import torch.nn.functional as F
from perception import EncodeLinear
import mate
from filter import env_base_filter as eb_f
import numpy as np 
import time
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, context):
        attn_output, _ = self.attn(x, context, context)
        return self.norm(x + attn_output)  # Residual Connection + LayerNorm

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)  # LayerNorm sau Multi-Head Attention
        self.norm2 = nn.LayerNorm(embed_dim)  # LayerNorm sau FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
    
    def forward(self, x, context):
        # Multi-Head Attention + Residual + LayerNorm
        attn_out = self.attn(x, context)
        x = self.norm1(x + attn_out)

        # Feed Forward + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class WorldModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, ff_dim=256, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        self.encoder_camera = EncodeLinear(45, embed_dim)
        self.encoder_target = EncodeLinear(20, embed_dim)
        self.encoder_obstacle = EncodeLinear(3, embed_dim)
        self.encode_env = EncodeLinear(12, embed_dim)
        env =  mate.make('MATE-4v4-0-v0')
        env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
        env_base = env.reset()
        self.env_base = eb_f.collected_infos(env_base)


    
    def forward(self, targets, obstacles, cameras):
        cameras = self.encoder_camera(cameras)
        targets = self.encoder_target(targets)
        batch_size = targets.shape[0]
        new_env_base = np.tile(self.env_base, (batch_size, 1))  # (batch_size, 32)
        new_env_base = np.expand_dims(new_env_base, axis=1)  # Thêm 1 chiều: (batch_size, 1, 32)
        # print("new_env_base = ",new_env_base )
        new_env_base = self.encode_env(new_env_base)
        # print("env = ", self.env_base)

        context = torch.cat([targets, cameras, new_env_base], dim=1)  # Targets học từ tất cả
        #print(context.shape)
        for layer in self.layers:
            targets = layer(targets, context)  # Targets học từ tất cả đối tượng
        future_states = self.prediction_head(targets)  # Dự đoán state tương lai

           # Lấy nhãn dự đoán (0–4) và chuyển thành one-hot
        predicted_labels = future_states.argmax(dim=-1)  # (batch_size, num_targets)
        future_states_one_hot = F.one_hot(predicted_labels, num_classes=5).float()  # (batch_size, num_targets, 5)
        return future_states, future_states_one_hot
