import torch
import torch.nn as nn
import torch.nn.functional as F
from perception import EncodeLinear, FeedForward
import mate
from filter import env_base_filter as eb_f
import numpy as np 
import time
import math
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, context):
        attn_output, _ = self.attn(x, context, context)
        return self.norm(x + attn_output)  # Residual Connection + LayerNorm

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)  # LayerNorm sau Multi-Head Attention
        self.norm2 = nn.LayerNorm(embed_dim)  # LayerNorm sau FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    def __init__(self, init_embed_dim=32, final_embed_dim=128, init_num_heads=2, num_heads=8, 
                 init_ff_dim=64, final_ff_dim=512, num_layers=1, 
                 num_timesteps=30, steps_per_segment=5, 
                 num_targets=4, target_features=4, 
                 num_cameras=4, camera_features=13, dropout=0.1):
        super().__init__()
        self.target_features = target_features
        self.camera_features = camera_features
        self.dropout = dropout  # Thêm thuộc tính dropout

        # Targets
        self.target_projection = nn.Linear(target_features, init_embed_dim)  # 4 -> 32
        self.target_segment_attention = nn.MultiheadAttention(init_embed_dim, init_num_heads, batch_first=True)
        self.target_segment_ff = nn.Sequential(
            nn.Linear(steps_per_segment * init_embed_dim, init_ff_dim),  # 5 * 32 = 160 -> 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(init_ff_dim, init_ff_dim)  # 64 -> 64
        )
        self.target_global_attention = nn.MultiheadAttention(init_ff_dim, init_num_heads, batch_first=True)
        self.target_global_ff = nn.Sequential(
            nn.Linear((num_timesteps // steps_per_segment) * init_ff_dim, final_embed_dim),  # 6 * 64 = 384 -> 128
            nn.Dropout(dropout)
        )
        
        # Cameras
        self.camera_projection = nn.Linear(camera_features, init_embed_dim)  # 13 -> 32
        self.camera_segment_attention = nn.MultiheadAttention(init_embed_dim, init_num_heads, batch_first=True)
        self.camera_segment_ff = nn.Sequential(
            nn.Linear(steps_per_segment * init_embed_dim, init_ff_dim),  # 5 * 32 = 160 -> 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(init_ff_dim, init_ff_dim)  # 64 -> 64
        )
        self.camera_global_attention = nn.MultiheadAttention(init_ff_dim, init_num_heads, batch_first=True)
        self.camera_global_ff = nn.Sequential(
            nn.Linear((num_timesteps // steps_per_segment) * init_ff_dim, final_embed_dim),  # 6 * 64 = 384 -> 128
            nn.Dropout(dropout)
        )
        
        # Other components
        self.layers = nn.ModuleList([
            TransformerLayer(final_embed_dim, num_heads, final_ff_dim, dropout) for _ in range(num_layers)
        ])
        self.prediction_head = nn.Sequential(
            nn.Linear(final_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5),
        )
        self.encoder_target = EncodeLinear(final_embed_dim, final_embed_dim)
        self.encoder_camera = EncodeLinear(final_embed_dim, final_embed_dim)
        self.encoder_obstacle = EncodeLinear(3, final_embed_dim)
        self.encode_env = EncodeLinear(12, final_embed_dim)
        
        self.num_timesteps = num_timesteps
        self.steps_per_segment = steps_per_segment
        self.init_embed_dim = init_embed_dim
        self.num_targets = num_targets 
        self.num_cameras = num_cameras
        self.init_ff_dim = init_ff_dim

        env = mate.make('MATE-4v4-0-v0')
        env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
        env_base = env.reset()
        self.env_base = eb_f.collected_infos(env_base)  # Giả sử eb_f là module bạn định nghĩa

    def get_sinusoidal_pos_encoding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, targets, obstacles, cameras):
        # --- Xử lý targets ---
        batch_size, num_targets, target_flat_dim = targets.shape  # [batch_size, num_targets, features]

        targets_reshaped = targets.view(batch_size, num_targets, self.num_timesteps, self.target_features)  # [batch_size, num_targets, num_steps, features]
        targets_projected = self.target_projection(targets_reshaped)  # [batch_size, num_targets, num_steps, embed_dim]
        
        pos_encoding = self.get_sinusoidal_pos_encoding(self.num_timesteps, self.init_embed_dim, targets.device)
        pos_encoding = pos_encoding.expand(batch_size, num_targets, self.num_timesteps, self.init_embed_dim)
        targets_with_pos = targets_projected + pos_encoding # [batch_size, nume_targets, num_steps, embed_dim]

        num_segments = self.num_timesteps // self.steps_per_segment  # 5
        targets_segments = targets_with_pos.view(batch_size, num_targets, num_segments, self.steps_per_segment, self.init_embed_dim) # [batch_size, num_targets, num_segments, steps_per_segment, embed_dim]
        targets_flat = targets_segments.view(batch_size * num_targets * num_segments, self.steps_per_segment, self.init_embed_dim) # [batch_size * num targets * num_segments, steps_per_segment, embed_dim]
        target_segment_attn_out, _ = self.target_segment_attention(targets_flat, targets_flat, targets_flat) # [batch_size * num targets * num_segments, steps_per_segment, embed_dim]
        
        target_segment_flat = target_segment_attn_out.contiguous().view(batch_size * num_targets * num_segments, self.steps_per_segment * self.init_embed_dim) # [batch_size * num targets * num_segments, steps_per_segment * embed_dim]
        target_segment_out = self.target_segment_ff(target_segment_flat)
        target_segment_out = target_segment_out.view(batch_size * num_targets, num_segments, self.init_ff_dim)

        target_global_attn_out, _ = self.target_global_attention(target_segment_out, target_segment_out, target_segment_out)
        target_global_attn_out = target_global_attn_out.contiguous().view(batch_size, num_targets, num_segments, self.init_ff_dim)
        target_global_flat = target_global_attn_out.view(batch_size, num_targets, num_segments * self.init_ff_dim)
        targets_final = self.target_global_ff(target_global_flat)  # (batch_size, 8, 128)
        targets_embedded = self.encoder_target(targets_final)

        # --- Xử lý cameras ---
        _, num_cameras, camera_flat_dim = cameras.shape

        cameras_reshaped = cameras.view(batch_size, num_cameras, self.num_timesteps, self.camera_features)  # (batch_size, 4, 25, 13)
        cameras_projected = self.camera_projection(cameras_reshaped)  # (batch_size, 4, 25, 64)

        pos_encoding_cameras = self.get_sinusoidal_pos_encoding(self.num_timesteps, self.init_embed_dim, cameras.device)
        pos_encoding_cameras = pos_encoding_cameras.expand(batch_size, num_cameras, self.num_timesteps, self.init_embed_dim)
        cameras_with_pos = cameras_projected + pos_encoding_cameras  # (batch_size, 4, 25, 64)

        cameras_segments = cameras_with_pos.view(batch_size, num_cameras, num_segments, self.steps_per_segment, self.init_embed_dim)  # (batch_size, 4, 5, 5, 64)
        cameras_flat = cameras_segments.view(batch_size * num_cameras * num_segments, self.steps_per_segment, self.init_embed_dim)  # (batch_size * 4 * 5, 5, 64)
        camera_segment_attn_out, _ = self.camera_segment_attention(cameras_flat, cameras_flat, cameras_flat)
        
        camera_segment_flat = camera_segment_attn_out.contiguous().view(batch_size * num_cameras * num_segments, self.steps_per_segment * self.init_embed_dim)
        camera_segment_out = self.camera_segment_ff(camera_segment_flat)  # (batch_size * 4 * 5, 64)
        camera_segment_out = camera_segment_out.view(batch_size * num_cameras, num_segments, self.init_ff_dim)  # (batch_size * 4, 5, 64)

        camera_global_attn_out, _ = self.camera_global_attention(camera_segment_out, camera_segment_out, camera_segment_out)
        camera_global_attn_out = camera_global_attn_out.contiguous().view(batch_size, num_cameras, num_segments, self.init_ff_dim)
        camera_global_flat = camera_global_attn_out.view(batch_size, num_cameras, num_segments * self.init_ff_dim)
        cameras_final = self.camera_global_ff(camera_global_flat)  # (batch_size, 4, 64)
        cameras_embedded = self.encoder_camera(cameras_final)

        # --- Xử lý obstacles và env ---
        new_env_base = np.tile(self.env_base, (batch_size, 1))
        new_env_base = np.expand_dims(new_env_base, axis=1)
        new_env_base = torch.tensor(new_env_base, dtype=torch.float32)
        new_env_base = self.encode_env(new_env_base)  # (batch_size, 1, 64)

        # Tạo context
        context = torch.cat([targets_embedded, cameras_embedded, new_env_base], dim=1)
        
        # Transformer layers
        targets_out = targets_embedded
        for layer in self.layers:
            targets_out = layer(targets_out, context)
        
        # Dự đoán
        future_states = self.prediction_head(targets_out)
        predicted_labels = future_states.argmax(dim=-1)
        future_states_one_hot = F.one_hot(predicted_labels, num_classes=5).float()

        return future_states, future_states_one_hot
