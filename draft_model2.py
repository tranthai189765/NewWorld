import torch
import torch.nn as nn
import torch.nn.functional as F
from perception import EncodeLinear, FeedForward
import mate
from filter import env_base_filter as eb_f
import numpy as np 
import time
import math

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):  # Tăng dropout lên 0.3 để giảm overfitting
        super().__init__()
        # Cross-attention từ cameras sang targets
        self.cross_attn_targets = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Self-attention cho targets
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Cross-attention từ targets sang cameras
        self.cross_attn_cameras = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Cross-attention từ targets sang env_base
        self.cross_attn_env = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # LayerNorm
        self.norm0 = nn.LayerNorm(embed_dim)  # Sau cross-attention từ cameras sang targets
        self.norm1 = nn.LayerNorm(embed_dim)  # Sau self-attention
        self.norm2 = nn.LayerNorm(embed_dim)  # Sau cross-attention từ targets sang cameras
        self.norm3 = nn.LayerNorm(embed_dim)  # Sau cross-attention từ targets sang env_base
        self.norm4 = nn.LayerNorm(embed_dim)  # Sau FFN
        # Feed Forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, targets, cameras, env_base):
        # 1. Cross-attention từ cameras sang targets
        cross_targets_out, _ = self.cross_attn_targets(cameras, targets, targets)
        cameras = self.norm0(cameras + self.dropout(cross_targets_out))  # Cập nhật cameras

        # 2. Self-attention trên targets
        self_attn_out, _ = self.self_attn(targets, targets, targets)
        targets = self.norm1(targets + self.dropout(self_attn_out))

        # 3. Cross-attention từ targets sang cameras (dùng cameras đã cập nhật)
        cross_cameras_out, _ = self.cross_attn_cameras(targets, cameras, cameras)
        targets = self.norm2(targets + self.dropout(cross_cameras_out))

        # 4. Cross-attention từ targets sang env_base
        cross_env_out, _ = self.cross_attn_env(targets, env_base, env_base)
        targets = self.norm3(targets + self.dropout(cross_env_out))

        # 5. Feed Forward
        ffn_out = self.ffn(targets)
        targets = self.norm4(targets + self.dropout(ffn_out))

        return targets, cameras

class WorldModel(nn.Module):
    def __init__(self, init_embed_dim=32, final_embed_dim=128, init_num_heads=2, num_heads=8, 
                 init_ff_dim=64, final_ff_dim=512, num_layers=1, 
                 num_timesteps=30, steps_per_segment=5, 
                 num_targets=4, target_features=4, 
                 num_cameras=4, camera_features=13, dropout=0.2):  # Tăng dropout lên 0.2
        super().__init__()
        self.target_features = target_features
        self.camera_features = camera_features
        self.dropout = dropout

        # Targets
        self.target_projection = nn.Linear(target_features, init_embed_dim)
        self.target_segment_attention = nn.MultiheadAttention(init_embed_dim, init_num_heads, batch_first=True)
        self.target_segment_ff = nn.Sequential(
            nn.Linear(steps_per_segment * init_embed_dim, init_ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(init_ff_dim, init_ff_dim)
        )
        self.target_global_attention = nn.MultiheadAttention(init_ff_dim, init_num_heads, batch_first=True)
        self.target_global_ff = nn.Sequential(
            nn.Linear((num_timesteps // steps_per_segment) * init_ff_dim, final_embed_dim),
            nn.Dropout(dropout)
        )
        
        # Cameras
        self.camera_projection = nn.Linear(camera_features, init_embed_dim)
        self.camera_segment_attention = nn.MultiheadAttention(init_embed_dim, init_num_heads, batch_first=True)
        self.camera_segment_ff = nn.Sequential(
            nn.Linear(steps_per_segment * init_embed_dim, init_ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(init_ff_dim, init_ff_dim)
        )
        self.camera_global_attention = nn.MultiheadAttention(init_ff_dim, init_num_heads, batch_first=True)
        self.camera_global_ff = nn.Sequential(
            nn.Linear((num_timesteps // steps_per_segment) * init_ff_dim, final_embed_dim),
            nn.Dropout(dropout)
        )
        
        # Transformer layers
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
        self.env_base = eb_f.collected_infos(env_base)

    def get_sinusoidal_pos_encoding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, targets, obstacles, cameras):
        batch_size, num_targets, target_flat_dim = targets.shape
        targets_reshaped = targets.view(batch_size, num_targets, self.num_timesteps, self.target_features)
        targets_projected = self.target_projection(targets_reshaped)
        pos_encoding = self.get_sinusoidal_pos_encoding(self.num_timesteps, self.init_embed_dim, targets.device)
        pos_encoding = pos_encoding.expand(batch_size, num_targets, self.num_timesteps, self.init_embed_dim)
        targets_with_pos = targets_projected + pos_encoding

        num_segments = self.num_timesteps // self.steps_per_segment
        targets_segments = targets_with_pos.view(batch_size, num_targets, num_segments, self.steps_per_segment, self.init_embed_dim)
        targets_flat = targets_segments.view(batch_size * num_targets * num_segments, self.steps_per_segment, self.init_embed_dim)
        target_segment_attn_out, _ = self.target_segment_attention(targets_flat, targets_flat, targets_flat)
        
        target_segment_flat = target_segment_attn_out.contiguous().view(batch_size * num_targets * num_segments, self.steps_per_segment * self.init_embed_dim)
        target_segment_out = self.target_segment_ff(target_segment_flat)
        target_segment_out = target_segment_out.view(batch_size * num_targets, num_segments, self.init_ff_dim)

        # Thêm positional embedding cho các segment của targets
        pos_encoding_segments = self.get_sinusoidal_pos_encoding(num_segments, self.init_ff_dim, targets.device)
        pos_encoding_segments = pos_encoding_segments.expand(batch_size * num_targets, num_segments, self.init_ff_dim)
        target_segment_out = target_segment_out + pos_encoding_segments

        target_global_attn_out, _ = self.target_global_attention(target_segment_out, target_segment_out, target_segment_out)
        target_global_attn_out = target_global_attn_out.contiguous().view(batch_size, num_targets, num_segments, self.init_ff_dim)
        target_global_flat = target_global_attn_out.view(batch_size, num_targets, num_segments * self.init_ff_dim)
        targets_final = self.target_global_ff(target_global_flat)
        targets_embedded = self.encoder_target(targets_final)

        _, num_cameras, camera_flat_dim = cameras.shape
        cameras_reshaped = cameras.view(batch_size, num_cameras, self.num_timesteps, self.camera_features)
        cameras_projected = self.camera_projection(cameras_reshaped)
        pos_encoding_cameras = self.get_sinusoidal_pos_encoding(self.num_timesteps, self.init_embed_dim, cameras.device)
        pos_encoding_cameras = pos_encoding_cameras.expand(batch_size, num_cameras, self.num_timesteps, self.init_embed_dim)
        cameras_with_pos = cameras_projected + pos_encoding_cameras

        cameras_segments = cameras_with_pos.view(batch_size, num_cameras, num_segments, self.steps_per_segment, self.init_embed_dim)
        cameras_flat = cameras_segments.view(batch_size * num_cameras * num_segments, self.steps_per_segment, self.init_embed_dim)
        camera_segment_attn_out, _ = self.camera_segment_attention(cameras_flat, cameras_flat, cameras_flat)
        
        camera_segment_flat = camera_segment_attn_out.contiguous().view(batch_size * num_cameras * num_segments, self.steps_per_segment * self.init_embed_dim)
        camera_segment_out = self.camera_segment_ff(camera_segment_flat)
        camera_segment_out = camera_segment_out.view(batch_size * num_cameras, num_segments, self.init_ff_dim)

        # Thêm positional embedding cho các segment của cameras
        pos_encoding_segments_cameras = self.get_sinusoidal_pos_encoding(num_segments, self.init_ff_dim, cameras.device)
        pos_encoding_segments_cameras = pos_encoding_segments_cameras.expand(batch_size * num_cameras, num_segments, self.init_ff_dim)
        camera_segment_out = camera_segment_out + pos_encoding_segments_cameras

        camera_global_attn_out, _ = self.camera_global_attention(camera_segment_out, camera_segment_out, camera_segment_out)
        camera_global_attn_out = camera_global_attn_out.contiguous().view(batch_size, num_cameras, num_segments, self.init_ff_dim)
        camera_global_flat = camera_global_attn_out.view(batch_size, num_cameras, num_segments * self.init_ff_dim)
        cameras_final = self.camera_global_ff(camera_global_flat)
        cameras_embedded = self.encoder_camera(cameras_final)

        # Xử lý obstacles và env
        # obstacles_embedded = self.encoder_obstacle(obstacles)
        new_env_base = np.tile(self.env_base, (batch_size, 1))
        new_env_base = np.expand_dims(new_env_base, axis=1)
        new_env_base = torch.tensor(new_env_base, dtype=torch.float32, device=targets.device)
        new_env_base = self.encode_env(new_env_base)

        # Transformer layers: Chỉ truyền targets và cameras, không cần context đầy đủ
        targets_out = targets_embedded
        cameras_out = cameras_embedded
        for layer in self.layers:
            targets_out, cameras_out = layer(targets_out, cameras_out, new_env_base)  # Self-attention trên targets, cross-attention với cameras
        
        # Dự đoán
        future_states = self.prediction_head(targets_out)
        predicted_labels = future_states.argmax(dim=-1)
        future_states_one_hot = F.one_hot(predicted_labels, num_classes=5).float()

        return future_states, future_states_one_hot
