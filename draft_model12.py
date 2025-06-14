import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import mate
from perception import EncodeLinear, FeedForward, SuperEncodeLinear
from filter import env_base_filter as eb_f

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.01):
        super().__init__()
        self.cross_attn_targets = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_cameras = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_env = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm0 = nn.LayerNorm(embed_dim)
        # self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, targets, cameras, env_base):
        cross_targets_out, _ = self.cross_attn_targets(cameras, targets, targets)
        cameras = self.norm0(cameras + self.dropout(cross_targets_out))
        # self_attn_out, _ = self.self_attn(targets, targets, targets)
        # targets = self.norm1(targets + self.dropout(self_attn_out))
        cross_cameras_out, _ = self.cross_attn_cameras(targets, cameras, cameras)
        targets = self.norm2(targets + self.dropout(cross_cameras_out))
        cross_env_out, _ = self.cross_attn_env(targets, env_base, env_base)
        targets = self.norm3(targets + self.dropout(cross_env_out))
        ffn_out = self.ffn(targets)
        targets = self.norm4(targets + self.dropout(ffn_out))
        return targets, cameras

class WorldModel(nn.Module):
    def __init__(self, init_embed_dim=64, final_embed_dim=256, init_num_heads=2, num_heads=8, 
                 init_ff_dim=64, final_ff_dim=512, num_layers=3, 
                 num_timesteps=100, steps_per_segment=5, 
                 num_targets=8, target_features=8, 
                 num_cameras=4, camera_features=17, dropout=0.3):
        super().__init__()
        self.target_features = target_features
        self.camera_features = camera_features
        self.dropout = dropout
        self.num_segments = num_timesteps // steps_per_segment
        self.num_timesteps = num_timesteps
        self.num_targets = num_targets
        self.steps_per_segment = steps_per_segment
        self.init_embed_dim = init_embed_dim
        self.num_cameras = num_cameras
        self.init_ff_dim = init_ff_dim

        # CLS tokens
        self.target_cls_token = nn.Parameter(torch.zeros(1, 1, init_embed_dim))
        self.camera_cls_token = nn.Parameter(torch.zeros(1, 1, init_embed_dim))

        # Projections
        self.target_projection = nn.Linear(target_features, init_embed_dim)
        self.camera_projection = nn.Linear(camera_features, init_embed_dim)

        # Segment attention
        self.target_segment_attention = nn.MultiheadAttention(init_embed_dim, init_num_heads, batch_first=True)
        self.camera_segment_attention = nn.MultiheadAttention(init_embed_dim, init_num_heads, batch_first=True)

        # CLS attention
        self.target_cls_attention = nn.MultiheadAttention(init_embed_dim, init_num_heads, batch_first=True)
        self.camera_cls_attention = nn.MultiheadAttention(init_embed_dim, init_num_heads, batch_first=True)

        # Layer Normalization
        self.target_segment_norm = nn.LayerNorm(init_embed_dim)
        self.camera_segment_norm = nn.LayerNorm(init_embed_dim)
        self.target_cls_norm = nn.LayerNorm(init_embed_dim)
        self.camera_cls_norm = nn.LayerNorm(init_embed_dim)
        self.prediction_norm = nn.LayerNorm(final_embed_dim)

        # Attention pooling
        self.target_pooling = nn.Linear(init_embed_dim, 1)
        self.camera_pooling = nn.Linear(init_embed_dim, 1)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(final_embed_dim, num_heads, final_ff_dim, dropout) for _ in range(num_layers)
        ])

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(final_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5),
        )

        # Encoders
        self.encoder_target = SuperEncodeLinear(init_embed_dim, final_embed_dim)
        self.encoder_camera = SuperEncodeLinear(init_embed_dim, final_embed_dim)
        self.encoder_obstacle = SuperEncodeLinear(3, final_embed_dim)
        self.encode_env = SuperEncodeLinear(12, final_embed_dim)

        # Environment
        env = mate.make('MATE-4v8-9-v0')
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
        _, num_cameras, camera_flat_dim = cameras.shape
        targets_reshaped = targets.view(batch_size, num_targets, self.num_timesteps, self.target_features)
        cameras_reshaped = cameras.view(batch_size, self.num_cameras, self.num_timesteps, self.camera_features)

        # Environment
        new_env_base = np.tile(self.env_base, (batch_size, 1))
        new_env_base = np.expand_dims(new_env_base, axis=1)
        new_env_base = torch.tensor(new_env_base, dtype=torch.float32, device=targets.device)
        new_env_base = self.encode_env(new_env_base)

        # Targets: Projection
        targets_projected = self.target_projection(targets_reshaped)
        targets_segments = targets_projected.view(batch_size, num_targets, self.num_segments, self.steps_per_segment, self.init_embed_dim)

        # Add CLS token for targets
        cls_tokens = self.target_cls_token.expand(batch_size, num_targets, self.num_segments, 1, self.init_embed_dim)
        targets_with_cls = torch.cat([cls_tokens, targets_segments], dim=3)  # [batch_size, num_targets, num_segments, steps_per_segment+1=6, init_embed_dim]

        # Positional encoding for targets (CLS + timesteps)
        pos_encoding = self.get_sinusoidal_pos_encoding(self.steps_per_segment + 1, self.init_embed_dim, targets.device)
        pos_encoding = pos_encoding.expand(batch_size, num_targets, self.num_segments, -1, -1)
        targets_with_pos = targets_with_cls + pos_encoding

        # Target segment attention
        targets_flat = targets_with_pos.view(batch_size * num_targets * self.num_segments, self.steps_per_segment + 1, self.init_embed_dim)
        target_segment_attn_out, _ = self.target_segment_attention(targets_flat, targets_flat, targets_flat)
        # Extract CLS and apply LayerNorm
        target_cls = target_segment_attn_out[:, 0:1, :].contiguous()  # [batch_size * num_targets * num_segments, 1, init_embed_dim]
        target_cls = target_cls.view(batch_size * num_targets * self.num_segments, self.init_embed_dim)
        target_cls = self.target_segment_norm(target_cls).view(batch_size * num_targets, self.num_segments, self.init_embed_dim)

        # Positional encoding for CLS
        pos_encoding_cls = self.get_sinusoidal_pos_encoding(self.num_segments, self.init_embed_dim, targets.device)
        pos_encoding_cls = pos_encoding_cls.expand(batch_size * num_targets, self.num_segments, self.init_embed_dim)
        target_cls = target_cls + pos_encoding_cls

        # CLS attention for targets
        target_cls_attn_out, _ = self.target_cls_attention(target_cls, target_cls, target_cls)
        # Attention pooling
        target_scores = self.target_pooling(target_cls_attn_out).squeeze(-1)  # [batch_size * num_targets, num_segments]
        target_weights = F.softmax(target_scores, dim=1).unsqueeze(-1)  # [batch_size * num_targets, num_segments, 1]
        targets_final = (target_cls_attn_out * target_weights).sum(dim=1)  # [batch_size * num_targets, init_embed_dim]
        targets_final = targets_final.view(batch_size, num_targets, self.init_embed_dim)
        targets_final = self.target_cls_norm(targets_final)
        targets_embedded = self.encoder_target(targets_final)

        # Cameras: Projection
        cameras_projected = self.camera_projection(cameras_reshaped)
        cameras_segments = cameras_projected.view(batch_size, self.num_cameras, self.num_segments, self.steps_per_segment, self.init_embed_dim)

        # Add CLS token for cameras
        cls_tokens_cameras = self.camera_cls_token.expand(batch_size, self.num_cameras, self.num_segments, 1, self.init_embed_dim)
        cameras_with_cls = torch.cat([cls_tokens_cameras, cameras_segments], dim=3)

        # Positional encoding for cameras (CLS + timesteps)
        pos_encoding_cameras = self.get_sinusoidal_pos_encoding(self.steps_per_segment + 1, self.init_embed_dim, cameras.device)
        pos_encoding_cameras = pos_encoding_cameras.expand(batch_size, self.num_cameras, self.num_segments, -1, -1)
        cameras_with_pos = cameras_with_cls + pos_encoding_cameras

        # Camera segment attention
        cameras_flat = cameras_with_pos.view(batch_size * self.num_cameras * self.num_segments, self.steps_per_segment + 1, self.init_embed_dim)
        camera_segment_attn_out, _ = self.camera_segment_attention(cameras_flat, cameras_flat, cameras_flat)
        camera_cls = camera_segment_attn_out[:, 0:1, :].contiguous()
        camera_cls = camera_cls.view(batch_size * self.num_cameras * self.num_segments, self.init_embed_dim)
        camera_cls = self.camera_segment_norm(camera_cls).view(batch_size * num_cameras, self.num_segments, self.init_embed_dim)

        # Positional encoding for CLS
        pos_encoding_cls_cameras = self.get_sinusoidal_pos_encoding(self.num_segments, self.init_embed_dim, cameras.device)
        pos_encoding_cls_cameras = pos_encoding_cls_cameras.expand(batch_size * self.num_cameras, self.num_segments, self.init_embed_dim)
        camera_cls = camera_cls + pos_encoding_cls_cameras

        # CLS attention for cameras
        camera_cls_attn_out, _ = self.camera_cls_attention(camera_cls, camera_cls, camera_cls)
        # Attention pooling
        camera_scores = self.camera_pooling(camera_cls_attn_out).squeeze(-1)  # [batch_size * num_cameras, num_segments]
        camera_weights = F.softmax(camera_scores, dim=1).unsqueeze(-1)  # [batch_size * num_cameras, num_segments, 1]
        cameras_final = (camera_cls_attn_out * camera_weights).sum(dim=1)  # [batch_size * num_cameras, init_embed_dim]
        cameras_final = cameras_final.view(batch_size, self.num_cameras, self.init_embed_dim)
        cameras_final = self.camera_cls_norm(cameras_final)
        cameras_embedded = self.encoder_camera(cameras_final)

        # Obstacles and Env
        # obstacles_embedded = self.encoder_obstacle(obstacles)
        targets_out = targets_embedded
        cameras_out = cameras_embedded
        for layer in self.layers:
            targets_out, cameras_out = layer(targets_out, cameras_out, new_env_base)
        
        # Prediction
        targets_out = self.prediction_norm(targets_out)
        future_states = self.prediction_head(targets_out)
        predicted_labels = future_states.argmax(dim=-1)
        future_states_one_hot = F.one_hot(predicted_labels, num_classes=5).float()

        return future_states, future_states_one_hot
