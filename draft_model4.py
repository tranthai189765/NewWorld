import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import mate
from perception import EncodeLinear, FeedForward
from filter import env_base_filter as eb_f
import random

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, targets, cameras, env_base, obstacles):
        batch_size = targets.shape[0]
        num_targets = targets.shape[1]
        num_cameras = cameras.shape[1]
        num_env = env_base.shape[1]
        num_obstacles = obstacles.shape[1]
        
        combined = torch.cat([targets, cameras, env_base, obstacles], dim=1)
        attn_out, _ = self.self_attn(combined, combined, combined)
        combined = self.norm1(combined + self.dropout(attn_out))
        ffn_out = self.ffn(combined)
        combined = self.norm2(combined + self.dropout(ffn_out))
        
        targets_out = combined[:, :num_targets, :].contiguous()
        cameras_out = combined[:, num_targets:num_targets+num_cameras, :].contiguous()
        env_base_out = combined[:, num_targets+num_cameras:num_targets+num_cameras+num_env, :].contiguous()
        obstacles_out = combined[:, num_targets+num_cameras+num_env:, :].contiguous()
        
        return targets_out, cameras_out, env_base_out, obstacles_out

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, targets_memory, cameras_memory, env_memory, obstacles_memory, tgt_mask=None):
        batch_size = tgt.shape[0] // targets_memory.shape[0]
        num_targets = tgt.shape[0] // batch_size
        num_cameras = cameras_memory.shape[0] // batch_size
        num_env = env_memory.shape[0] // batch_size
        num_obstacles = obstacles_memory.shape[0] // batch_size
        
        targets_memory = targets_memory.repeat_interleave(num_targets, dim=0)
        cameras_memory = cameras_memory.repeat_interleave(num_targets, dim=0)
        env_memory = env_memory.repeat_interleave(num_targets, dim=0)
        obstacles_memory = obstacles_memory.repeat_interleave(num_targets, dim=0)
        
        combined = torch.cat([tgt, targets_memory, cameras_memory, env_memory, obstacles_memory], dim=1)
        
        seq_len = tgt.shape[1]
        total_len = combined.shape[1]
        if tgt_mask is not None:
            extended_mask = torch.zeros(seq_len, total_len, device=tgt_mask.device)
            extended_mask[:, :seq_len] = tgt_mask
            attn_out, _ = self.self_attn(combined, combined, combined, attn_mask=extended_mask)
        else:
            attn_out, _ = self.self_attn(combined, combined, combined)
        
        combined = self.norm1(combined + self.dropout(attn_out))
        ffn_out = self.ffn(combined)
        combined = self.norm2(combined + self.dropout(ffn_out))
        
        tgt_out = combined[:, :tgt.shape[1], :].contiguous()
        return tgt_out

class WorldModel(nn.Module):
    def __init__(self, init_embed_dim=32, final_embed_dim=128, init_num_heads=2, num_heads=8, 
                 init_ff_dim=64, final_ff_dim=512, num_layers=1, num_decoder_layers=1,
                 num_timesteps=30, steps_per_segment=5, future_steps=11,
                 num_targets=4, target_features=4, 
                 num_cameras=8, camera_features=13,
                 num_obstacles=9, obstacle_features=3, dropout=0.3):
        super().__init__()
        self.target_features = target_features
        self.camera_features = camera_features
        self.obstacle_features = obstacle_features
        self.dropout = dropout
        self.num_segments = num_timesteps // steps_per_segment
        self.num_timesteps = num_timesteps
        self.future_steps = future_steps
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.steps_per_segment = steps_per_segment
        self.init_embed_dim = init_embed_dim
        self.final_embed_dim = final_embed_dim
        self.num_cameras = num_cameras
        self.init_ff_dim = init_ff_dim
        self.output_features = 2  # [x, y] for decoder_input

        # CLS tokens
        self.target_cls_token = nn.Parameter(torch.zeros(1, 1, init_embed_dim))
        self.camera_cls_token = nn.Parameter(torch.zeros(1, 1, init_embed_dim))
        self.sos_token = nn.Parameter(torch.zeros(1, 1, self.output_features))

        # Projections
        self.target_projection = nn.Linear(target_features, init_embed_dim)
        self.camera_projection = nn.Linear(camera_features, init_embed_dim)
        self.decoder_input_projection = nn.Linear(self.output_features, final_embed_dim)

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

        # Attention pooling
        self.target_pooling = nn.Linear(init_embed_dim, 1)
        self.camera_pooling = nn.Linear(init_embed_dim, 1)

        # Encoder Transformer layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(final_embed_dim, num_heads, final_ff_dim, dropout) for _ in range(num_layers)
        ])

        # Decoder Transformer layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(final_embed_dim, num_heads, final_ff_dim, dropout) for _ in range(num_decoder_layers)
        ])

        # Output heads
        self.output_head = nn.Sequential(  # For [x, y] at timestep 0
            nn.Linear(final_embed_dim, int(final_embed_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(final_embed_dim / 2), self.output_features)
        )
        self.output_head_dir_mag = nn.Sequential(  # For [direction, magnitude] at timestep 1+
            nn.Linear(final_embed_dim, int(final_embed_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(final_embed_dim / 2), self.output_features),
            nn.Sigmoid()  # direction: [0, 1], magnitude: [0, 1]
        )

        # Encoders
        self.encoder_target = EncodeLinear(init_embed_dim, final_embed_dim)
        self.encoder_camera = EncodeLinear(init_embed_dim, final_embed_dim)
        self.encode_env = EncodeLinear(12, final_embed_dim)
        self.encode_obstacle = EncodeLinear(obstacle_features, final_embed_dim)

        # Environment
        env = mate.make('MATE-4v8-9-v0')
        env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
        env_base = env.reset()
        self.env_base = eb_f.collected_infos(env_base)

        self.current_epoch = 0
        self.total_epochs = 300
    
    def get_teacher_forcing_ratio(self):
        """Cosine schedule từ 1.0 xuống 0.0"""
        return 0.5 * (1 + math.cos(math.pi * self.current_epoch / self.total_epochs))
    
    def get_sinusoidal_pos_encoding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def dir_mag_to_xy(self, dir_mag, prev_xy):
        """Chuyển [direction, magnitude] thành [x, y] dựa trên tọa độ trước đó"""
        direction = dir_mag[:, :, :, 0] * 2 * math.pi  # [0, 1] -> [0, 2pi]
        magnitude = dir_mag[:, :, :, 1]  # [0, 1]
        delta_x = magnitude * torch.cos(direction)
        delta_y = magnitude * torch.sin(direction)
        x = prev_xy[:, :, :, 0] + delta_x
        y = prev_xy[:, :, :, 1] + delta_y
        return torch.stack([x, y], dim=-1)

    def xy_to_dir_mag(self, xy):
        """Chuyển chuỗi [x, y] thành [direction, magnitude]"""
        delta_x = xy[:, :, 1:, 0] - xy[:, :, :-1, 0]  # [batch_size, num_targets, future_steps-1]
        delta_y = xy[:, :, 1:, 1] - xy[:, :, :-1, 1]
        magnitude = torch.sqrt(delta_x**2 + delta_y**2)
        direction = (torch.atan2(delta_y, delta_x) / (2 * math.pi)) % 1  # [0, 1]
        
        # Tạo đầu ra [batch_size, num_targets, future_steps, 2]
        dir_mag = torch.zeros_like(xy)
        dir_mag[:, :, 0, :] = xy[:, :, 0, :]  # Timestep 0: [x, y]
        dir_mag[:, :, 1:, 0] = direction
        dir_mag[:, :, 1:, 1] = magnitude
        return dir_mag

    def encode(self, targets, cameras, obstacles):
        batch_size, num_targets, target_flat_dim = targets.shape
        _, num_cameras, camera_flat_dim = cameras.shape
        targets_reshaped = targets.view(batch_size, num_targets, self.num_timesteps, self.target_features)
        cameras_reshaped = cameras.view(batch_size, self.num_cameras, self.num_timesteps, self.camera_features)

        # Environment
        new_env_base = np.tile(self.env_base, (batch_size, 1))
        new_env_base = np.expand_dims(new_env_base, axis=1)
        new_env_base = torch.tensor(new_env_base, dtype=torch.float32, device=targets.device)
        new_env_base = self.encode_env(new_env_base)

        # Obstacles
        obstacles_embedded = self.encode_obstacle(obstacles)

        # Targets: Projection
        targets_projected = self.target_projection(targets_reshaped)
        targets_segments = targets_projected.view(batch_size, num_targets, self.num_segments, self.steps_per_segment, self.init_embed_dim)

        # Add CLS token for targets
        cls_tokens = self.target_cls_token.expand(batch_size, num_targets, self.num_segments, 1, self.init_embed_dim)
        targets_with_cls = torch.cat([cls_tokens, targets_segments], dim=3)

        # Positional encoding for targets
        pos_encoding = self.get_sinusoidal_pos_encoding(self.steps_per_segment + 1, self.init_embed_dim, targets.device)
        pos_encoding = pos_encoding.expand(batch_size, num_targets, self.num_segments, -1, -1)
        targets_with_pos = targets_with_cls + pos_encoding

        # Target segment attention
        targets_flat = targets_with_pos.view(batch_size * num_targets * self.num_segments, self.steps_per_segment + 1, self.init_embed_dim)
        target_segment_attn_out, _ = self.target_segment_attention(targets_flat, targets_flat, targets_flat)
        target_cls = target_segment_attn_out[:, 0:1, :].contiguous()
        target_cls = target_cls.view(batch_size * num_targets * self.num_segments, self.init_embed_dim)
        target_cls = self.target_segment_norm(target_cls).view(batch_size * num_targets, self.num_segments, self.init_embed_dim)

        # Positional encoding for CLS
        pos_encoding_cls = self.get_sinusoidal_pos_encoding(self.num_segments, self.init_embed_dim, targets.device)
        pos_encoding_cls = pos_encoding_cls.expand(batch_size * num_targets, self.num_segments, self.init_embed_dim)
        target_cls = target_cls + pos_encoding_cls

        # CLS attention for targets
        target_cls_attn_out, _ = self.target_cls_attention(target_cls, target_cls, target_cls)
        target_scores = self.target_pooling(target_cls_attn_out).squeeze(-1)
        target_weights = F.softmax(target_scores, dim=1).unsqueeze(-1)
        targets_final = (target_cls_attn_out * target_weights).sum(dim=1)
        targets_final = targets_final.view(batch_size, num_targets, self.init_embed_dim)
        targets_final = self.target_cls_norm(targets_final)
        targets_embedded = self.encoder_target(targets_final)

        # Cameras: Projection
        cameras_projected = self.camera_projection(cameras_reshaped)
        cameras_segments = cameras_projected.view(batch_size, self.num_cameras, self.num_segments, self.steps_per_segment, self.init_embed_dim)

        # Add CLS token for cameras
        cls_tokens_cameras = self.camera_cls_token.expand(batch_size, self.num_cameras, self.num_segments, 1, self.init_embed_dim)
        cameras_with_cls = torch.cat([cls_tokens_cameras, cameras_segments], dim=3)

        # Positional encoding for cameras
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
        camera_scores = self.camera_pooling(camera_cls_attn_out).squeeze(-1)
        camera_weights = F.softmax(camera_scores, dim=1).unsqueeze(-1)
        cameras_final = (camera_cls_attn_out * camera_weights).sum(dim=1)
        cameras_final = cameras_final.view(batch_size, self.num_cameras, self.init_embed_dim)
        cameras_final = self.camera_cls_norm(cameras_final)
        cameras_embedded = self.encoder_camera(cameras_final)

        # Encoder layers
        targets_out = targets_embedded
        cameras_out = cameras_embedded
        obstacles_out = obstacles_embedded
        for layer in self.encoder_layers:
            targets_out, cameras_out, new_env_base, obstacles_out = layer(
                targets_out, cameras_out, new_env_base, obstacles_out
            )
        
        return targets_out, cameras_out, new_env_base, obstacles_out

    def decode(self, tgt, targets_memory, cameras_memory, env_memory, obstacles_memory, teacher_forcing=True):
        batch_size, num_targets, seq_len, _ = tgt.shape
        tgt = tgt.view(batch_size * num_targets, seq_len, self.output_features)
        tgt_embedded = self.decoder_input_projection(tgt)
        pos_encoding = self.get_sinusoidal_pos_encoding(seq_len, self.final_embed_dim, tgt.device)
        pos_encoding = pos_encoding.expand(batch_size * num_targets, -1, -1)
        tgt_embedded = tgt_embedded + pos_encoding
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)

        for layer in self.decoder_layers:
            tgt_embedded = layer(tgt_embedded, targets_memory, cameras_memory, env_memory, obstacles_memory, tgt_mask)

        # Dự đoán đầu ra
        coords = []
        dir_mags = []
        for t in range(seq_len):
            if t == 0:
                output = self.output_head(tgt_embedded[:, t:t+1, :])  # [batch_size * num_targets, 1, 2]
                coords.append(output.view(batch_size, num_targets, 1, 2))
                dir_mags.append(output.view(batch_size, num_targets, 1, 2))  # [x, y] tại t=0
            else:
                output = self.output_head_dir_mag(tgt_embedded[:, t:t+1, :])  # [batch_size * num_targets, 1, 2]
                output = output.view(batch_size, num_targets, 1, 2)
                dir_mags.append(output)  # [direction, magnitude]
                # Chuyển thành [x, y]
                prev_xy = coords[-1]
                xy = self.dir_mag_to_xy(output, prev_xy)
                coords.append(xy)
        
        coords = torch.cat(coords, dim=2)  # [batch_size, num_targets, seq_len, 2]
        dir_mags = torch.cat(dir_mags, dim=2)  # [batch_size, num_targets, seq_len, 2]
        return coords, dir_mags

    def forward(self, targets, cameras, obstacles, future_targets=None, teacher_forcing=True):
        # Encode
        targets_out, cameras_out, embedded_env_base, obstacles_out = self.encode(targets, cameras, obstacles)

        # Prepare decoder input
        batch_size = targets.shape[0]
        if teacher_forcing and future_targets is not None:
            teacher_forcing_ratio = self.get_teacher_forcing_ratio()
            if random.random() < teacher_forcing_ratio:
                # Teacher forcing: Sử dụng future_targets [batch_size, num_targets, future_steps, 2] ([x, y])
                sos = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features)
                decoder_input = torch.cat([sos, future_targets[:, :, :-1, :]], dim=2)
                coords, dir_mags = self.decode(decoder_input, targets_out, cameras_out, 
                                             embedded_env_base, obstacles_out, teacher_forcing)
                # Chuyển coords thành dir_mags nếu future_targets là [x, y]
                dir_mags = self.xy_to_dir_mag(coords)
                return coords, dir_mags
            else:
                # Non-teacher forcing trong training
                decoder_input = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features)
                coords = []
                dir_mags = []
                prev_xy = decoder_input
                for t in range(self.future_steps):
                    coord_t, dir_mag_t = self.decode(decoder_input, targets_out, cameras_out, 
                                                   embedded_env_base, obstacles_out, teacher_forcing=False)
                    coords.append(coord_t[:, :, -1:, :])
                    dir_mags.append(dir_mag_t[:, :, -1:, :])
                    if t == 0:
                        next_xy = coord_t[:, :, -1:, :]
                    else:
                        next_xy = self.dir_mag_to_xy(dir_mag_t[:, :, -1:, :], prev_xy)
                    decoder_input = torch.cat([decoder_input, next_xy], dim=2)
                    prev_xy = next_xy
                return torch.cat(coords, dim=2), torch.cat(dir_mags, dim=2)
        else:
            # Inference mode
            decoder_input = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features)
            coords = []
            dir_mags = []
            prev_xy = decoder_input
            for t in range(self.future_steps):
                coord_t, dir_mag_t = self.decode(decoder_input, targets_out, cameras_out, 
                                               embedded_env_base, obstacles_out, teacher_forcing=False)
                coords.append(coord_t[:, :, -1:, :])
                dir_mags.append(dir_mag_t[:, :, -1:, :])
                if t == 0:
                    next_xy = coord_t[:, :, -1:, :]
                else:
                    next_xy = self.dir_mag_to_xy(dir_mag_t[:, :, -1:, :], prev_xy)
                decoder_input = torch.cat([decoder_input, next_xy], dim=2)
                prev_xy = next_xy
            return torch.cat(coords, dim=2), torch.cat(dir_mags, dim=2)
    
    def set_current_epoch(self, epoch):
        """Cập nhật epoch hiện tại để tính teacher_forcing_ratio"""
        self.current_epoch = epoch
