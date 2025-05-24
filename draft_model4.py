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
        # Concatenate all inputs into a single sequence
        batch_size = targets.shape[0]
        num_targets = targets.shape[1]
        num_cameras = cameras.shape[1]
        num_env = env_base.shape[1]
        num_obstacles = obstacles.shape[1]
        
        # Combine all objects: [batch_size, num_targets + num_cameras + num_env + num_obstacles, embed_dim]
        combined = torch.cat([targets, cameras, env_base, obstacles], dim=1)
        
        # Self-attention on combined sequence
        attn_out, _ = self.self_attn(combined, combined, combined)
        combined = self.norm1(combined + self.dropout(attn_out))
        
        # Feed-forward network
        ffn_out = self.ffn(combined)
        combined = self.norm2(combined + self.dropout(ffn_out))
        
        # Split back into individual components
        targets_out = combined[:, :num_targets, :]
        cameras_out = combined[:, num_targets:num_targets+num_cameras, :]
        env_base_out = combined[:, num_targets+num_cameras:num_targets+num_cameras+num_env, :]
        obstacles_out = combined[:, num_targets+num_cameras+num_env:, :]
        
        return targets_out, cameras_out, env_base_out, obstacles_out

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_targets = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_cameras = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_env = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_obstacles = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.norm5 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, targets_memory, cameras_memory, env_memory, obstacles_memory, tgt_mask=None):
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))
        cross_targets_out, _ = self.cross_attn_targets(tgt, targets_memory, targets_memory)
        tgt = self.norm2(tgt + self.dropout(cross_targets_out))
        cross_cameras_out, _ = self.cross_attn_cameras(tgt, cameras_memory, cameras_memory)
        tgt = self.norm3(tgt + self.dropout(cross_cameras_out))
        cross_env_out, _ = self.cross_attn_env(tgt, env_memory, env_memory)
        tgt = self.norm4(tgt + self.dropout(cross_env_out))
        cross_obstacles_out, _ = self.cross_attn_obstacles(tgt, obstacles_memory, obstacles_memory)
        tgt = self.norm5(tgt + self.dropout(cross_obstacles_out))
        ffn_out = self.ffn(tgt)
        tgt = self.norm5(tgt + self.dropout(ffn_out))
        return tgt

class WorldModel(nn.Module):
    def __init__(self, init_embed_dim=32, final_embed_dim=128, init_num_heads=2, num_heads=8, 
                 init_ff_dim=64, final_ff_dim=512, num_layers=1, num_decoder_layers=1,
                 num_timesteps=100, steps_per_segment=5, future_steps=10,
                 num_targets=4, target_features=8, 
                 num_cameras=8, camera_features=17, 
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
        self.output_features_first = 2  # [x, y] for first timestep
        self.output_features_rest = 3  # [cos(theta), sin(theta), magnitude] for rest

        # CLS tokens
        self.target_cls_token = nn.Parameter(torch.zeros(1, 1, init_embed_dim))
        self.camera_cls_token = nn.Parameter(torch.zeros(1, 1, init_embed_dim))
        self.sos_token = nn.Parameter(torch.zeros(1, 1, self.output_features_first))

        # Projections
        self.target_projection = nn.Linear(target_features, init_embed_dim)
        self.camera_projection = nn.Linear(camera_features, init_embed_dim)
        self.decoder_input_projection = nn.Linear(max(self.output_features_first, self.output_features_rest), final_embed_dim)

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
        self.output_head_first = nn.Linear(final_embed_dim, self.output_features_first)  # For t=1
        self.output_head_rest = nn.Linear(final_embed_dim, self.output_features_rest)   # For t=2 to t=10

        # Encoders
        self.encoder_target = EncodeLinear(init_embed_dim, final_embed_dim)
        self.encoder_camera = EncodeLinear(init_embed_dim, final_embed_dim)
        self.encoder_obstacle = EncodeLinear(obstacle_features, final_embed_dim)
        self.encode_env = EncodeLinear(12, final_embed_dim)

        # Environment
        env = mate.make('MATE-4v8-9-v0')
        env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
        env_base = env.reset()
        self.env_base = eb_f.collected_infos(env_base)

        self.current_epoch = 0
        self.total_epochs = 300
    
    def get_teacher_forcing_ratio(self):
        """Cosine schedule from 1.0 to 0.0"""
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

    def encode(self, targets, cameras, obstacles):
        batch_size, num_targets, target_flat_dim = targets.shape
        _, num_cameras, camera_flat_dim = cameras.shape
        _, num_obstacles, obstacle_flat_dim = obstacles.shape
        targets_reshaped = targets.view(batch_size, num_targets, self.num_timesteps, self.target_features)
        cameras_reshaped = cameras.view(batch_size, self.num_cameras, self.num_timesteps, self.camera_features)

        # Environment
        new_env_base = np.tile(self.env_base, (batch_size, 1))
        new_env_base = np.expand_dims(new_env_base, axis=1)
        new_env_base = torch.tensor(new_env_base, dtype=torch.float32, device=targets.device)
        new_env_base = self.encode_env(new_env_base)

        # Obstacles: Projection
        obstacles_embedded = self.encoder_obstacle(obstacles)  # [batch_size, num_obstacles, final_embed_dim]

        # Targets: Projection
        targets_projected = self.target_projection(targets_reshaped)
        targets_segments = targets_projected.view(batch_size, num_targets, self.num_segments, self.steps_per_segment, self.init_embed_dim)

        # Add CLS token for targets
        cls_tokens = self.target_cls_token.expand(batch_size, num_targets, self.num_segments, 1, self.init_embed_dim)
        targets_with_cls = torch.cat([cls_tokens, targets_segments], dim=3)

        # Positional encoding for targets (CLS + timesteps)
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
        camera_scores = self.camera_pooling(camera_cls_attn_out).squeeze(-1)
        camera_weights = F.softmax(camera_scores, dim=1).unsqueeze(-1)
        cameras_final = (camera_cls_attn_out * camera_weights).sum(dim=1)
        cameras_final = cameras_final.view(batch_size, self.num_cameras, self.init_embed_dim)
        cameras_final = self.camera_cls_norm(cameras_final)
        cameras_embedded = self.encoder_camera(cameras_final)

        # Encoder layers
        targets_out = targets_embedded
        cameras_out = cameras_embedded
        env_base_out = new_env_base
        obstacles_out = obstacles_embedded
        for layer in self.encoder_layers:
            targets_out, cameras_out, env_base_out, obstacles_out = layer(
                targets_out, cameras_out, env_base_out, obstacles_out
            )
        
        return targets_out, cameras_out, env_base_out, obstacles_out

    def decode(self, tgt, targets_memory, cameras_memory, env_memory, obstacles_memory, teacher_forcing=True):
        batch_size, num_targets, seq_len, _ = tgt.shape
        tgt = tgt.view(batch_size * num_targets, seq_len, -1)
        tgt_embedded = self.decoder_input_projection(tgt)
        pos_encoding = self.get_sinusoidal_pos_encoding(seq_len, self.final_embed_dim, tgt.device)
        pos_encoding = pos_encoding.expand(batch_size * num_targets, -1, -1)
        tgt_embedded = tgt_embedded + pos_encoding
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)
        targets_memory = targets_memory.view(batch_size * num_targets, -1, self.final_embed_dim)
        cameras_memory = cameras_memory.view(batch_size * self.num_cameras, -1, self.final_embed_dim)
        env_memory = env_memory.repeat_interleave(num_targets, dim=0)
        obstacles_memory = obstacles_memory.repeat_interleave(num_targets, dim=0)
        for layer in self.decoder_layers:
            tgt_embedded = layer(tgt_embedded, targets_memory, cameras_memory, env_memory, obstacles_memory, tgt_mask)
        
        # Apply different output heads based on timestep
        outputs = []
        for t in range(seq_len):
            if t == 0:
                output_t = self.output_head_first(tgt_embedded[:, t:t+1, :])
            else:
                output_t = self.output_head_rest(tgt_embedded[:, t:t+1, :])
                direction = output_t[:, :, :2]
                direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
                magnitude = output_t[:, :, 2:3].clamp(min=0)
                output_t = torch.cat([direction, magnitude], dim=-1)
            outputs.append(output_t)
        output = torch.cat(outputs, dim=1)
        output = output.view(batch_size, num_targets, seq_len, -1)
        return output

    def vector_to_position(self, prev_pos, direction, magnitude):
        """Convert direction vector and magnitude to new position"""
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
        displacement = direction * magnitude
        new_pos = prev_pos + displacement
        return new_pos

    def forward(self, targets, cameras, obstacles, future_targets=None, teacher_forcing=True):
        # Encode
        targets_out, cameras_out, embedded_env_base, obstacles_out = self.encode(targets, cameras, obstacles)
        batch_size = targets.shape[0]

        if teacher_forcing and future_targets is not None:
            teacher_forcing_ratio = self.get_teacher_forcing_ratio()
            if random.random() < teacher_forcing_ratio:
                # Prepare teacher forcing input
                decoder_input = torch.zeros(batch_size, self.num_targets, self.future_steps, 3, device=targets.device)
                decoder_input[:, :, 0, :2] = future_targets[:, :, 0, :2]  # First timestep: [x, y]
                for t in range(1, self.future_steps):
                    delta = future_targets[:, :, t, :2] - future_targets[:, :, t-1, :2]
                    magnitude = torch.norm(delta, dim=-1, keepdim=True)
                    direction = delta / (magnitude + 1e-8)
                    decoder_input[:, :, t, :2] = direction
                    decoder_input[:, :, t, 2:3] = magnitude
                sos = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features_first)
                decoder_input = torch.cat([sos, decoder_input[:, :, :-1, :]], dim=2)
            else:
                # Inference mode
                decoder_input = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features_first)
                outputs = []
                prev_pos = None
                for t in range(self.future_steps):
                    output = self.decode(decoder_input, targets_out, cameras_out, embedded_env_base, obstacles_out, teacher_forcing=False)
                    if t == 0:
                        outputs.append(output[:, :, -1:, :2])
                        prev_pos = output[:, :, -1, :2]
                        decoder_input = torch.cat([decoder_input, output[:, :, -1:, :]], dim=2)
                    else:
                        direction = output[:, :, -1, :2]
                        magnitude = output[:, :, -1, 2:3]
                        new_pos = self.vector_to_position(prev_pos, direction, magnitude)
                        outputs.append(new_pos.unsqueeze(2))
                        next_input = torch.cat([direction, magnitude], dim=-1).unsqueeze(2)
                        decoder_input = torch.cat([decoder_input, next_input], dim=2)
                        prev_pos = new_pos
                return torch.cat(outputs, dim=2)

        else:
            # Inference mode
            decoder_input = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features_first)
            outputs = []
            prev_pos = None
            for t in range(self.future_steps):
                output = self.decode(decoder_input, targets_out, cameras_out, embedded_env_base, obstacles_out, teacher_forcing=False)
                if t == 0:
                    outputs.append(output[:, :, -1:, :2])
                    prev_pos = output[:, :, -1, :2]
                    decoder_input = torch.cat([decoder_input, output[:, :, -1:, :]], dim=2)
                else:
                    direction = output[:, :, -1, :2]
                    magnitude = output[:, :, -1, 2:3]
                    new_pos = self.vector_to_position(prev_pos, direction, magnitude)
                    outputs.append(new_pos.unsqueeze(2))
                    next_input = torch.cat([direction, magnitude], dim=-1).unsqueeze(2)
                    decoder_input = torch.cat([decoder_input, next_input], dim=2)
                    prev_pos = new_pos
            return torch.cat(outputs, dim=2)

        # Decode
        output = self.decode(decoder_input, targets_out, cameras_out, embedded_env_base, obstacles_out, teacher_forcing)
        return output
    
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

def prepare_training_labels(coordinates):
    """
    Convert ground truth coordinates to training labels.
    Args:
        coordinates: Tensor of shape [batch_size, num_targets, future_steps, 2]
                     containing [x, y] coordinates for each timestep.
    Returns:
        labels: Tensor of shape [batch_size, num_targets, future_steps, 3]
                where t=0 has [x, y, 0], and t=1 to t=9 have [cos(theta), sin(theta), magnitude].
    """
    batch_size, num_targets, future_steps, _ = coordinates.shape
    labels = torch.zeros(batch_size, num_targets, future_steps, 3, device=coordinates.device)
    
    # First timestep: [x, y, 0]
    labels[:, :, 0, :2] = coordinates[:, :, 0, :]
    
    # Subsequent timesteps: [cos(theta), sin(theta), magnitude]
    for t in range(1, future_steps):
        delta = coordinates[:, :, t, :] - coordinates[:, :, t-1, :]
        magnitude = torch.norm(delta, dim=-1, keepdim=True)
        direction = delta / (magnitude + 1e-8)
        labels[:, :, t, :2] = direction
        labels[:, :, t, 2:3] = magnitude
    
    return labels
