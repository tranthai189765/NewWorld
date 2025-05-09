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
        self.cross_attn_targets = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_cameras = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_env = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm0 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
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
        self_attn_out, _ = self.self_attn(targets, targets, targets)
        targets = self.norm1(targets + self.dropout(self_attn_out))
        cross_cameras_out, _ = self.cross_attn_cameras(targets, cameras, cameras)
        targets = self.norm2(targets + self.dropout(cross_cameras_out))
        cross_env_out, _ = self.cross_attn_env(targets, env_base, env_base)
        targets = self.norm3(targets + self.dropout(cross_env_out))
        ffn_out = self.ffn(targets)
        targets = self.norm4(targets + self.dropout(ffn_out))
        return targets, cameras

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_targets = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_cameras = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_env = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
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
    
    def forward(self, tgt, targets_memory, cameras_memory, env_memory, tgt_mask=None):
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))
        cross_targets_out, _ = self.cross_attn_targets(tgt, targets_memory, targets_memory)
        tgt = self.norm2(tgt + self.dropout(cross_targets_out))
        cross_cameras_out, _ = self.cross_attn_cameras(tgt, cameras_memory, cameras_memory)
        tgt = self.norm3(tgt + self.dropout(cross_cameras_out))
        cross_env_out, _ = self.cross_attn_env(tgt, env_memory, env_memory)
        tgt = self.norm4(tgt + self.dropout(cross_env_out))
        ffn_out = self.ffn(tgt)
        tgt = self.norm4(tgt + self.dropout(ffn_out))
        return tgt

class WorldModel(nn.Module):
    def __init__(self, init_embed_dim=32, final_embed_dim=128, init_num_heads=2, num_heads=8, 
                 init_ff_dim=64, final_ff_dim=512, num_layers=1, num_decoder_layers=1,
                 num_timesteps=30, steps_per_segment=5, future_steps=11,
                 num_targets=4, target_features=4, 
                 num_cameras=4, camera_features=13, dropout=0.3):
        super().__init__()
        self.target_features = target_features
        self.camera_features = camera_features
        self.dropout = dropout
        self.num_segments = num_timesteps // steps_per_segment
        self.num_timesteps = num_timesteps
        self.future_steps = future_steps
        self.num_targets = num_targets
        self.steps_per_segment = steps_per_segment
        self.init_embed_dim = init_embed_dim
        self.final_embed_dim = final_embed_dim
        self.num_cameras = num_cameras
        self.init_ff_dim = init_ff_dim
        self.output_features = 2  # Only predict [x, y]

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

        # Output head
        self.output_head = nn.Linear(final_embed_dim, self.output_features)

        # Encoders
        self.encoder_target = EncodeLinear(init_embed_dim, final_embed_dim)
        self.encoder_camera = EncodeLinear(init_embed_dim, final_embed_dim)
        self.encode_env = EncodeLinear(12, final_embed_dim)

        # Environment
        env = mate.make('MATE-4v4-0-v0')
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

    def encode(self, targets, cameras):
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
        for layer in self.encoder_layers:
            targets_out, cameras_out = layer(targets_out, cameras_out, new_env_base)
        
        return targets_out, cameras_out, new_env_base

    def decode(self, tgt, targets_memory, cameras_memory, env_memory, teacher_forcing=True):
            batch_size, num_targets, seq_len, _ = tgt.shape
            tgt = tgt.view(batch_size * num_targets, seq_len, self.output_features)
            tgt_embedded = self.decoder_input_projection(tgt)
            pos_encoding = self.get_sinusoidal_pos_encoding(seq_len, self.final_embed_dim, tgt.device)
            pos_encoding = pos_encoding.expand(batch_size * num_targets, -1, -1)
            tgt_embedded = tgt_embedded + pos_encoding
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)
            # Reshape memory để khớp với batch_size * num_targets
            targets_memory = targets_memory.view(batch_size * num_targets, -1, self.final_embed_dim)
            cameras_memory = cameras_memory.view(batch_size * self.num_cameras, -1, self.final_embed_dim)
            env_memory = env_memory.repeat_interleave(num_targets, dim=0)  # [batch_size * num_targets, 1, final_embed_dim]
            for layer in self.decoder_layers:
                tgt_embedded = layer(tgt_embedded, targets_memory, cameras_memory, env_memory, tgt_mask)
            output = self.output_head(tgt_embedded)
            output = output.view(batch_size, num_targets, seq_len, self.output_features)
            return output

    def forward(self, targets, cameras, future_targets=None, teacher_forcing=True):
        # Encode
        targets_out, cameras_out, embedded_env_base = self.encode(targets, cameras)

        # Prepare decoder input
        batch_size = targets.shape[0]
        if teacher_forcing and future_targets is not None:
            teacher_forcing_ratio = self.get_teacher_forcing_ratio()
            if random.random() < teacher_forcing_ratio:
                # Teacher forcing: Use ground truth future_targets [batch_size, num_targets, future_steps, 2]
                sos = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features)
                decoder_input = torch.cat([sos, future_targets[:, :, :-1, :]], dim=2)  # [batch_size, num_targets, future_steps, 2]
            else:
                # Inference: Initialize with SOS token
                decoder_input = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features)
                outputs = []
                for _ in range(self.future_steps):
                    output = self.decode(decoder_input, targets_out, cameras_out, embedded_env_base, teacher_forcing=False)
                    outputs.append(output[:, :, -1:, :])  # Take last timestep
                    decoder_input = torch.cat([decoder_input, output[:, :, -1:, :]], dim=2)
                return torch.cat(outputs, dim=2)  # [batch_size, num_targets, future_steps, 2]

        else:
            # Inference: Initialize with SOS token
            decoder_input = self.sos_token.expand(batch_size, self.num_targets, 1, self.output_features)
            outputs = []
            for _ in range(self.future_steps):
                output = self.decode(decoder_input, targets_out, cameras_out, embedded_env_base, teacher_forcing=False)
                outputs.append(output[:, :, -1:, :])  # Take last timestep
                decoder_input = torch.cat([decoder_input, output[:, :, -1:, :]], dim=2)
            return torch.cat(outputs, dim=2)  # [batch_size, num_targets, future_steps, 2]

        # Decode
        output = self.decode(decoder_input, targets_out, cameras_out, embedded_env_base, teacher_forcing)
        return output
    
    def set_current_epoch(self, epoch):
        """Cập nhật epoch hiện tại để tính teacher_forcing_ratio"""
        self.current_epoch = epoch
