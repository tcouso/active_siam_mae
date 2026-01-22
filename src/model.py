import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from src.dataset import read_platonic_solids_dataset

num_channels = 3
patch_size = 16
grid_side_length = 14
hidden_dim = 512
img_size = 224
num_attn_heads = 8
num_layers = 4

def generate_pos_embeddings(hidden_dim: int, grid_side_length: int) -> torch.Tensor:
    grid_arange = torch.arange(grid_side_length)
    grid_x, grid_y = torch.meshgrid(grid_arange, grid_arange, indexing='xy')

    omega_i = torch.exp(((-2 * torch.arange(hidden_dim // 4)) / hidden_dim) * math.log(10_000))
    sin_grid_x = torch.sin(grid_x.unsqueeze(-1) * omega_i)
    cos_grid_x = torch.cos(grid_x.unsqueeze(-1) * omega_i)
    grid_x_pos_embeddings = torch.concat((sin_grid_x, cos_grid_x), dim=-1)

    sin_grid_y = torch.sin(grid_y.unsqueeze(-1) * omega_i)
    cos_grid_y = torch.cos(grid_y.unsqueeze(-1) * omega_i)
    grid_y_pos_embeddings = torch.concat((sin_grid_y, cos_grid_y), dim=-1)

    pos_embeddings = torch.concat((grid_x_pos_embeddings, grid_y_pos_embeddings), dim=-1)
    pos_embeddings = pos_embeddings.view(grid_side_length * grid_side_length, -1).unsqueeze(0)

    return pos_embeddings


class ActiveSiamMAEPatchifier(nn.Module):
    def __init__(self):
        super(ActiveSiamMAEPatchifier, self).__init__()
        self.patch_embed_layer = nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embed_layer(frame)
        flat_embeddings = torch.flatten(embeddings.permute(0, 2, 3, 1), start_dim=1, end_dim=2)

        return flat_embeddings


class ActiveSiamMAEDepatchifier(nn.Module):
    def __init__(self):
        super(ActiveSiamMAEDepatchifier, self).__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:

        patches = embeddings.view(-1, grid_side_length, grid_side_length, num_channels, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5)
        img = patches.reshape(-1, num_channels, img_size, img_size)

        return img


class ActiveSiamMAEMultiHeadAttention(nn.Module):
    def __init__(self):
        super(ActiveSiamMAEMultiHeadAttention, self).__init__()
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.head_dimension = hidden_dim // num_attn_heads


    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        # TODO: This implementation only allows for self attention
        key = self.key_layer(embedding)
        query = self.query_layer(embedding)
        value = self.value_layer(embedding)

        # Reshape for multi-headed attention
        key = key.view(-1, grid_side_length * grid_side_length, num_attn_heads, self.head_dimension)
        key = key.permute(0, 2, 1, 3)

        query = query.view(-1, grid_side_length * grid_side_length, num_attn_heads, self.head_dimension)
        query = query.permute(0, 2, 1, 3)

        value = value.view(-1, grid_side_length * grid_side_length, num_attn_heads, self.head_dimension)
        value = value.permute(0, 2, 1, 3)

        simmilarity_scores = torch.matmul(query, torch.transpose(key, 2, 3))
        scaled_simmilarity_scores = simmilarity_scores / math.sqrt(self.head_dimension)

        similarity_probs = torch.softmax(scaled_simmilarity_scores, dim=-1)
        attn = torch.matmul(similarity_probs, value)

        # Restore batch, seq length, embedding shape
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(-1, grid_side_length * grid_side_length, hidden_dim)

        return attn


class ActiveSiamMAEAttentionBlock(nn.Module):
    def __init__(self):
        super(ActiveSiamMAEAttentionBlock, self).__init__()
        self.num_attn_heads = num_attn_heads
        self.multi_head_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            ActiveSiamMAEMultiHeadAttention(),
            )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        attn_embedding = self.multi_head_attn(embedding)
        attn_embedding += embedding

        mlp_embedding = self.mlp(attn_embedding)
        mlp_embedding += attn_embedding

        return mlp_embedding


class ActiveSiamMAEEncoder(nn.Module):
    def __init__(self, num_layers: int):
        super(ActiveSiamMAEEncoder, self).__init__()
        self.num_layers = num_layers
        pos_embeddings = generate_pos_embeddings(hidden_dim=hidden_dim, grid_side_length=grid_side_length)
        self.register_buffer('pos_embeddings', pos_embeddings)

        self.patch_embed_layer = ActiveSiamMAEPatchifier()
        self.attn_blocks = nn.ModuleList([
            ActiveSiamMAEAttentionBlock() for _ in range(self.num_layers)
        ])


    def forward(self, past_frame: torch.Tensor, future_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = [past_frame, future_frame]
        enc_embeddings = []

        for frame in frames:
            embedding = self.patch_embed_layer(frame)
            embedding += self.pos_embeddings

            for attn_block in self.attn_blocks:
                embedding = attn_block(embedding)

            enc_embeddings.append(embedding)

        return tuple(enc_embeddings)
    

class ActiveSiamMAEDecoder(nn.Module):
    def __init__(self):
        super(ActiveSiamMAEDecoder, self).__init__()
        self.key_layer= nn.Linear(hidden_dim, hidden_dim)
        self.query_layer= nn.Linear(hidden_dim, hidden_dim)
        self.value_layer= nn.Linear(hidden_dim, hidden_dim)
        self.output_layer= nn.Linear(hidden_dim, num_channels * patch_size * patch_size)
        self.unpatchify_layer = ActiveSiamMAEDepatchifier()

    def forward(self, past_embeddings: torch.Tensor, future_embeddings: torch.Tensor) -> torch.Tensor:
        
        key = self.key_layer(past_embeddings)
        query = self.query_layer(future_embeddings)
        value = self.value_layer(past_embeddings)


        simmilarity_scores = torch.matmul(query, torch.transpose(key, 1, 2))
        scaled_simmilarity_scores = simmilarity_scores / math.sqrt(hidden_dim)

        similarity_probs = torch.softmax(scaled_simmilarity_scores, dim=-1)
        attn = torch.matmul(similarity_probs, value)

        out = self.output_layer(attn)
        patches = self.unpatchify_layer(out)

        return patches


if __name__ == '_main__':

    dataset = read_platonic_solids_dataset("../data/medium_data/")

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    batch = next(iter(train_loader))

    patch_embed = ActiveSiamMAEPatchifier()
    encoder = ActiveSiamMAEEncoder(1)
    decoder = ActiveSiamMAEDecoder()

    past_frame = batch['images'][:, 0, :, :].permute(0, 3, 1, 2).float()
    future_frame = batch['images'][:, 1, :, :].permute(0, 3, 1, 2).float()

    past_embeddings, future_embeddings = encoder(past_frame, future_frame)
    out = decoder(past_embeddings, future_embeddings)