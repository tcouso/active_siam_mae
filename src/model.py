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
seq_length = grid_side_length * grid_side_length

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
    pos_embeddings = pos_embeddings.view(seq_length, -1).unsqueeze(0)

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


    def forward(self, query_embedding: torch.Tensor, key_value_embedding: torch.Tensor) -> torch.Tensor:

        seq_length = query_embedding.shape[1]

        query = self.query_layer(query_embedding)
        key = self.key_layer(key_value_embedding)
        value = self.value_layer(key_value_embedding)

        # Reshape for multi-headed attention
        query = query.view(-1, seq_length, num_attn_heads, self.head_dimension)
        query = query.permute(0, 2, 1, 3)

        key = key.view(-1, seq_length, num_attn_heads, self.head_dimension)
        key = key.permute(0, 2, 1, 3)

        value = value.view(-1, seq_length, num_attn_heads, self.head_dimension)
        value = value.permute(0, 2, 1, 3)

        simmilarity_scores = torch.matmul(query, torch.transpose(key, 2, 3))
        scaled_simmilarity_scores = simmilarity_scores / math.sqrt(self.head_dimension)

        similarity_probs = torch.softmax(scaled_simmilarity_scores, dim=-1)
        attn = torch.matmul(similarity_probs, value)

        # Restore batch, seq length, embedding shape
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(-1, seq_length, hidden_dim)

        return attn


class ActiveSiamMAEEncoderBlock(nn.Module):
    def __init__(self):
        super(ActiveSiamMAEEncoderBlock, self).__init__()
        self.num_attn_heads = num_attn_heads
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.multi_head_attn = ActiveSiamMAEMultiHeadAttention()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        norm_embeddings = self.layer_norm1(embeddings)
        attn_embeddings = embeddings + self.multi_head_attn(norm_embeddings, norm_embeddings)
        mlp_embeddings = attn_embeddings + self.mlp(self.layer_norm2(attn_embeddings))

        return mlp_embeddings


class ActiveSiamMAEDecoderBlock(nn.Module):
    def __init__(self):
        super(ActiveSiamMAEDecoderBlock, self).__init__()
        self.num_attn_heads = num_attn_heads
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        self.multi_head_self_attn = ActiveSiamMAEMultiHeadAttention()
        self.multi_head_cross_attn = ActiveSiamMAEMultiHeadAttention()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        
    def forward(self, past_embeddings: torch.Tensor, future_embeddings: torch.Tensor) -> torch.Tensor:
        norm_future_embeddings = self.layer_norm1(future_embeddings)
        attn_embeddings = future_embeddings + self.multi_head_self_attn(norm_future_embeddings, norm_future_embeddings)
        attn_embeddings = attn_embeddings + self.multi_head_cross_attn(self.layer_norm2(attn_embeddings), past_embeddings)
        mlp_embeddings = attn_embeddings + self.mlp(self.layer_norm3(attn_embeddings))

        return mlp_embeddings


class ActiveSiamMAEEncoder(nn.Module):
    def __init__(self, num_layers: int, masking_ratio: float = 0.75):
        super(ActiveSiamMAEEncoder, self).__init__()
        self.num_layers = num_layers
        self.masking_ratio = masking_ratio
        pos_embeddings = generate_pos_embeddings(hidden_dim=hidden_dim, grid_side_length=grid_side_length)
        self.register_buffer('pos_embeddings', pos_embeddings)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.patch_embed_layer = ActiveSiamMAEPatchifier()
        self.attn_blocks = nn.ModuleList([
            ActiveSiamMAEEncoderBlock() for _ in range(self.num_layers)
        ])


    def forward(self, past_frame: torch.Tensor, future_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Past frame is given complete
        past_embeddings = self.patch_embed_layer(past_frame)
        past_embeddings += self.pos_embeddings

        for attn_block in self.attn_blocks:
            past_embeddings = attn_block(past_embeddings)

        past_embeddings = self.layer_norm(past_embeddings)

        # Future frame is masked
        future_embeddings = self.patch_embed_layer(future_frame)
        future_embeddings += self.pos_embeddings

        batch_size = past_frame.shape[0]
        rand_tensor = torch.rand(batch_size, seq_length)
        _, ids_shuffle = rand_tensor.sort(dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_shuffle_expanded = ids_shuffle.unsqueeze(-1).expand(-1, -1, hidden_dim)
        future_embeddings = torch.gather(future_embeddings, dim=1, index=ids_shuffle_expanded)

        num_masked_embeddings = int(seq_length * (1 - self.masking_ratio))
        future_embeddings = future_embeddings[:, :num_masked_embeddings]

        for attn_block in self.attn_blocks:
            future_embeddings = attn_block(future_embeddings)

        future_embeddings = self.layer_norm(future_embeddings)

        return past_embeddings, future_embeddings, ids_restore

    

class ActiveSiamMAEDecoder(nn.Module):
    def __init__(self, num_layers: int):
        super(ActiveSiamMAEDecoder, self).__init__()
        self.num_layers = num_layers
        pos_embeddings = generate_pos_embeddings(hidden_dim=hidden_dim, grid_side_length=grid_side_length)
        self.register_buffer('pos_embeddings', pos_embeddings)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_layer= nn.Linear(hidden_dim, num_channels * patch_size * patch_size)
        self.unpatchify_layer = ActiveSiamMAEDepatchifier()
        self.attn_blocks = nn.ModuleList([
            ActiveSiamMAEDecoderBlock() for _ in range(self.num_layers)
        ])


    def forward(self, past_embeddings: torch.Tensor, future_embeddings: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        batch_size = future_embeddings.shape[0]
        masked_seq_length = future_embeddings.shape[1]

        mask_embeddings = self.mask_token.repeat(batch_size, seq_length - masked_seq_length, 1)
        future_embeddings = torch.concat((future_embeddings, mask_embeddings), dim=1)
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, hidden_dim)
        future_embeddings = torch.gather(future_embeddings, dim=1, index=ids_restore_expanded)
        future_embeddings += self.pos_embeddings

        for attn_block in self.attn_blocks:
            future_embeddings = attn_block(past_embeddings, future_embeddings)

        future_embeddings = self.layer_norm(future_embeddings)
        out = self.output_layer(future_embeddings)
        patches = self.unpatchify_layer(out)

        return patches

if __name__ == '__main__':
    print("Test run")
    dataset = read_platonic_solids_dataset("../data/medium_data/")

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    batch = next(iter(train_loader))

    encoder = ActiveSiamMAEEncoder(8)
    decoder = ActiveSiamMAEDecoder(8)

    past_frame = batch['images'][:, 0, :, :, :].permute(0, 3, 1, 2).float()
    future_frame = batch['images'][:, 1, :, :, :].permute(0, 3, 1, 2).float()

    past_embeddings, future_embeddings, ids_restore = encoder(past_frame, future_frame)
    patches = decoder(past_embeddings, future_embeddings, ids_restore)

    print(patches.shape)
    