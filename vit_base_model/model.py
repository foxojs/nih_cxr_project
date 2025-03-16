import torch
import torch.nn as nn
from utils import extract_patches
import config

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=config.HIDDEN_SIZE, num_heads=config.NUM_HEADS):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.ELU(), nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, x):
        x = self.multihead_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0] + x
        x = self.mlp(self.norm2(x)) + x
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(config.PATCH_SIZE ** 2 * 3, config.HIDDEN_SIZE)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config.NUM_LAYERS)])
        self.fc_out = nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES)
        self.pos_embedding = nn.Parameter(torch.empty(1, (config.IMAGE_SIZE[0] // config.PATCH_SIZE) ** 2, config.HIDDEN_SIZE).normal_(std=0.001))

    def forward(self, image):
        patch_seq = extract_patches(image, patch_size=config.PATCH_SIZE)
        embs = self.fc_in(patch_seq) + self.pos_embedding
        for block in self.blocks:
            embs = block(embs)
        return self.fc_out(embs[:, 0])

