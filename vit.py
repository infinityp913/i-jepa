import torch
from torch import nn

class VisionTransformer(nn.Module):
    """ A simple implementation of a Vision Transformer (ViT) model. """

    def __init__(self, n_embed, n_head):
        """
        Args:
            n_embed: The dimensionality of the embedding space.
            n_head: The number of attention heads.
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(n_embed, n_head)
        self.norm = nn.LayerNorm(n_embed)
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed)
        )

    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, n_embed).
        """
        normed_x = self.norm(x)
        attn_output, _ = self.mha(normed_x, normed_x, normed_x)
        x = x + attn_output  # Residual connection
        normed_x = self.norm(x)
        ffn_output = self.ffn(normed_x)
        x = x + ffn_output  # Residual connection
        return x