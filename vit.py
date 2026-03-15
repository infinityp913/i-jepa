import torch
from torch import nn

class Transpose(nn.Module):
""" A simple module to transpose the dimensions of a tensor. """

    def __init__(self, d_1, d_2):
        super().__init__()
        
        self.dims = (d_1, d_2)

    def forward(self, x):
        return x.transpose(*self.dims)

class Tokenizer(nn.Module):
    """ A module to tokenize an image into patches and reconstruct it back. """

    def __init__(self, img_size, patch_size):
        super().__init__()

        self.unfold = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size),
            Transpose(1, 2)
        )

        self.fold = nn.Sequential(
            Transpose(1, 2), 
            nn.Fold(img_size, kernel_size=patch_size, stride=patch_size)
        )
    
    def encode(self, x):
        """
        Encodes an image into patches.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, patch_size * patch_size * channels).
        """
        return self.unfold(x.unsqueeze(1))
    
    def decode(self, x):
        """
        Decodes the patches back into an image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, patch_size * patch_size * channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        return self.fold(x).squeeze(1)

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