import torch
from torch import nn


class Transpose(nn.Module):
    """A simple module to transpose two tensor dimensions."""

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
            Transpose(1, 2),
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
        return self.unfold(x)
    
    def decode(self, x):
        """
        Decodes the patches back into an image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, patch_size * patch_size * channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        return self.fold(x)

class TransformerBlock(nn.Module):
    """A standard Transformer block for vision tokens."""

    def __init__(self, n_embed, n_head):
        """
        Args:
            n_embed: The dimensionality of the embedding space.
            n_head: The number of attention heads.
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embed)
        self.mha = nn.MultiheadAttention(n_embed, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.GELU(), # The ViT paper uses GELU activation function
            nn.Linear(n_embed * 4, n_embed),
        )

    def forward(self, x):
        """
        Forward pass of the Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).
        """
        normed_x = self.ln1(x)
        x = x + self.mha(normed_x, normed_x, normed_x, need_weights=False)[0] # need_weights=False uses the optimized scaled_dot_product_attention
        x = x + self.ffn(self.ln2(x)) 
        return x


class Encoder(nn.Module):
    """A Transformer based encoder for encoding images into patch-level embeddings."""

    def __init__(self, num_patches, patch_size=16, img_channels=3, d_model=768, n_head=12, n_layers=12):
        """
        Initializes the Encoder.

        Args:
            num_patches (int): Number of patches.
            patch_size (int): Square patch size (paper default: 16).
            img_channels (int): Number of input image channels (default: 3 for RGB).
            d_model (int): Embedding dimension (paper default: 768).
            n_head (int): Number of attention heads (paper default: 12).
            n_layers (int): Number of Transformer encoder blocks (paper default: 12).
        """
        super().__init__()        
        self.embed = nn.Linear(patch_size ** 2 * img_channels, d_model)

        self.positional_embedding = nn.Parameter(torch.randn(num_patches, d_model))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)  # Initialize positional embeddings with truncated normal distribution

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(d_model, n_head) for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, masks=None):
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Kept patches [B, n_keep, patch_size**2 * img_channels].
            masks (torch.Tensor): Patch indices to keep [B, n_keep].
        Returns:
            torch.Tensor: Embeddings of kept patches [B, n_keep, n_embed].
        """
        x = self.embed(x) + self.positional_embedding[masks]   # positional emb for kept positions only
        x = self.transformer_blocks(x)
        x = self.norm(x)
        return x

class Predictor(nn.Module):
    """A Transformer based predictor for predicting the masked patches."""

    def __init__(self, num_patches, embed_dim=768, d_model=768, n_head=12, n_layers=6):
        """
        Initializes the Predictor.

        Args:
            num_patches (int): Number of patches.
            embed_dim (int): Embedding dimension (paper default: 768).
            d_model (int): Embedding dimension (paper default: 768).
            n_head (int): Number of attention heads.
            n_layers (int): Number of Transformer blocks (paper uses a narrower/shallower predictor).
        """
        super().__init__()
        self.embed = nn.Linear(embed_dim, d_model)

        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.positional_embedding = nn.Parameter(torch.randn(num_patches, d_model))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(d_model, n_head) for _ in range(n_layers)])

        self.ln1 = nn.LayerNorm(d_model)

        self.proj = nn.Linear(d_model, embed_dim)

        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, x_masks, y_masks):
        """
        Forward pass of the Predictor.

        Args:
            x (torch.Tensor):             Context encoder outputs [B, n_keep_enc, n_embed].
            x_masks (torch.Tensor): Patch indices kept by the encoder [B, n_keep_enc].
            y_masks (torch.Tensor):  Patch indices to predict [B, n_keep_pred].
        Returns:
            torch.Tensor: Predicted embeddings at target positions [B, n_keep_pred, n_embed].
        """
        # add positional encoding to context tokens
        x = self.embed(x) + self.positional_embedding[x_masks]

        # build mask tokens and add positional encoding for target positions
        y = self.mask_token.expand(*y_masks.shape, -1) + self.positional_embedding[y_masks]

        # return only the predictions for the target positions
        y = self.transformer_blocks(torch.cat([x, y], dim=1))[:, x_masks.shape[1]:]

        # project back to original embedding dimension
        y = self.ln1(y)
        y = self.proj(y)

        y = self.ln2(y)
        return y