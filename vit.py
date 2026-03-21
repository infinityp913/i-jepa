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

class PatchEmbedding(nn.Module):
    """ A module to embed the patches into a higher-dimensional space. """

    def __init__(self, patch_size, n_embed, img_channels=3):
        """
        Initialize patch embedding layer.

        Args:
            patch_size (int): Square patch size (paper default: 16).
            n_embed (int): The dimensionality of the embedding space.
            img_channels (int): Number of channels in the input image (default is 3 for RGB images).
        """
        super().__init__()
        self.linear = nn.Linear(patch_size * patch_size * img_channels, n_embed)

    def forward(self, x):
        """
        Embeds the patches into a higher-dimensional space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, patch_size * patch_size * channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, n_embed).
        """
        return self.linear(x)


class TransformerEncoderBlock(nn.Module):
    """A standard Transformer encoder block for vision tokens."""

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
        Forward pass of the Transformer encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        normed_x = self.ln1(x)
        x = x + self.mha(normed_x, normed_x, normed_x)[0]
        x = x + self.ffn(self.ln2(x)) 
        return x


class ViT(nn.Module):
    """A stack of Transformer encoder blocks forming the core Vision Transformer backbone."""

    def __init__(self, n_embed, n_head, n_layers):
        """
        Args:
            n_embed (int): Embedding dimension.
            n_head (int): Number of attention heads.
            n_layers (int): Number of Transformer encoder blocks.
        """
        super().__init__()

        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoderBlock(n_embed, n_head) for _ in range(n_layers)
        ])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        return self.transformer_blocks(x)

class ViTContextEncoder(nn.Module):
    """A simple Vision Transformer (ViT) model for encoding images into patch-level embeddings."""

    def __init__(self, img_size, patch_size=16, n_embed=768, n_head=12, n_layers=12, img_channels=3):
        """
        Initializes the Vision Transformer model that consists of a tokenizer, patch embedding, and multiple Transformer encoder blocks.

        Args:
            img_size (int): Square input image size (e.g. 224 for ImageNet).
            patch_size (int): Square patch size (paper default: 16).
            n_embed (int): Embedding dimension (paper default: 768).
            n_head (int): Number of attention heads (paper default: 12).
            n_layers (int): Number of Transformer encoder blocks (paper default: 12).
            img_channels (int): Number of input image channels (default: 3 for RGB).
        """
        super().__init__()

        self.tokenizer = Tokenizer(img_size, patch_size)

        num_patches = (img_size // patch_size) ** 2

        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, n_embed))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)  # Initialize positional embeddings with truncated normal distribution

        self.patch_embedding = PatchEmbedding(patch_size, n_embed, img_channels)
        
        self.vit = ViT(n_embed, n_head, n_layers)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer encoder that encodes the input image into patch-level embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Patch-level embeddings tensor of shape (batch_size, num_patches, n_embed).
        """
        x = self.tokenizer.encode(x)
        x = self.patch_embedding(x)
        x = x + self.positional_embedding
        x = self.vit(x)
        return x
    


class ViTPredictor(nn.Module):
    """
    ViT-based predictor that estimates target patch embeddings from context representations.

    Takes context encoder output and injects learned mask tokens at target positions,
    then runs a Transformer so every position can attend to the full context when
    producing its prediction.
    """

    def __init__(self, num_patches, n_embed=768, n_head=12, n_layers=12):
        """
        Args:
            num_patches (int): Total number of patches in the image.
            n_embed (int): Embedding dimension (default: 768).
            n_head (int): Number of attention heads (default: 12).
            n_layers (int): Number of Transformer encoder blocks (default: 12).
        """
        super().__init__()


        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, n_embed))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)  # Initialize positional embeddings with truncated normal distribution

        self.patch_embedding = nn.Linear(n_embed, n_embed)

        self.target_mask = nn.Parameter(torch.randn(1,1,n_embed))
        
        self.vit = ViT(n_embed, n_head, n_layers)

    def forward(self, x, target_indices):
        """
        Predicts patch embeddings at target positions from context encoder representations.

        The learned mask token is added at target positions so the Transformer can
        distinguish which patches it should predict vs. which are context.

        Args:
            x (torch.Tensor): Context encoder output of shape (batch_size, num_patches, n_embed).
            target_indices (torch.Tensor): Float mask of shape (batch_size, num_patches, n_embed)
                where 1.0 marks target patch positions.

        Returns:
            torch.Tensor: Predictor output of shape (batch_size, num_patches, n_embed).
        """
        B, P, D = x.shape

        # target_indices is the (B, P, D) mask expanded in IJepa.forward. Multiplying the
        # learned mask token (shape 1,1,D) by target_indices injects it only at target
        # positions and leaves context positions as zero — so when added to the patch
        # embeddings below, context patches are unchanged while target patches get an extra
        # signal telling the Transformer "you need to predict what goes here".
        target_mask_matrix = self.target_mask.expand(B, P, -1)*target_indices

        x = self.patch_embedding(x)
        x = x + self.positional_embedding + target_mask_matrix
        x = self.vit(x)

        return x