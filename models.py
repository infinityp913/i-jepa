import torch
import torch.nn.functional as F
from torch import nn
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.layers.rotary import apply_rotary_emb_qkv_
from flash_attn.modules.mlp import GatedMlp

class Tokenizer(nn.Module):
    """ A module to tokenize an image into patches and reconstruct it back. """

    def __init__(self, img_size, patch_size):
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        self.fold = nn.Fold(img_size, kernel_size=patch_size, stride=patch_size)
    
    def encode(self, x):
        """
        Encodes an image into patches.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, patch_size * patch_size * channels).
        """
        return self.unfold(x).transpose(-2, -1)
    
    def decode(self, x):
        """
        Decodes the patches back into an image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, patch_size * patch_size * channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        return self.fold(x.transpose(-2, -1))

def _precompute_rope_embeddings(img_size, patch_size, head_dim, base=10_000.0):
    """Computes static 2D RoPE."""
    dim = head_dim // 2

    freqs = torch.pow(base, -torch.arange(0, dim, 2).float() / dim)

    rows, cols = torch.meshgrid(
        torch.arange(img_size[0] // patch_size),
        torch.arange(img_size[1] // patch_size),
        indexing='ij'
    )

    angles = torch.cat([
        rows.flatten()[:, None] * freqs,
        cols.flatten()[:, None] * freqs
    ], dim=-1)

    return torch.cos(angles), torch.sin(angles)

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with fused QKV, RoPE, and SwiGLU FFN."""

    def __init__(self, d_model, n_head):
        """
        Initializes the TransformerBlock.

        Args:
            d_model (int): Model dimension.
            n_head (int): Number of attention heads.
        """
        super().__init__()
        self.n_head = n_head

        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.head_d = d_model // n_head

        self.qkv = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model * 3, bias=False)
        )

        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.RMSNorm(d_model),
            GatedMlp(d_model, hidden_features=d_model * 8 // 3, activation=F.silu, bias1=False, bias2=False)
        )

    def forward(self, x, cos, sin):
        """
        Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): Input patch embeddings [B, N, d_model].
            cos (torch.Tensor): Precomputed 2D RoPE cos [B, N, head_dim//2].
            sin (torch.Tensor): Precomputed 2D RoPE sin [B, N, head_dim//2].
        Returns:
            torch.Tensor: Output patch embeddings [B, N, d_model].
        """
        qkv = self.qkv(x).reshape(len(x), -1, 3, self.n_head, self.head_d)

        apply_rotary_emb_qkv_(
            qkv.flatten(0, 1).unsqueeze(0),
            cos.flatten(0, 1),
            sin.flatten(0, 1)
        )

        x = x + self.proj(flash_attn_qkvpacked_func(qkv).flatten(2))

        return x + self.ffn(x)

class ViT(nn.Module):
    """A Vision Transformer handling RoPE, attention blocks, and normalization."""

    def __init__(self, img_size, patch_size, d_model, n_head, n_layers):
        """
        Initializes the ViT.

        Args:
            img_size (tuple[int, int]): Spatial resolution (H, W) of the input image.
            patch_size (int): Square patch size.
            d_model (int): Vision Transformer dimension.
            n_head (int): Number of attention heads.
            n_layers (int): Number of Vision Transformer blocks.
        """
        super().__init__()
        
        cos, sin = _precompute_rope_embeddings(img_size, patch_size, d_model // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, n_head) for _ in range(n_layers)])
        
        self.norm = nn.RMSNorm(d_model)

    def forward(self, tokens, cos, sin):
        """
        Forward pass of the ViT.

        Args:
            tokens (torch.Tensor): Input sequence embeddings [B, N, d_model].
            cos (torch.Tensor): Pre-sliced cosine frequencies for RoPE.
            sin (torch.Tensor): Pre-sliced sine frequencies for RoPE.
        Returns:
            torch.Tensor: Normalized output embeddings [B, N, d_model].
        """
        for block in self.transformer_blocks: tokens = block(tokens, cos, sin)
            
        return self.norm(tokens)

class Encoder(nn.Module):
    """A Transformer based encoder for encoding images into patch-level embeddings."""

    def __init__(self, img_size, patch_size=16, in_channels=3, d_model=768, n_head=12, n_layers=12):
        """
        Initializes the Encoder.

        Args:
            img_size (tuple[int, int]): Spatial resolution (H, W) of the input image.
            patch_size (int): Square patch size (paper default: 16).
            in_channels (int): Number of input image channels (default: 3 for RGB).
            d_model (int): Encoder dimensions (paper default: 768).
            n_head (int): Number of attention heads (paper default: 12).
            n_layers (int): Number of Transformer encoder blocks (paper default: 12).
        """
        super().__init__()
        self.embed = nn.Linear(patch_size ** 2 * in_channels, d_model, bias=False)
        
        self.core = ViT(img_size, patch_size, d_model, n_head, n_layers)

    def forward(self, x, masks=None):
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Kept patches [B, n_context, patch_size**2 * img_channels].
            masks (torch.Tensor): Patch indices to keep [B, n_context]. Pass None to use all patches.
        Returns:
            torch.Tensor: Embeddings of kept patches [B, n_context, d_model].
        """
        cos = self.core.cos[masks]
        sin = self.core.sin[masks]
        if masks is None:
            broadcast_shape = (len(x), -1, -1)
            cos = cos.expand(*broadcast_shape)
            sin = sin.expand(*broadcast_shape)

        return self.core(self.embed(x), cos, sin)

class Predictor(nn.Module):
    """A Transformer based predictor for predicting the masked patches."""

    def __init__(self, img_size, patch_size=16, in_channels=768, d_model=384, n_head=12, n_layers=6):
        """
        Initializes the Predictor.

        Args:
            img_size (tuple[int, int]): Spatial resolution (H, W) of the input image.
            patch_size (int): Square patch size (paper default: 16).
            in_channels (int): Encoder dimension (paper default: 768).
            d_model (int): Predictor dimension (paper default: 384).
            n_head (int): Number of attention heads.
            n_layers (int): Number of Transformer blocks (paper uses a narrower/shallower predictor).
        """
        super().__init__()
        self.embed = nn.Linear(in_channels, d_model, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.core = ViT(img_size, patch_size, d_model, n_head, n_layers)

        self.proj = nn.Linear(d_model, in_channels, bias=False)

    def forward(self, x, x_masks, y_masks):
        """
        Forward pass of the Predictor.

        Args:
            x (torch.Tensor): Context encoder outputs [B, n_context, embed_dim].
            x_masks (torch.Tensor): Patch indices kept as context by context encoder [B, n_context].
            y_masks (torch.Tensor): Patch indices to predict [B, n_target].
        Returns:
            torch.Tensor: Predicted embeddings at target positions [B, n_target, embed_dim].
        """
        return self.proj(self.core(
            torch.cat([self.embed(x), self.mask_token.expand(*y_masks.shape, -1)], dim=1),
            torch.cat([self.core.cos[x_masks], self.core.cos[y_masks]], dim=1),
            torch.cat([self.core.sin[x_masks], self.core.sin[y_masks]], dim=1)
        )[:, x_masks.shape[1]:])

class ImageClassifier(nn.Module):
    """A Vision Transformer based image classifier."""

    def __init__(self, img_size, num_classes, patch_size=16, in_channels=3, d_model=768, n_head=12, n_layers=12):
        """
        Initializes the ImageClassifier.

        Args:
            img_size (tuple[int, int]): Spatial resolution (H, W) of the input image.
            num_classes (int): Number of classes.
            patch_size (int): Square patch size.
            img_channels (int): Number of input image channels.
            d_model (int): Encoder dimension.
            n_head (int): Number of attention heads.
            n_layers (int): Number of Transformer encoder blocks.
        """
        super().__init__()
        self.feature_extractor = Encoder(img_size, patch_size, in_channels, d_model, n_head, n_layers)

        self.linear_head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, x):
        """
        Forward pass of the ImageClassifier.

        Args:
            x (torch.Tensor): Pre-tokenized patches [B, N, patch_size**2 * img_channels].
        Returns:
            torch.Tensor: Class logits [B, num_classes].
        """
        x = self.feature_extractor(x)

        return self.linear_head(x.mean(dim=1))
