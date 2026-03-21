import torch
import pytest
from vit import ViT, ViTContextEncoder, ViTPredictor, Tokenizer, PatchEmbedding, TransformerEncoderBlock


def test_tokenizer_encode_decode_roundtrip():
    tokenizer = Tokenizer(img_size=224, patch_size=16)
    x = torch.randn(4, 3, 224, 224)
    encoded = tokenizer.encode(x)
    assert encoded.shape == (4, 196, 16 * 16 * 3), f"Unexpected encode shape: {encoded.shape}"
    decoded = tokenizer.decode(encoded)
    assert decoded.shape == x.shape, f"Roundtrip shape mismatch: {decoded.shape}"


def test_patch_embedding():
    embed = PatchEmbedding(patch_size=16, n_embed=768)
    x = torch.randn(4, 196, 16 * 16 * 3)
    out = embed(x)
    assert out.shape == (4, 196, 768), f"Unexpected shape: {out.shape}"


def test_patch_embedding_custom_channels():
    embed = PatchEmbedding(patch_size=16, n_embed=768, img_channels=1)
    x = torch.randn(4, 196, 16 * 16 * 1)
    out = embed(x)
    assert out.shape == (4, 196, 768)


def test_transformer_encoder_block():
    block = TransformerEncoderBlock(n_embed=768, n_head=12)
    x = torch.randn(4, 196, 768)
    out = block(x)
    assert out.shape == x.shape, f"Shape changed: {out.shape}"


def test_vit_output_shape():
    """Test output shapes for ViT, ViTContextEncoder, and ViTPredictor with default config."""
    B, num_patches, n_embed = 4, 196, 768

    # ViT: operates directly on patch token sequences
    vit = ViT(n_embed=n_embed, n_head=12, n_layers=2)
    x = torch.randn(B, num_patches, n_embed)
    out = vit(x)
    assert out.shape == (B, num_patches, n_embed), f"ViT unexpected shape: {out.shape}"

    # ViTContextEncoder: encodes raw images into patch embeddings
    context_encoder = ViTContextEncoder(img_size=224, patch_size=16, n_embed=n_embed, n_head=12, n_layers=2)
    imgs = torch.randn(B, 3, 224, 224)
    out = context_encoder(imgs)
    assert out.shape == (B, num_patches, n_embed), f"ViTContextEncoder unexpected shape: {out.shape}"

    # ViTPredictor: predicts target patches from context representations
    predictor = ViTPredictor(num_patches=num_patches, n_embed=n_embed, n_head=12, n_layers=2)
    target_indices = torch.zeros(B, num_patches, n_embed)
    target_indices[:, :50, :] = 1.0
    out = predictor(x, target_indices)
    assert out.shape == (B, num_patches, n_embed), f"ViTPredictor unexpected shape: {out.shape}"


def test_vit_single_channel():
    """Test ViT, ViTContextEncoder, and ViTPredictor with non-default configurations."""
    B, num_patches, n_embed = 2, 49, 512

    # ViT: smaller embedding dimension
    vit = ViT(n_embed=n_embed, n_head=8, n_layers=2)
    x = torch.randn(B, num_patches, n_embed)
    out = vit(x)
    assert out.shape == (B, num_patches, n_embed), f"ViT unexpected shape: {out.shape}"

    # ViTContextEncoder: single-channel images with larger patch size
    context_encoder = ViTContextEncoder(img_size=224, patch_size=32, n_embed=n_embed, n_head=8, n_layers=2, img_channels=1)
    imgs = torch.randn(B, 1, 224, 224)
    out = context_encoder(imgs)
    assert out.shape == (B, num_patches, n_embed), f"ViTContextEncoder unexpected shape: {out.shape}"

    # ViTPredictor: smaller embedding dimension
    predictor = ViTPredictor(num_patches=num_patches, n_embed=n_embed, n_head=8, n_layers=2)
    target_indices = torch.zeros(B, num_patches, n_embed)
    target_indices[:, :10, :] = 1.0
    out = predictor(x, target_indices)
    assert out.shape == (B, num_patches, n_embed), f"ViTPredictor unexpected shape: {out.shape}"
