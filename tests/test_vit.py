import torch
import pytest
from vit import ViT, Tokenizer, PatchEmbedding, TransformerEncoderBlock


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
    model = ViT(img_size=224, patch_size=16, n_embed=768, n_head=12, n_layers=2)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 196, 768), f"Unexpected output shape: {out.shape}"



def test_vit_single_channel():
    model = ViT(img_size=224, patch_size=16, n_embed=768, n_head=12, n_layers=2, img_channels=1)
    x = torch.randn(4, 1, 224, 224)
    out = model(x)
    assert out.shape == (4, 196, 768)
