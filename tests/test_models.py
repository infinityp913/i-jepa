import torch
import pytest
from models import ViT, Tokenizer, TransformerBlock, Encoder, Predictor


def test_tokenizer_encode_decode_roundtrip():
    tokenizer = Tokenizer(img_size=224, patch_size=16)
    x = torch.randn(4, 3, 224, 224)
    encoded = tokenizer.encode(x)
    assert encoded.shape == (4, 196, 16 * 16 * 3), f"Unexpected encode shape: {encoded.shape}"
    decoded = tokenizer.decode(encoded)
    assert decoded.shape == x.shape, f"Roundtrip shape mismatch: {decoded.shape}"


def test_transformer_block():
    block = TransformerBlock(d_model=768, n_head=12)
    x = torch.randn(4, 196, 768)
    out = block(x)
    assert out.shape == x.shape, f"Shape changed: {out.shape}"


def test_encoder_all_patches():
    num_patches = 196
    encoder = Encoder(num_patches=num_patches, patch_size=16, img_channels=3, d_model=768, n_head=12, n_layers=2)
    x = torch.randn(4, num_patches, 16 * 16 * 3)
    out = encoder(x)
    assert out.shape == (4, num_patches, 768), f"Unexpected shape: {out.shape}"


def test_encoder_with_masks():
    num_patches, n_ctx = 196, 100
    encoder = Encoder(num_patches=num_patches, patch_size=16, img_channels=3, d_model=768, n_head=12, n_layers=2)
    x = torch.randn(4, n_ctx, 16 * 16 * 3)
    masks = torch.stack([torch.randperm(num_patches)[:n_ctx] for _ in range(4)])
    out = encoder(x, masks)
    assert out.shape == (4, n_ctx, 768), f"Unexpected shape: {out.shape}"


def test_vit_output_shape():
    num_patches, num_classes = 196, 10
    model = ViT(num_patches=num_patches, num_classes=num_classes, patch_size=16, n_head=12, n_layers=2)
    x = torch.randn(4, num_patches, 16 * 16 * 3)
    out = model(x)
    assert out.shape == (4, num_classes), f"Unexpected output shape: {out.shape}"


def test_vit_single_channel():
    num_patches, num_classes = 196, 10
    model = ViT(num_patches=num_patches, num_classes=num_classes, patch_size=16, img_channels=1, n_head=12, n_layers=2)
    x = torch.randn(4, num_patches, 16 * 16 * 1)
    out = model(x)
    assert out.shape == (4, num_classes)


def test_predictor():
    num_patches, n_ctx, n_target = 196, 100, 20
    predictor = Predictor(num_patches=num_patches, encoder_dim=768, d_model=384, n_head=6, n_layers=2)
    x = torch.randn(4, n_ctx, 768)
    x_masks = torch.stack([torch.randperm(num_patches)[:n_ctx] for _ in range(4)])
    y_masks = torch.stack([torch.randperm(num_patches)[:n_target] for _ in range(4)])
    out = predictor(x, x_masks, y_masks)
    assert out.shape == (4, n_target, 768), f"Unexpected shape: {out.shape}"
