import torch
import pytest
from i_jepa import IJepa


def test_ijepa_initialization():
    """Test that IJepa model initializes correctly."""
    model = IJepa(img_size=224, patch_size=16, n_embed=768, n_head=12, n_layers=2)
    assert model.context_encoder is not None
    assert model.predictor is not None
    assert model.n_embed == 768


def test_ijepa_forward_pass():
    """Test IJepa forward pass with correct output shapes."""
    model = IJepa(img_size=224, patch_size=16, n_embed=768, n_head=12, n_layers=2)

    B, C, H, W = 4, 3, 224, 224
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    # Create target mask (one-hot vector indicating which patches to predict)
    num_patches = (224 // 16) ** 2  # 196
    target_indices = torch.zeros(B, num_patches)
    target_indices[:, :50] = 1  # Mask first 50 patches

    predicted_y, actual_y = model(x, y, target_indices)

    # Both outputs should have shape (B, num_patches, n_embed)
    assert predicted_y.shape == (B, num_patches, 768), f"Unexpected predicted shape: {predicted_y.shape}"
    assert actual_y.shape == (B, num_patches, 768), f"Unexpected actual shape: {actual_y.shape}"


def test_ijepa_masking():
    """Test that masking correctly zeros out non-target patches."""
    model = IJepa(img_size=224, patch_size=16, n_embed=768, n_head=12, n_layers=2)

    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    num_patches = (224 // 16) ** 2
    target_indices = torch.zeros(B, num_patches)
    target_indices[:, 10:20] = 1  # Only mask patches 10-19

    predicted_y, actual_y = model(x, y, target_indices)

    # Non-masked patches should be zero
    assert torch.allclose(predicted_y[:, :10, :], torch.zeros_like(predicted_y[:, :10, :])), \
        "Non-masked patches should be zero in predicted output"
    assert torch.allclose(actual_y[:, :10, :], torch.zeros_like(actual_y[:, :10, :])), \
        "Non-masked patches should be zero in actual output"


def test_ijepa_different_patch_sizes():
    """Test IJepa with different patch sizes."""
    model = IJepa(img_size=224, patch_size=32, n_embed=512, n_head=8, n_layers=2)

    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    num_patches = (224 // 32) ** 2  # 49
    target_indices = torch.zeros(B, num_patches)
    target_indices[:, :10] = 1

    predicted_y, actual_y = model(x, y, target_indices)

    assert predicted_y.shape == (B, num_patches, 512)
    assert actual_y.shape == (B, num_patches, 512)


def test_ijepa_single_batch():
    """Test IJepa with batch size of 1."""
    model = IJepa(img_size=224, patch_size=16, n_embed=768, n_head=12, n_layers=2)

    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3, 224, 224)

    num_patches = (224 // 16) ** 2
    target_indices = torch.zeros(1, num_patches)
    target_indices[:, :50] = 1

    predicted_y, actual_y = model(x, y, target_indices)

    assert predicted_y.shape == (1, num_patches, 768)
    assert actual_y.shape == (1, num_patches, 768)


def test_ijepa_all_patches_masked():
    """Test IJepa when all patches are masked."""
    model = IJepa(img_size=224, patch_size=16, n_embed=768, n_head=12, n_layers=2)

    B = 2
    x = torch.randn(B, 3, 224, 224)
    y = torch.randn(B, 3, 224, 224)

    num_patches = (224 // 16) ** 2
    target_indices = torch.ones(B, num_patches)  # All patches masked

    predicted_y, actual_y = model(x, y, target_indices)

    assert predicted_y.shape == (B, num_patches, 768)
    # No patches should be zero since all are masked
    assert not torch.allclose(predicted_y, torch.zeros_like(predicted_y))


def test_ijepa_no_patches_masked():
    """Test IJepa when no patches are masked."""
    model = IJepa(img_size=224, patch_size=16, n_embed=768, n_head=12, n_layers=2)

    B = 2
    x = torch.randn(B, 3, 224, 224)
    y = torch.randn(B, 3, 224, 224)

    num_patches = (224 // 16) ** 2
    target_indices = torch.zeros(B, num_patches)  # No patches masked

    predicted_y, actual_y = model(x, y, target_indices)

    assert predicted_y.shape == (B, num_patches, 768)
    # All patches should be zero since none are masked
    assert torch.allclose(predicted_y, torch.zeros_like(predicted_y))
    assert torch.allclose(actual_y, torch.zeros_like(actual_y))
