import pytest
import torch
import torch.nn as nn
from src.models import (
    EfficientNetB0Binary,
    ViTB16Binary,
    create_model,
    get_model_info,
)
from src.training.losses import (
    compute_image_level_class_weights,
    get_loss_function,
    FocalLoss,
)


def test_efficientnet_architecture():
    """Verify EfficientNet-B0 binary classification head and forward pass."""
    model = EfficientNetB0Binary(pretrained=False, num_classes=2, dropout_rate=0.3)
    model.eval()

    # Synthetic input batch (batch_size=2, channels=3, height=224, width=224)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        features = model.extract_features(x)

    # Assertions
    assert logits.shape == (2, 2), f"Expected logits (2, 2), got {logits.shape}"
    assert features.shape == (2, 1280), f"Expected features (2, 1280), got {features.shape}"

    # Verify classifier structure
    classifier = model.backbone.classifier
    assert isinstance(classifier[0], nn.Dropout)
    assert classifier[0].p == 0.3
    assert isinstance(classifier[1], nn.Linear)
    assert classifier[1].in_features == 1280
    assert classifier[1].out_features == 2


def test_vit_architecture():
    """Verify ViT-B/16 binary classification head and forward pass."""
    model = ViTB16Binary(pretrained=False, num_classes=2, dropout_rate=0.3)
    model.eval()

    # Synthetic input batch
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        features = model.extract_features(x)

    # Assertions
    assert logits.shape == (2, 2), f"Expected logits (2, 2), got {logits.shape}"
    assert features.shape == (2, 768), f"Expected CLS features (2, 768), got {features.shape}"

    # Verify head structure
    heads = model.backbone.heads
    assert isinstance(heads[0], nn.Dropout)
    assert heads[0].p == 0.3
    assert isinstance(heads[1], nn.Linear)
    assert heads[1].in_features == 768
    assert heads[1].out_features == 2


def test_model_factory():
    """Verify factory instantiation, introspection, and error handling."""
    # Test valid models
    m_eff = create_model("efficientnet_b0", pretrained=False)
    m_vit = create_model("vit_b_16", pretrained=False)

    assert isinstance(m_eff, EfficientNetB0Binary)
    assert isinstance(m_vit, ViTB16Binary)

    info_eff = get_model_info(m_eff)
    info_vit = get_model_info(m_vit)

    assert info_eff["total_parameters"] > 3_000_000
    assert info_vit["total_parameters"] > 80_000_000

    # Test invalid model name
    with pytest.raises(ValueError, match="Unknown model_name"):
        create_model("resnet50")


def test_image_level_class_weights_computation():
    """Verify that class weights are computed from actual training images, not patient ratios."""
    w_mg = compute_image_level_class_weights("mammogram", normalize_w0_to_one=True)
    w_us = compute_image_level_class_weights("ultrasound", normalize_w0_to_one=True)

    # Mammogram actual images: 570 Class 0, 74 Class 1 -> Ratio: 570 / 74 = 7.7027
    assert w_mg[0].item() == 1.0
    assert pytest.approx(w_mg[1].item(), rel=1e-3) == (570.0 / 74.0)

    # Ultrasound actual images: 541 Class 0, 130 Class 1 -> Ratio: 541 / 130 = 4.1615
    assert w_us[0].item() == 1.0
    assert pytest.approx(w_us[1].item(), rel=1e-3) == (541.0 / 130.0)

    # Ensure weights reflect empirical modality differences
    assert w_mg[1].item() > w_us[1].item(), "Mammography should have higher minority weight than ultrasound"


def test_synthetic_loss_and_backward_pass():
    """Verify end-to-end forward pass, loss calculation, and backward gradient flow on synthetic batch."""
    for model_name, mod in [("efficientnet_b0", "mammogram"), ("vit_b_16", "ultrasound")]:
        model = create_model(model_name, pretrained=False)
        model.train()

        loss_fn = get_loss_function(modality=mod, loss_type="weighted_ce")
        focal_fn = get_loss_function(modality=mod, loss_type="focal")

        x = torch.randn(4, 3, 224, 224, requires_grad=False)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        # Forward pass
        logits = model(x)
        loss = loss_fn(logits, y)
        floss = focal_fn(logits, y)

        assert not torch.isnan(loss) and not torch.isinf(loss)
        assert not torch.isnan(floss) and not torch.isinf(floss)
        assert loss.item() > 0.0

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Check gradient flow to final linear layer
        if model_name == "efficientnet_b0":
            head_weight = model.backbone.classifier[1].weight
        else:
            head_weight = model.backbone.heads[1].weight

        assert head_weight.grad is not None
        assert torch.norm(head_weight.grad).item() > 0.0


def test_independent_experiment_isolation():
    """Verify that all four experiment model instances have completely isolated parameters (no fusion)."""
    exp1_model = create_model("efficientnet_b0", pretrained=False)  # Exp 1: EffNet Mammogram
    exp2_model = create_model("vit_b_16", pretrained=False)         # Exp 2: ViT Ultrasound
    exp3_model = create_model("vit_b_16", pretrained=False)         # Exp 3: ViT Mammogram
    exp4_model = create_model("efficientnet_b0", pretrained=False)  # Exp 4: EffNet Ultrasound

    # Check memory isolation between independent models of same architecture
    ptr_exp1 = exp1_model.backbone.classifier[1].weight.data_ptr()
    ptr_exp4 = exp4_model.backbone.classifier[1].weight.data_ptr()
    assert ptr_exp1 != ptr_exp4, "Exp 1 and Exp 4 share classifier parameters!"

    ptr_exp2 = exp2_model.backbone.heads[1].weight.data_ptr()
    ptr_exp3 = exp3_model.backbone.heads[1].weight.data_ptr()
    assert ptr_exp2 != ptr_exp3, "Exp 2 and Exp 3 share classifier parameters!"
