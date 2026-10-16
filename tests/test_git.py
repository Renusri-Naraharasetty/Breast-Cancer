import os
import json
import pytest
import torch
import torch.nn as nn
from src.models.gat import GraphAttentionLayer, MultimodalGAT
from src.models.git import (
    GraphInfusedTransformer,
    EndToEndMultimodalGIT,
    AttentivePooling,
    CrossModalAttention,
)
from src.data.multimodal_dataset import (
    BCMIDPatientMultimodalDataset,
    collate_patient_multimodal,
    SIDE_TO_INT,
)
from src.models.factory import create_model


def test_gat_layer_mechanics():
    """Verify GAT layer forward pass, shape preservation, and masking."""
    layer = GraphAttentionLayer(in_features=64, out_features=64, num_heads=4, concat=False)
    layer.eval()

    # Synthetic batch of 2 graphs with 5 nodes each
    x = torch.randn(2, 5, 64)
    adj = torch.eye(5).unsqueeze(0).repeat(2, 1, 1)
    adj[:, 0, 1] = 1.0
    adj[:, 1, 0] = 1.0

    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 4] = False  # Node 4 is padded in graph 1

    with torch.no_grad():
        out = layer(x, adj, node_mask=mask)

    assert out.shape == (2, 5, 64)
    # Masked node should have 0 vector
    assert torch.all(out[1, 4] == 0.0)


def test_multimodal_gat_multi_layer():
    """Verify deep MultimodalGAT forward pass and non-linearity."""
    gat = MultimodalGAT(hidden_dim=128, num_layers=3, num_heads=4)
    gat.train()

    x = torch.randn(2, 6, 128)
    adj = torch.eye(6).unsqueeze(0).repeat(2, 1, 1)
    adj[:, 0, 2] = 1.0
    adj[:, 2, 0] = 1.0

    out = gat(x, adj)
    assert out.shape == (2, 6, 128)

    loss = out.sum()
    loss.backward()

    # Verify gradients flow to all 3 layers
    for i, layer in enumerate(gat.layers):
        assert layer.W.weight.grad is not None
        assert torch.norm(layer.W.weight.grad).item() > 0.0


def test_attentive_pooling_and_cross_attention():
    """Verify attentive pooling and cross-attention blocks."""
    hidden_dim = 128
    pool = AttentivePooling(hidden_dim=hidden_dim)
    cross_attn = CrossModalAttention(hidden_dim=hidden_dim, num_heads=4)

    # 2 patients: P1 has 3 views, P2 has 2 views (padded to 3)
    views = torch.randn(2, 3, hidden_dim)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    pooled = pool(views, mask=mask)
    assert pooled.shape == (2, hidden_dim)

    # Cross-attention test
    us_views = torch.randn(2, 4, hidden_dim)
    us_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    us_pool = pool(us_views, mask=us_mask)

    c_mg, c_us = cross_attn(
        mg_summary=pooled.unsqueeze(1),
        us_summary=us_pool.unsqueeze(1),
        mg_views=views,
        us_views=us_views,
        mg_mask=mask,
        us_mask=us_mask,
    )
    assert c_mg.shape == (2, hidden_dim)
    assert c_us.shape == (2, hidden_dim)


def test_git_ipsilateral_anatomical_graph_construction():
    """Verify that same-side views are connected and cross-modality graph is constructed."""
    git = GraphInfusedTransformer(
        mg_feat_dim=1280,
        us_feat_dim=768,
        hidden_dim=64,
        num_gat_layers=1,
        gat_heads=2,
        transformer_layers=1,
        transformer_heads=2,
        dim_feedforward=128,
    )

    B, M, U = 1, 2, 2
    z_g = torch.randn(B, 64)
    z_m = torch.randn(B, 64)
    z_u = torch.randn(B, 64)
    z_f = torch.randn(B, 64)

    mg_tokens = torch.randn(B, M, 64)
    us_tokens = torch.randn(B, U, 64)

    # MG 0 is Left (0), MG 1 is Right (1)
    # US 0 is Left (0), US 1 is Right (1)
    mg_sides = torch.tensor([[0, 1]])
    us_sides = torch.tensor([[0, 1]])

    mg_mask = torch.tensor([[True, True]])
    us_mask = torch.tensor([[True, True]])

    nodes, adj, mask = git._build_patient_graph(
        z_global=z_g,
        z_mg=z_m,
        z_us=z_u,
        z_fused=z_f,
        mg_tokens=mg_tokens,
        us_tokens=us_tokens,
        mg_sides=mg_sides,
        us_sides=us_sides,
        mg_mask=mg_mask,
        us_mask=us_mask,
    )

    # Total nodes = 4 + 2 + 2 = 8 nodes
    assert nodes.shape == (1, 8, 64)
    assert adj.shape == (1, 8, 8)
    assert mask.shape == (1, 8)

    # Nodes:
    # 0: Global CLS, 1: MG summary, 2: US summary, 3: Fused
    # 4: MG 0 (Left), 5: MG 1 (Right)
    # 6: US 0 (Left), 7: US 1 (Right)
    # Ipsilateral checks:
    # MG 0 (Left, index 4) should connect to US 0 (Left, index 6)
    assert adj[0, 4, 6] == 1.0 and adj[0, 6, 4] == 1.0
    # MG 1 (Right, index 5) should connect to US 1 (Right, index 7)
    assert adj[0, 5, 7] == 1.0 and adj[0, 7, 5] == 1.0
    # Contralateral check:
    # MG 0 (Left, index 4) should NOT connect to US 1 (Right, index 7)
    assert adj[0, 4, 7] == 0.0 and adj[0, 7, 4] == 0.0


def test_git_end_to_end_gradient_flow():
    """Verify complete forward pass and backward gradients through GIT."""
    git = GraphInfusedTransformer(
        mg_feat_dim=1280,
        us_feat_dim=768,
        hidden_dim=128,
        num_gat_layers=2,
        gat_heads=4,
        transformer_layers=2,
        transformer_heads=4,
        dim_feedforward=256,
        num_classes=2,
        dropout_rate=0.2,
    )
    git.train()

    B, M, U = 2, 3, 4
    mg_f = torch.randn(B, M, 1280, requires_grad=True)
    us_f = torch.randn(B, U, 768, requires_grad=True)
    mg_s = torch.tensor([[0, 0, 1], [0, 1, 3]])
    us_s = torch.tensor([[0, 0, 1, 1], [0, 1, 3, 3]])
    mg_m = torch.tensor([[True, True, True], [True, True, False]])
    us_m = torch.tensor([[True, True, True, True], [True, True, False, False]])

    logits = git(mg_f, us_f, mg_s, us_s, mg_m, us_m)
    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()

    target = torch.tensor([1, 0], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    # Check gradient exists on classification head
    assert git.classifier[-1].weight.grad is not None
    assert torch.norm(git.classifier[-1].weight.grad).item() > 0.0
    # Check gradient propagated back to projection layers
    assert git.mg_proj[0].weight.grad is not None
    assert git.us_proj[0].weight.grad is not None


def test_model_factory_git_registration():
    """Verify that factory creates GIT model with valid parameter counts."""
    m = create_model("git", num_classes=2)
    assert isinstance(m, GraphInfusedTransformer)
    assert m.num_classes == 2

    # Introspection
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params > 1_000_000, f"Expected > 1M params, got {n_params}"


def test_multimodal_dataset_split_counts_and_zero_leakage():
    """Verify that BCMIDPatientMultimodalDataset preserves the 220/48/47 split with zero leakage."""
    splits_path = "data/splits/patient_stratified_split.json"
    assert os.path.exists(splits_path)

    ds_train = BCMIDPatientMultimodalDataset(split="train")
    ds_val = BCMIDPatientMultimodalDataset(split="val")
    ds_test = BCMIDPatientMultimodalDataset(split="test")

    assert len(ds_train) == 220, f"Expected 220 train patients, got {len(ds_train)}"
    assert len(ds_val) == 48, f"Expected 48 val patients, got {len(ds_val)}"
    assert len(ds_test) == 47, f"Expected 47 test patients, got {len(ds_test)}"

    train_ids = set(p["patient_id"] for p in ds_train.patients)
    val_ids = set(p["patient_id"] for p in ds_val.patients)
    test_ids = set(p["patient_id"] for p in ds_test.patients)

    assert len(train_ids & val_ids) == 0, "Train-Val patient leakage!"
    assert len(train_ids & test_ids) == 0, "Train-Test patient leakage!"
    assert len(val_ids & test_ids) == 0, "Val-Test patient leakage!"
