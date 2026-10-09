import os
import sys
import shutil
import tempfile
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.pooling import AttentionPooling
from src.models.densenet_mammogram import MammogramDenseNet
from src.models.resnet_ultrasound import UltrasoundResNet
from src.models.graph import MultimodalGraphBuilder
from src.models.gat import GATv2Layer, MultimodalGAT
from src.models.transformer import GraphInfusedTransformer
from src.models.fusion_baseline import SimpleFusionBaseline
from src.models.multimodal_model import MultimodalGraphTransformer
from src.training.losses import get_loss_function
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from src.explainability.gradcam import GradCAM, visualize_multimodal_attention


def run_component_verification():
    print("=" * 60)
    print("      RUNNING ARCHITECTURE & COMPONENT VERIFICATION")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TEST 0] Target Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Verify Attention Pooling
    print("[TEST 1] Testing Learnable Attention Pooling...")
    pooler = AttentionPooling(input_dim=128, hidden_dim=64).to(device)
    dummy_feats = torch.randn(7, 128, device=device) # 7 images across 3 patients
    dummy_batch_indices = torch.tensor([0, 0, 1, 1, 1, 2, 2], device=device)
    pooled = pooler(dummy_feats, dummy_batch_indices, batch_size=3)
    assert pooled.shape == (3, 128), f"Expected shape (3, 128), got {pooled.shape}"
    print(" -> PASSED: AttentionPooling aggregates variable views into [B, D].")

    # 2. Verify DenseNet-121 (Mammogram)
    print("[TEST 2] Testing DenseNet-121 Mammogram Backbone...")
    # Using pretrained=False for fast instant shape checking
    mammo_net = MammogramDenseNet(pretrained=False, feature_dim=1024, num_classes=2).to(device)
    dummy_m_imgs = torch.randn(4, 3, 224, 224, device=device)
    dummy_m_idx = torch.tensor([0, 0, 1, 1], device=device)
    m_logits, m_feats = mammo_net(dummy_m_imgs, dummy_m_idx, batch_size=2)
    assert m_feats.shape == (2, 1024), f"Expected mammogram features (2, 1024), got {m_feats.shape}"
    assert m_logits.shape == (2, 2), f"Expected logits (2, 2), got {m_logits.shape}"
    print(" -> PASSED: DenseNet-121 extracts 1024-d embeddings and outputs logits [B, 2].")

    # 3. Verify ResNet-50 (Ultrasound)
    print("[TEST 3] Testing ResNet-50 Ultrasound Backbone...")
    us_net = UltrasoundResNet(pretrained=False, feature_dim=2048, num_classes=2).to(device)
    dummy_u_imgs = torch.randn(5, 3, 224, 224, device=device)
    dummy_u_idx = torch.tensor([0, 0, 0, 1, 1], device=device)
    u_logits, u_feats = us_net(dummy_u_imgs, dummy_u_idx, batch_size=2)
    assert u_feats.shape == (2, 2048), f"Expected ultrasound features (2, 2048), got {u_feats.shape}"
    assert u_logits.shape == (2, 2), f"Expected logits (2, 2), got {u_logits.shape}"
    print(" -> PASSED: ResNet-50 extracts 2048-d embeddings and outputs logits [B, 2].")

    # 4. Verify Multimodal Graph Construction
    print("[TEST 4] Testing Multimodal Graph Construction...")
    graph_builder = MultimodalGraphBuilder(mammo_dim=1024, us_dim=2048, hidden_dim=256).to(device)
    node_tokens, adj_mask = graph_builder(m_feats, u_feats)
    assert node_tokens.shape == (2, 2, 256), f"Expected tokens (2, 2, 256), got {node_tokens.shape}"
    assert adj_mask.shape == (2, 2), f"Expected adj_mask (2, 2), got {adj_mask.shape}"
    print(" -> PASSED: GraphBuilder projects modalities to shared dimension [B, 2, 256].")

    # 5. Verify GATv2
    print("[TEST 5] Testing Multi-Head GATv2 Information Propagation...")
    gat = MultimodalGAT(hidden_dim=256, num_layers=2, num_heads=4, dropout=0.1).to(device)
    graph_tokens, gat_attn = gat(node_tokens, adj_mask)
    assert graph_tokens.shape == (2, 2, 256), f"Expected graph_tokens (2, 2, 256), got {graph_tokens.shape}"
    assert gat_attn.shape == (2, 4, 2, 2), f"Expected GAT attention (2, 4, 2, 2), got {gat_attn.shape}"
    print(" -> PASSED: GATv2 performs cross-modal message passing with attention weights.")

    # 6. Verify Graph-Infused Transformer
    print("[TEST 6] Testing Graph-Infused Transformer...")
    transformer = GraphInfusedTransformer(d_model=256, nhead=4, num_layers=2, dim_feedforward=512).to(device)
    trans_tokens, trans_attns = transformer(graph_tokens)
    assert trans_tokens.shape == (2, 2, 256), f"Expected transformer output (2, 2, 256), got {trans_tokens.shape}"
    assert len(trans_attns) == 2, f"Expected 2 attention matrices, got {len(trans_attns)}"
    assert trans_attns[0].shape == (2, 2, 2), f"Expected attention matrix (2, 2, 2), got {trans_attns[0].shape}"
    print(" -> PASSED: Transformer encoder applies Multi-Head Self-Attention over graph tokens.")

    # 7. Verify Full Multimodal End-to-End Forward Pass
    print("[TEST 7] Testing Full Proposed MultimodalGraphTransformer...")
    multimodal_model = MultimodalGraphTransformer(
        pretrained=False,
        mammo_dim=1024,
        us_dim=2048,
        hidden_dim=256,
        gat_heads=4,
        gat_layers=2,
        transformer_heads=4,
        transformer_layers=2,
        num_classes=2
    ).to(device)
    logits, explanations = multimodal_model(
        dummy_m_imgs, dummy_m_idx,
        dummy_u_imgs, dummy_u_idx,
        batch_size=2,
        return_explanations=True
    )
    assert logits.shape == (2, 2), f"Expected logits (2, 2), got {logits.shape}"
    assert "gat_attention" in explanations, "Explanations missing gat_attention"
    assert "transformer_attention" in explanations, "Explanations missing transformer_attention"
    print(" -> PASSED: Full proposed architecture forward pass and attention extraction.")

    # 8. Verify Loss Computation, Backpropagation, and Optimizer Step
    print("[TEST 8] Testing Loss Computation & Backpropagation...")
    criterion = get_loss_function(class_weights=torch.tensor([1.0, 3.0]), device=device)
    dummy_targets = torch.tensor([0, 1], dtype=torch.long, device=device)
    loss = criterion(logits, dummy_targets)
    assert loss.item() > 0.0, "Loss must be positive"
    
    optimizer = torch.optim.AdamW(multimodal_model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f" -> PASSED: Backward pass and gradient step successful (Loss: {loss.item():.4f}).")

    # 9. Verify Simple Fusion Baseline Forward Pass
    print("[TEST 9] Testing Simple Fusion Baseline...")
    fusion_model = SimpleFusionBaseline(pretrained=False, mammo_dim=1024, us_dim=2048, hidden_dim=256).to(device)
    f_logits = fusion_model(dummy_m_imgs, dummy_m_idx, dummy_u_imgs, dummy_u_idx, batch_size=2)
    assert f_logits.shape == (2, 2), f"Expected fusion logits (2, 2), got {f_logits.shape}"
    print(" -> PASSED: Simple fusion baseline operational.")

    # 10. Verify Medical Metrics & Plots Generation
    print("[TEST 10] Testing Medical Metrics & Plot Generation...")
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.85, 0.3, 0.9, 0.75, 0.15, 0.6, 0.25, 0.05])
    metrics = compute_metrics(y_true, y_prob)
    assert "sensitivity" in metrics or "recall" in metrics, "Missing sensitivity metric"
    assert "specificity" in metrics, "Missing specificity metric"
    assert "roc_auc" in metrics, "Missing ROC-AUC"
    assert "pr_auc" in metrics, "Missing PR-AUC"

    temp_dir = tempfile.mkdtemp()
    try:
        plot_roc_curve(y_true, y_prob, metrics["roc_auc"], os.path.join(temp_dir, "test_roc.png"))
        plot_pr_curve(y_true, y_prob, metrics["pr_auc"], os.path.join(temp_dir, "test_pr.png"))
        plot_confusion_matrix(metrics["confusion_matrix"], os.path.join(temp_dir, "test_cm.png"))
        assert os.path.exists(os.path.join(temp_dir, "test_roc.png"))
        assert os.path.exists(os.path.join(temp_dir, "test_pr.png"))
        assert os.path.exists(os.path.join(temp_dir, "test_cm.png"))
        print(" -> PASSED: Metric computation and plotting routines working flawlessly.")
    finally:
        shutil.rmtree(temp_dir)

    print("=" * 60)
    print("ALL 10 ARCHITECTURAL & COMPONENT TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_component_verification()
