"""
Train a single LAVA probe for a given concept.

Usage:
    python scripts/train_probe.py \\
        --features path/to/features.pt \\
        --labels path/to/labels.pt \\
        --concept math_correctness \\
        --probe_type mlp \\
        --supervision classification \\
        --d_model 2048 \\
        --output probes/math_correctness

Features file: torch tensor of shape (N, d_in) for classification,
               or (N, 2, d_in) for preference.
Labels file:   torch tensor of shape (N,) with 0/1 values (classification only).
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from lava import (
    ProbeBank,
    ConceptConfig,
    TrainConfig,
    evaluate_probe,
    compute_feature_dim,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train a LAVA probe for one concept.")
    p.add_argument("--features", required=True, help="Path to features .pt file")
    p.add_argument("--labels", default=None, help="Path to labels .pt file (classification only)")
    p.add_argument("--test_features", default=None, help="Path to test features .pt file")
    p.add_argument("--test_labels", default=None, help="Path to test labels .pt file")
    p.add_argument("--concept", required=True, help="Concept name (e.g. math_correctness)")
    p.add_argument("--probe_type", default="mlp", choices=["mlp", "linear"])
    p.add_argument("--supervision", default="classification", choices=["classification", "preference"])
    p.add_argument("--d_model", type=int, required=True, help="Backbone hidden dimension")
    p.add_argument("--n_tail", type=int, default=5, help="Tail tokens to concatenate")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True, help="Directory to save probe bank")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Features may be saved in fp16/bf16 by extract_features.py; probes are fp32.
    features = torch.load(args.features, map_location=args.device).float()
    labels = torch.load(args.labels, map_location=args.device) if args.labels else None

    print(f"Loaded features: {features.shape}")
    if labels is not None:
        print(f"Loaded labels: {labels.shape}, pos_rate={labels.float().mean():.2%}")

    # Validate --d_model against the feature dim (concat: d_in = n_tail * d_model).
    # All-layer features are (N, L+1, d_in); single-layer are (N, d_in).
    feat_dim = features.shape[-1]
    expected_d_model = feat_dim // args.n_tail
    if args.d_model != expected_d_model:
        print(f"WARNING: --d_model {args.d_model} != feat_dim/{args.n_tail} = {expected_d_model}; "
              f"using {expected_d_model} to match the features.")
        args.d_model = expected_d_model
    if features.dim() == 3 and args.supervision == "classification":
        print(f"All-layer features detected: searching {features.shape[1]} layers for L*.")

    bank = ProbeBank(d_model=args.d_model, n_tail=args.n_tail, device=args.device)
    cfg = ConceptConfig(
        name=args.concept,
        probe_type=args.probe_type,
        supervision=args.supervision,
        threshold=args.threshold,
    )
    train_cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    print(f"Training {args.probe_type} probe for concept '{args.concept}'...")
    history = bank.add_concept(cfg, features, labels, train_config=train_cfg, verbose=args.verbose)
    final_loss = history["train_loss_history"][-1]
    print(f"Training complete. Final loss: {final_loss:.4f}")
    if "best_layer" in history:
        print(f"Selected layer L* = {history['best_layer']} (val_acc={history['val_acc']:.2%})")
        print(f"  per-layer val_acc: " +
              ", ".join(f"{i}:{a:.2f}" for i, a in enumerate(history["per_layer_acc"])))

    if args.test_features and args.test_labels:
        test_features = torch.load(args.test_features, map_location=args.device).float()
        test_labels = torch.load(args.test_labels, map_location=args.device)
        # Match the layer the probe was trained on.
        if test_features.dim() == 3 and cfg.best_layer is not None:
            test_features = test_features[:, cfg.best_layer, :]
        probe = bank._probes[0]
        metrics = evaluate_probe(probe, test_features, test_labels, threshold=args.threshold, device=args.device)
        print(f"Test Accuracy: {metrics['accuracy']:.2%}")
        print(f"Test F1 (negative class): {metrics['f1_negative']:.3f}")

    bank.save(args.output)
    print(f"Probe bank saved to '{args.output}'")


if __name__ == "__main__":
    main()
