"""
Unit tests for LAVA core components.

Run with:  python -m pytest tests/ -v
       or: python tests/test_lava.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import tempfile
import unittest
import torch

from lava.probes import LinearProbe, MLPProbe, build_probe
from lava.feature_extraction import (
    aggregate_hidden_states,
    extract_step_feature,
    compute_feature_dim,
)
from lava.training import train_probe, evaluate_probe, TrainConfig, bradley_terry_loss
from lava.probe_bank import ProbeBank, ConceptConfig
from lava.data_pipeline import (
    decompose_trajectory,
    confidence_filter,
    AnnotatedStep,
    build_classification_dataset,
)
from lava.continual import (
    AccuracyMatrix,
    TaskData,
    run_experiment_a,
    cumulative_false_accept_bound,
    empirical_false_accept_rate,
)


# ---------------------------------------------------------------------------
# Probe architecture tests
# ---------------------------------------------------------------------------

class TestProbes(unittest.TestCase):
    def test_linear_probe_shape(self):
        d_in = 10
        probe = LinearProbe(d_in)
        x = torch.randn(4, d_in)
        logits = probe(x)
        self.assertEqual(logits.shape, (4,))

    def test_linear_probe_score_range(self):
        probe = LinearProbe(16)
        x = torch.randn(100, 16)
        scores = probe.score(x)
        self.assertGreaterEqual(scores.min().item(), 0.0)
        self.assertLessEqual(scores.max().item(), 1.0)

    def test_mlp_probe_shape(self):
        d_in = 32
        probe = MLPProbe(d_in)
        x = torch.randn(8, d_in)
        logits = probe(x)
        self.assertEqual(logits.shape, (8,))

    def test_mlp_probe_score_range(self):
        probe = MLPProbe(64)
        x = torch.randn(20, 64)
        scores = probe.score(x)
        self.assertGreaterEqual(scores.min().item(), 0.0)
        self.assertLessEqual(scores.max().item(), 1.0)

    def test_build_probe_linear(self):
        p = build_probe("linear", 32)
        self.assertIsInstance(p, LinearProbe)

    def test_build_probe_mlp(self):
        p = build_probe("mlp", 32)
        self.assertIsInstance(p, MLPProbe)

    def test_build_probe_unknown(self):
        with self.assertRaises(ValueError):
            build_probe("unknown", 32)

    def test_xavier_init_linear(self):
        probe = LinearProbe(512)
        w = probe.linear.weight
        fan_in, fan_out = w.shape[1], w.shape[0]
        bound = 0.1 * math.sqrt(6.0 / (fan_in + fan_out))
        self.assertLessEqual(w.abs().max().item(), bound + 1e-4)

    def test_zero_bias_linear(self):
        probe = LinearProbe(64)
        self.assertEqual(probe.linear.bias.abs().max().item(), 0.0)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction(unittest.TestCase):
    def test_concat_mode_shape(self):
        L, T, d = 6, 20, 128
        hs = torch.randn(L, T, d)
        f = aggregate_hidden_states(hs, mode="concat", n_tail=5)
        self.assertEqual(f.shape, (5 * d,))

    def test_concat_padding_short_seq(self):
        L, T, d = 6, 3, 128
        hs = torch.randn(L, T, d)
        f = aggregate_hidden_states(hs, mode="concat", n_tail=5)
        self.assertEqual(f.shape, (5 * d,))

    def test_pooling_mode_shape(self):
        L, T, d = 8, 15, 64
        hs = torch.randn(L, T, d)
        f = aggregate_hidden_states(hs, mode="pooling")
        self.assertEqual(f.shape, (d,))

    def test_min_mode_shape(self):
        L, T, d = 8, 15, 64
        hs = torch.randn(L, T, d)
        f = aggregate_hidden_states(hs, mode="min")
        self.assertEqual(f.shape, (d,))

    def test_last_mode_shape(self):
        L, T, d = 4, 10, 32
        hs = torch.randn(L, T, d)
        f = aggregate_hidden_states(hs, mode="last")
        self.assertEqual(f.shape, (d,))

    def test_extract_step_from_list(self):
        d = 64
        T = 10
        layers = [torch.randn(T, d) for _ in range(6)]
        f = extract_step_feature(layers, mode="concat", n_tail=5)
        self.assertEqual(f.shape, (5 * d,))

    def test_compute_feature_dim_concat(self):
        self.assertEqual(compute_feature_dim(d_model=128, mode="concat", n_tail=5), 640)

    def test_compute_feature_dim_pooling(self):
        self.assertEqual(compute_feature_dim(d_model=128, mode="pooling"), 128)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining(unittest.TestCase):
    def _make_data(self, N=200, d=40):
        torch.manual_seed(0)
        X = torch.randn(N, d)
        y = (X[:, 0] > 0).float()
        return X, y

    def test_classification_loss_decreases(self):
        X, y = self._make_data()
        probe = MLPProbe(40, hidden=(32, 16))
        cfg = TrainConfig(epochs=20, batch_size=32, lr=1e-2)
        hist = train_probe(probe, X, y, mode="classification", config=cfg)
        self.assertLess(hist["train_loss_history"][-1], hist["train_loss_history"][0])

    def test_evaluate_probe_accuracy(self):
        torch.manual_seed(42)
        d = 20
        X_test = torch.randn(100, d)
        y_test = (X_test[:, 0] > 0).float()
        probe = LinearProbe(d)
        cfg = TrainConfig(epochs=200, batch_size=32, lr=5e-2)
        train_probe(probe, X_test, y_test, mode="classification", config=cfg)
        metrics = evaluate_probe(probe, X_test, y_test)
        self.assertGreater(metrics["accuracy"], 0.7)

    def test_preference_loss_shape(self):
        logits_pos = torch.randn(8)
        logits_neg = torch.randn(8)
        loss = bradley_terry_loss(logits_pos, logits_neg)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0)

    def test_preference_training(self):
        torch.manual_seed(0)
        d = 20
        N = 100
        f_pos = torch.randn(N, d) + 2
        f_neg = torch.randn(N, d) - 2
        features = torch.stack([f_pos, f_neg], dim=1)
        probe = LinearProbe(d)
        cfg = TrainConfig(epochs=50, batch_size=32, lr=1e-2)
        hist = train_probe(probe, features, labels=None, mode="preference", config=cfg)
        self.assertLess(hist["train_loss_history"][-1], hist["train_loss_history"][0])


# ---------------------------------------------------------------------------
# ProbeBank tests
# ---------------------------------------------------------------------------

class TestProbeBank(unittest.TestCase):
    def _make_bank(self):
        return ProbeBank(d_model=16, n_tail=5, device="cpu")

    def _make_concept_data(self, N=150, d_in=80):
        torch.manual_seed(1)
        X = torch.randn(N, d_in)
        y = (X[:, 0] > 0).float()
        return X, y

    def test_add_concept_increments_k(self):
        bank = self._make_bank()
        d_in = bank.n_tail * bank.d_model
        X, y = self._make_concept_data(d_in=d_in)
        cfg = ConceptConfig(name="test", probe_type="linear")
        bank.add_concept(cfg, X, y, train_config=TrainConfig(epochs=5))
        self.assertEqual(bank.k, 1)

    def test_probe_frozen_after_add(self):
        bank = self._make_bank()
        d_in = bank.n_tail * bank.d_model
        X, y = self._make_concept_data(d_in=d_in)
        cfg = ConceptConfig(name="frozen_test", probe_type="linear")
        bank.add_concept(cfg, X, y, train_config=TrainConfig(epochs=5))
        for p in bank._probes[0].parameters():
            self.assertFalse(p.requires_grad)

    def test_verify_accept_shape(self):
        bank = self._make_bank()
        d_in = bank.n_tail * bank.d_model
        X, y = self._make_concept_data(d_in=d_in)
        cfg = ConceptConfig(name="c1", probe_type="linear", threshold=0.0)
        bank.add_concept(cfg, X, y, train_config=TrainConfig(epochs=5))
        f = torch.randn(d_in)
        accept, scores = bank.verify(f)
        self.assertIsInstance(accept, bool)
        self.assertEqual(scores.shape, (1,))

    def test_verify_batch(self):
        bank = self._make_bank()
        d_in = bank.n_tail * bank.d_model
        X, y = self._make_concept_data(d_in=d_in)
        cfg = ConceptConfig(name="c1", probe_type="linear", threshold=0.5)
        bank.add_concept(cfg, X, y, train_config=TrainConfig(epochs=5))
        F = torch.randn(10, d_in)
        mask, scores = bank.verify_batch(F)
        self.assertEqual(mask.shape, (10,))
        self.assertEqual(scores.shape, (10, 1))

    def test_bwt_zero(self):
        """Adding a second probe must not change the first probe's output (Prop 6.1)."""
        bank = self._make_bank()
        d_in = bank.n_tail * bank.d_model
        X1, y1 = self._make_concept_data(d_in=d_in)
        X2, y2 = self._make_concept_data(N=150, d_in=d_in)

        cfg1 = ConceptConfig(name="c1", probe_type="linear")
        bank.add_concept(cfg1, X1, y1, train_config=TrainConfig(epochs=5))

        test_f = torch.randn(20, d_in)
        scores_before = bank._probes[0].score(test_f).detach().clone()

        cfg2 = ConceptConfig(name="c2", probe_type="linear")
        bank.add_concept(cfg2, X2, y2, train_config=TrainConfig(epochs=5))

        scores_after = bank._probes[0].score(test_f).detach()
        self.assertTrue(torch.allclose(scores_before, scores_after))

    def test_save_load(self):
        bank = self._make_bank()
        d_in = bank.n_tail * bank.d_model
        X, y = self._make_concept_data(d_in=d_in)
        cfg = ConceptConfig(name="save_test", probe_type="mlp")
        bank.add_concept(cfg, X, y, train_config=TrainConfig(epochs=5))

        with tempfile.TemporaryDirectory() as tmpdir:
            bank.save(tmpdir)
            loaded = ProbeBank.load(tmpdir, device="cpu")

        self.assertEqual(loaded.k, 1)
        self.assertEqual(loaded.concept_names(), ["save_test"])
        f = torch.randn(d_in)
        a1, s1 = bank.verify(f)
        a2, s2 = loaded.verify(f)
        self.assertEqual(a1, a2)
        self.assertTrue(torch.allclose(s1, s2))


# ---------------------------------------------------------------------------
# Data pipeline tests
# ---------------------------------------------------------------------------

class TestDataPipeline(unittest.TestCase):
    def test_decompose_trajectory(self):
        traj = "Step 1: do this.\n\nStep 2: do that.\n\nStep 3: done."
        steps = decompose_trajectory(traj)
        self.assertEqual(len(steps), 3)

    def test_decompose_unitary(self):
        traj = "A single unsafe response."
        steps = decompose_trajectory(traj, unitary=True)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0], traj)

    def test_confidence_filter(self):
        items = [
            AnnotatedStep("ctx", "step", 1, confidence=0.9),
            AnnotatedStep("ctx", "step", 0, confidence=0.6),
            AnnotatedStep("ctx", "step", 1, confidence=0.85),
        ]
        filtered = confidence_filter(items, gamma=0.8)
        self.assertEqual(len(filtered), 2)

    def test_build_classification_dataset(self):
        d = 32
        steps = []
        for i in range(20):
            hs = torch.randn(1, 8, d)
            steps.append(AnnotatedStep("ctx", "step", i % 2, confidence=0.9, hidden_states=hs))
        features, labels = build_classification_dataset(steps, mode="concat", n_tail=3)
        self.assertEqual(features.shape, (20, 3 * d))
        self.assertEqual(labels.shape, (20,))


# ---------------------------------------------------------------------------
# Continual learning tests
# ---------------------------------------------------------------------------

class TestContinual(unittest.TestCase):
    def _make_task(self, name, d_in, N=200):
        torch.manual_seed(hash(name) % 1000)
        X = torch.randn(N, d_in)
        y = (X[:, 0] > 0).float()
        return TaskData(
            name=name,
            supervision="classification",
            train_features=X[:150],
            train_labels=y[:150],
            test_features=X[150:],
            test_labels=y[150:],
        )

    def test_bwt_zero_by_construction(self):
        """BWT must be 0 for LAVA (Prop. 6.1)."""
        d_model, n_tail = 8, 5
        d_in = n_tail * d_model
        tasks = [self._make_task(f"task_{i}", d_in, N=160) for i in range(3)]
        cfg = TrainConfig(epochs=20, device="cpu")
        _, matrix = run_experiment_a(
            tasks, d_model=d_model, n_tail=n_tail,
            probe_type="linear", train_config=cfg, threshold=0.5,
        )
        self.assertAlmostEqual(matrix.bwt(), 0.0, places=6)

    def test_accuracy_matrix_summary_keys(self):
        d_model, n_tail = 8, 5
        d_in = n_tail * d_model
        tasks = [self._make_task(f"t{i}", d_in, N=120) for i in range(2)]
        cfg = TrainConfig(epochs=10, device="cpu")
        _, matrix = run_experiment_a(tasks, d_model=d_model, n_tail=n_tail, train_config=cfg)
        summary = matrix.summary()
        for key in ("ACC", "BWT", "FWT"):
            self.assertIn(key, summary)

    def test_cumulative_fa_bound(self):
        bound = cumulative_false_accept_bound([0.1, 0.2, 0.15])
        self.assertAlmostEqual(bound, 0.1 * 0.2 * 0.15, places=9)

    def test_empirical_fa_rate_in_range(self):
        d_model, n_tail = 8, 5
        d_in = n_tail * d_model
        bank = ProbeBank(d_model=d_model, n_tail=n_tail, device="cpu")
        X, y = torch.randn(100, d_in), torch.zeros(100)
        cfg = ConceptConfig(name="c", probe_type="linear", threshold=0.5)
        bank.add_concept(cfg, X, y, train_config=TrainConfig(epochs=5))
        invalid = torch.randn(50, d_in)
        rate = empirical_false_accept_rate(bank, invalid)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
