from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import simulator_server


class PrometheusExporterTest(unittest.TestCase):
    def test_rendered_payload_contains_required_metric_families(self) -> None:
        state = simulator_server.ExporterState()
        snapshot = simulator_server.MetricsSnapshot(
            active_neurons_ratio=0.04,
            spikes_per_tick=111.0,
            ei_balance=2.5,
            train_accuracy=0.77,
            test_accuracy=0.74,
            synapse_count=12345.0,
            pruned_total=88.0,
            sprouted_total=55.0,
            stdp_updates_total=999.0,
            tick_duration_seconds=0.00075,
            e_rate_hz=7.5,
            i_rate_hz=3.0,
            training_progress_ratio=0.42,
            training_samples_per_sec=128.0,
            training_eta_seconds=360.0,
        )
        state.observe(snapshot)
        payload = simulator_server.render_metrics_payload(snapshot, state)

        expected_types = [
            "senna_active_neurons_ratio",
            "senna_spikes_per_tick",
            "senna_ei_balance",
            "senna_train_accuracy",
            "senna_test_accuracy",
            "senna_synapse_count",
            "senna_pruned_total",
            "senna_sprouted_total",
            "senna_tick_duration_seconds",
            "senna_training_progress_ratio",
            "senna_training_samples_per_sec",
            "senna_training_eta_seconds",
        ]
        for metric in expected_types:
            self.assertIn(f"# TYPE {metric}", payload)

        self.assertIn('senna_tick_duration_seconds_bucket{le="0.0001"}', payload)
        self.assertIn('senna_tick_duration_seconds_bucket{le="+Inf"}', payload)
        self.assertIn("senna_tick_duration_seconds_sum", payload)
        self.assertIn("senna_tick_duration_seconds_count", payload)

    def test_snapshot_loads_from_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "metrics.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "active_neurons_ratio": 0.04,
                        "spikes_per_tick": 111,
                        "e_rate_hz": 7.5,
                        "i_rate_hz": 3.0,
                        "train_accuracy": 0.77,
                        "test_accuracy": 0.74,
                        "synapse_count": 12345,
                        "pruned_total": 88,
                        "sprouted_total": 55,
                        "stdp_updates_total": 999,
                        "tick_duration_seconds": 0.00075,
                        "training_progress_ratio": 0.42,
                        "training_samples_per_sec": 128.0,
                        "training_eta_seconds": 360.0,
                    }
                ),
                encoding="utf-8",
            )

            snapshot = simulator_server.read_snapshot(snapshot_path=snapshot_path)
            self.assertIsNotNone(snapshot)
            assert snapshot is not None

            self.assertAlmostEqual(snapshot.active_neurons_ratio, 0.04)
            self.assertAlmostEqual(snapshot.spikes_per_tick, 111.0)
            self.assertAlmostEqual(snapshot.ei_balance, 2.5)
            self.assertAlmostEqual(snapshot.train_accuracy, 0.77)
            self.assertAlmostEqual(snapshot.test_accuracy, 0.74)
            self.assertAlmostEqual(snapshot.synapse_count, 12345.0)
            self.assertAlmostEqual(snapshot.pruned_total, 88.0)
            self.assertAlmostEqual(snapshot.sprouted_total, 55.0)
            self.assertAlmostEqual(snapshot.stdp_updates_total, 999.0)
            self.assertAlmostEqual(snapshot.tick_duration_seconds, 0.00075)
            self.assertAlmostEqual(snapshot.training_progress_ratio, 0.42)
            self.assertAlmostEqual(snapshot.training_samples_per_sec, 128.0)
            self.assertAlmostEqual(snapshot.training_eta_seconds, 360.0)

    def test_snapshot_returns_none_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.json"
            snapshot = simulator_server.read_snapshot(snapshot_path=missing_path)
            self.assertIsNone(snapshot)


if __name__ == "__main__":
    unittest.main()
