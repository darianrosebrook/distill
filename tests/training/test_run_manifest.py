"""
Tests for training/run_manifest.py - Run manifest schema for training runs.

Tests manifest creation, serialization, gate status, and metrics.
"""
# @author: @darianrosebrook

import pytest
import json
import yaml
from pathlib import Path
from training.run_manifest import (
    GateStatus,
    PhaseGateStatus,
    MetricThreshold,
    RunManifest,
)


class TestGateStatus:
    """Test GateStatus enum."""

    def test_gate_status_values(self):
        """Test GateStatus enum values."""
        assert GateStatus.PASS.value == "pass"
        assert GateStatus.FAIL.value == "fail"
        assert GateStatus.PENDING.value == "pending"
        assert GateStatus.SKIP.value == "skip"


class TestPhaseGateStatus:
    """Test PhaseGateStatus dataclass."""

    def test_phase_gate_status_creation(self):
        """Test creating PhaseGateStatus."""
        gate_status = PhaseGateStatus(
            phase="phase0",
            status=GateStatus.PASS,
            gates={"test": GateStatus.PASS},
            notes="All tests passed",
        )
        assert gate_status.phase == "phase0"
        assert gate_status.status == GateStatus.PASS
        assert gate_status.gates["test"] == GateStatus.PASS
        assert gate_status.notes == "All tests passed"

    def test_phase_gate_status_defaults(self):
        """Test PhaseGateStatus with defaults."""
        gate_status = PhaseGateStatus(phase="phase0", status=GateStatus.PENDING)
        assert gate_status.gates == {}
        assert gate_status.notes == ""


class TestMetricThreshold:
    """Test MetricThreshold dataclass."""

    def test_metric_threshold_creation(self):
        """Test creating MetricThreshold."""
        metric = MetricThreshold(
            name="coverage",
            value=85.0,
            threshold=80.0,
            unit="%",
            passed=True,
        )
        assert metric.name == "coverage"
        assert metric.value == 85.0
        assert metric.threshold == 80.0
        assert metric.unit == "%"
        assert metric.passed == True

    def test_metric_threshold_defaults(self):
        """Test MetricThreshold with defaults."""
        metric = MetricThreshold(name="coverage", value=85.0, threshold=80.0)
        assert metric.unit == ""
        assert metric.passed == True


class TestRunManifest:
    """Test RunManifest class."""

    def test_run_manifest_creation(self):
        """Test creating RunManifest."""
        manifest = RunManifest(
            run_id="run-001",
            config_fingerprint="abc123",
            code_commit_sha="def456",
        )
        assert manifest.run_id == "run-001"
        assert manifest.config_fingerprint == "abc123"
        assert manifest.code_commit_sha == "def456"
        assert manifest.schema_version == "1.0"

    def test_run_manifest_defaults(self):
        """Test RunManifest with defaults."""
        manifest = RunManifest()
        assert manifest.schema_version == "1.0"
        assert manifest.run_id == ""
        assert manifest.dataset_fingerprints == []
        assert manifest.phase_gates == []
        assert manifest.key_metrics == []

    def test_to_dict(self):
        """Test converting manifest to dictionary."""
        manifest = RunManifest(
            run_id="run-001",
            phase_gates=[
                PhaseGateStatus(phase="phase0", status=GateStatus.PASS)
            ],
        )
        data = manifest.to_dict()

        assert data["run_id"] == "run-001"
        assert data["phase_gates"][0]["phase"] == "phase0"
        assert data["phase_gates"][0]["status"] == "pass"  # Enum converted to string

    def test_from_dict(self):
        """Test creating manifest from dictionary."""
        data = {
            "run_id": "run-001",
            "config_fingerprint": "abc123",
            "phase_gates": [
                {
                    "phase": "phase0",
                    "status": "pass",
                    "gates": {"test": "pass"},
                    "notes": "",
                }
            ],
        }
        manifest = RunManifest.from_dict(data)

        assert manifest.run_id == "run-001"
        assert len(manifest.phase_gates) == 1
        assert manifest.phase_gates[0].status == GateStatus.PASS

    def test_save_json(self, tmp_path):
        """Test saving manifest as JSON."""
        manifest = RunManifest(run_id="run-001")
        output_path = tmp_path / "manifest.json"

        manifest.save_json(output_path)

        assert output_path.exists()
        with open(output_path, "r") as f:
            data = json.load(f)
        assert data["run_id"] == "run-001"

    def test_save_yaml(self, tmp_path):
        """Test saving manifest as YAML."""
        manifest = RunManifest(run_id="run-001")
        output_path = tmp_path / "manifest.yaml"

        manifest.save_yaml(output_path)

        assert output_path.exists()
        with open(output_path, "r") as f:
            data = yaml.safe_load(f)
        assert data["run_id"] == "run-001"

    def test_load_json(self, tmp_path):
        """Test loading manifest from JSON."""
        output_path = tmp_path / "manifest.json"
        data = {
            "run_id": "run-001",
            "config_fingerprint": "abc123",
            "schema_version": "1.0",
        }
        with open(output_path, "w") as f:
            json.dump(data, f)

        manifest = RunManifest.load_json(output_path)

        assert manifest.run_id == "run-001"
        assert manifest.config_fingerprint == "abc123"

    def test_load_yaml(self, tmp_path):
        """Test loading manifest from YAML."""
        output_path = tmp_path / "manifest.yaml"
        data = {
            "run_id": "run-001",
            "config_fingerprint": "abc123",
            "schema_version": "1.0",
        }
        with open(output_path, "w") as f:
            yaml.dump(data, f)

        manifest = RunManifest.load_yaml(output_path)

        assert manifest.run_id == "run-001"
        assert manifest.config_fingerprint == "abc123"

    def test_get_phase_status(self):
        """Test getting status for specific phase."""
        manifest = RunManifest(
            phase_gates=[
                PhaseGateStatus(phase="phase0", status=GateStatus.PASS),
                PhaseGateStatus(phase="phase1", status=GateStatus.PENDING),
            ]
        )

        phase0_status = manifest.get_phase_status("phase0")
        assert phase0_status is not None
        assert phase0_status.status == GateStatus.PASS

        phase1_status = manifest.get_phase_status("phase1")
        assert phase1_status is not None
        assert phase1_status.status == GateStatus.PENDING

        phase2_status = manifest.get_phase_status("phase2")
        assert phase2_status is None

    def test_add_phase_gate(self):
        """Test adding phase gate."""
        manifest = RunManifest()
        manifest.add_phase_gate("phase0", GateStatus.PASS)

        assert len(manifest.phase_gates) == 1
        assert manifest.phase_gates[0].phase == "phase0"
        assert manifest.phase_gates[0].status == GateStatus.PASS

    def test_add_phase_gate_with_gates(self):
        """Test adding phase gate with sub-gates."""
        manifest = RunManifest()
        manifest.add_phase_gate(
            "phase0",
            GateStatus.PASS,
            gates={"test": GateStatus.PASS, "lint": GateStatus.PASS},
        )

        assert len(manifest.phase_gates) == 1
        assert len(manifest.phase_gates[0].gates) == 2

    def test_add_metric(self):
        """Test adding metric."""
        manifest = RunManifest()
        manifest.add_metric("coverage", 85.0, 80.0, unit="%")

        assert len(manifest.key_metrics) == 1
        assert manifest.key_metrics[0].name == "coverage"
        assert manifest.key_metrics[0].value == 85.0
        assert manifest.key_metrics[0].threshold == 80.0
        assert manifest.key_metrics[0].passed == True

    def test_add_metric_failed(self):
        """Test adding metric that failed threshold."""
        manifest = RunManifest()
        manifest.add_metric("coverage", 75.0, 80.0, unit="%")

        assert manifest.key_metrics[0].passed == False

    def test_round_trip_json(self, tmp_path):
        """Test round-trip JSON serialization."""
        manifest = RunManifest(
            run_id="run-001",
            config_fingerprint="abc123",
            phase_gates=[
                PhaseGateStatus(phase="phase0", status=GateStatus.PASS)
            ],
            key_metrics=[
                MetricThreshold(name="coverage", value=85.0, threshold=80.0)
            ],
        )

        output_path = tmp_path / "manifest.json"
        manifest.save_json(output_path)

        loaded = RunManifest.load_json(output_path)

        assert loaded.run_id == manifest.run_id
        assert loaded.config_fingerprint == manifest.config_fingerprint
        assert len(loaded.phase_gates) == len(manifest.phase_gates)
        assert len(loaded.key_metrics) == len(manifest.key_metrics)

    def test_round_trip_yaml(self, tmp_path):
        """Test round-trip YAML serialization."""
        manifest = RunManifest(
            run_id="run-001",
            config_fingerprint="abc123",
        )

        output_path = tmp_path / "manifest.yaml"
        manifest.save_yaml(output_path)

        loaded = RunManifest.load_yaml(output_path)

        assert loaded.run_id == manifest.run_id
        assert loaded.config_fingerprint == manifest.config_fingerprint

    def test_manifest_with_all_fields(self):
        """Test manifest with all fields populated."""
        manifest = RunManifest(
            run_id="run-001",
            config_fingerprint="abc123",
            dataset_fingerprints=["fp1", "fp2"],
            code_commit_sha="def456",
            environment_versions={"python": "3.11", "torch": "2.0"},
            training_logs_path="logs/training.log",
            evaluation_results_path="results/eval.json",
            export_artifacts_path="artifacts/model.pt",
            coreml_benchmarks_path="benchmarks/coreml.json",
            checkpoint_paths=["checkpoints/ckpt1.pt", "checkpoints/ckpt2.pt"],
            phase_gates=[
                PhaseGateStatus(phase="phase0", status=GateStatus.PASS),
                PhaseGateStatus(phase="phase1", status=GateStatus.PENDING),
            ],
            key_metrics=[
                MetricThreshold(name="coverage", value=85.0, threshold=80.0),
                MetricThreshold(name="loss", value=0.5, threshold=1.0),
            ],
            metadata={"key": "value"},
        )

        assert manifest.run_id == "run-001"
        assert len(manifest.dataset_fingerprints) == 2
        assert len(manifest.checkpoint_paths) == 2
        assert len(manifest.phase_gates) == 2
        assert len(manifest.key_metrics) == 2


class TestRunManifestIntegration:
    """Test integration of run manifest components."""

    def test_complete_manifest_workflow(self, tmp_path):
        """Test complete manifest workflow."""
        manifest = RunManifest(run_id="run-001")

        # Add phase gates
        manifest.add_phase_gate("phase0", GateStatus.PASS)
        manifest.add_phase_gate("phase1", GateStatus.PENDING)

        # Add metrics
        manifest.add_metric("coverage", 85.0, 80.0, unit="%")
        manifest.add_metric("loss", 0.5, 1.0)

        # Save and load
        output_path = tmp_path / "manifest.json"
        manifest.save_json(output_path)

        loaded = RunManifest.load_json(output_path)

        # Verify
        assert loaded.run_id == "run-001"
        assert len(loaded.phase_gates) == 2
        assert len(loaded.key_metrics) == 2

        # Get phase status
        phase0 = loaded.get_phase_status("phase0")
        assert phase0.status == GateStatus.PASS

