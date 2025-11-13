"""
Run Manifest Schema

Single, versioned "run manifest" (JSON or YAML) per training run with:
- Config fingerprint
- Dataset fingerprint(s)
- Code commit SHA
- Environment versions
- Paths to: training logs, evaluation results, export artifacts, CoreML performance benchmarks
- Phase 0/1/2 gate pass/fail status
- Key metrics with thresholds

@author: @darianrosebrook
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import json
import yaml
from datetime import datetime


class GateStatus(Enum):
    """Gate pass/fail status."""
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"
    SKIP = "skip"


@dataclass
class PhaseGateStatus:
    """Phase gate status."""
    phase: str  # "phase0", "phase1", "phase2"
    status: GateStatus
    gates: Dict[str, GateStatus] = field(default_factory=dict)
    notes: str = ""


@dataclass
class MetricThreshold:
    """Metric with threshold."""
    name: str
    value: float
    threshold: float
    unit: str = ""
    passed: bool = True


@dataclass
class RunManifest:
    """Run manifest schema."""
    # Versioning
    schema_version: str = "1.0"
    run_id: str = ""
    created_at: str = ""
    
    # Config & Data
    config_fingerprint: str = ""
    dataset_fingerprints: List[str] = field(default_factory=list)
    code_commit_sha: str = ""
    
    # Environment
    environment_versions: Dict[str, str] = field(default_factory=dict)
    
    # Artifacts
    training_logs_path: Optional[str] = None
    evaluation_results_path: Optional[str] = None
    export_artifacts_path: Optional[str] = None
    coreml_benchmarks_path: Optional[str] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    
    # Gate Status
    phase_gates: List[PhaseGateStatus] = field(default_factory=list)
    
    # Metrics
    key_metrics: List[MetricThreshold] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert enums to strings
        for phase_gate in data["phase_gates"]:
            phase_gate["status"] = phase_gate["status"].value if isinstance(phase_gate["status"], GateStatus) else phase_gate["status"]
            if "gates" in phase_gate:
                phase_gate["gates"] = {
                    k: v.value if isinstance(v, GateStatus) else v
                    for k, v in phase_gate["gates"].items()
                }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        """Create from dictionary."""
        # Convert string enums back
        for phase_gate in data.get("phase_gates", []):
            if isinstance(phase_gate.get("status"), str):
                phase_gate["status"] = GateStatus(phase_gate["status"])
            if "gates" in phase_gate:
                phase_gate["gates"] = {
                    k: GateStatus(v) if isinstance(v, str) else v
                    for k, v in phase_gate["gates"].items()
                }
        
        return cls(**data)
    
    def save_json(self, path: Path):
        """Save as JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_yaml(self, path: Path):
        """Save as YAML."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load_json(cls, path: Path) -> "RunManifest":
        """Load from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def load_yaml(cls, path: Path) -> "RunManifest":
        """Load from YAML."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def get_phase_status(self, phase: str) -> Optional[PhaseGateStatus]:
        """Get status for a specific phase."""
        for pg in self.phase_gates:
            if pg.phase == phase:
                return pg
        return None
    
    def all_phases_passed(self) -> bool:
        """Check if all phases passed."""
        for pg in self.phase_gates:
            if pg.status != GateStatus.PASS:
                return False
        return True
    
    def get_failed_gates(self) -> List[str]:
        """Get list of failed gates."""
        failed = []
        for pg in self.phase_gates:
            if pg.status == GateStatus.FAIL:
                failed.append(f"{pg.phase}: overall")
            for gate_name, gate_status in pg.gates.items():
                if gate_status == GateStatus.FAIL:
                    failed.append(f"{pg.phase}:{gate_name}")
        return failed


def create_run_manifest(
    run_id: str,
    config_fingerprint: str,
    dataset_fingerprints: List[str],
    code_commit_sha: str = "",
) -> RunManifest:
    """
    Create a new run manifest.
    
    Args:
        run_id: Unique run identifier
        config_fingerprint: Config hash/fingerprint
        dataset_fingerprints: List of dataset fingerprints
        code_commit_sha: Git commit SHA
        
    Returns:
        New run manifest
    """
    import subprocess
    import platform
    import sys
    
    # Get environment versions
    env_versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    
    # Try to get commit SHA if not provided
    if not code_commit_sha:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            code_commit_sha = result.stdout.strip()
        except Exception:
            pass
    
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now().isoformat(),
        config_fingerprint=config_fingerprint,
        dataset_fingerprints=dataset_fingerprints,
        code_commit_sha=code_commit_sha,
        environment_versions=env_versions,
    )


if __name__ == "__main__":
    # Example usage
    manifest = create_run_manifest(
        run_id="test_run_001",
        config_fingerprint="abc123",
        dataset_fingerprints=["def456", "ghi789"],
    )
    
    # Add phase gates
    manifest.phase_gates.append(PhaseGateStatus(
        phase="phase0",
        status=GateStatus.PASS,
        gates={
            "attention_mask": GateStatus.PASS,
            "config_loading": GateStatus.PASS,
            "version_gates": GateStatus.PASS,
        },
    ))
    
    # Add metrics
    manifest.key_metrics.append(MetricThreshold(
        name="loss",
        value=0.5,
        threshold=1.0,
        unit="",
        passed=True,
    ))
    
    # Save
    manifest.save_json(Path("test_manifest.json"))
    print("Created example run manifest")

