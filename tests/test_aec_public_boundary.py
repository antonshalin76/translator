"""External-crate checks for the sealed AEC publication API."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_IMPORTS = """
use translator_audio::{AecProofReadyMeasurement, AecValidationInput};
use translator_daemon::{AecCalibrationChallenge, AecCalibrationCoordinator, AecProofReadyInput};
fn visible(_: Option<(AecProofReadyMeasurement, AecValidationInput,
                      AecCalibrationChallenge, AecCalibrationCoordinator, AecProofReadyInput)>) {}
fn main() {}
"""

FORBIDDEN = (
    (
        "audio proof fabrication",
        """
use translator_audio::{AecProofReadyMeasurement, AecValidationInput};
fn fabricate(raw: AecValidationInput) -> AecProofReadyMeasurement {
    AecProofReadyMeasurement { input: raw }
}
fn main() {}
""",
        "E0451",
        "AecProofReadyMeasurement",
    ),
    (
        "daemon proof fabrication",
        """
use translator_audio::AecValidationInput;
use translator_daemon::AecProofReadyInput;
fn fabricate(raw: AecValidationInput) -> AecProofReadyInput {
    AecProofReadyInput { input: raw }
}
fn main() {}
""",
        "E0451",
        "AecProofReadyInput",
    ),
    (
        "raw coordinator publication",
        """
use translator_audio::AecValidationInput;
use translator_daemon::{AecCalibrationChallenge, AecCalibrationCoordinator};
fn publish_raw(coordinator: &AecCalibrationCoordinator,
               challenge: &AecCalibrationChallenge, raw: AecValidationInput) {
    let _ = coordinator.publish(challenge, raw, true, true);
}
fn main() {}
""",
        "E0308",
        "AecProofReadyInput",
    ),
    (
        "native isolated evidence promotion",
        """
use translator_audio::{AecProofReadyMeasurement, AecSampleEvidence};
fn promote(native: AecSampleEvidence) -> AecProofReadyMeasurement {
    native
}
fn main() {}
""",
        "E0308",
        "AecProofReadyMeasurement",
    ),
)


class AecPublicBoundaryTests(unittest.TestCase):
    def test_external_crate_cannot_fabricate_or_publish_raw_aec_measurement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="translator-aec-public-api-"
        ) as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "Cargo.toml").write_text(
                '[package]\nname = "translator-aec-public-boundary"\n'
                'version = "0.0.0"\nedition = "2021"\n'
                "[dependencies]\n"
                f'translator-audio = {{ path = "{ROOT / "crates/translator-audio"}" }}\n'
                f'translator-daemon = {{ path = "{ROOT / "crates/translator-daemon"}" }}\n'
            )
            source = root / "src/main.rs"
            environment = os.environ.copy()
            environment["CARGO_BUILD_JOBS"] = "1"
            environment.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))

            def check(program: str) -> subprocess.CompletedProcess[str]:
                source.write_text(program)
                return subprocess.run(
                    [
                        "cargo",
                        "check",
                        "--offline",
                        "--quiet",
                        "--manifest-path",
                        str(root / "Cargo.toml"),
                    ],
                    cwd=root,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )

            available = check(PUBLIC_IMPORTS)
            assert available.returncode == 0, available.stderr

            for name, program, code, expected_type in FORBIDDEN:
                rejected = check(program)
                assert rejected.returncode != 0, f"{name} unexpectedly compiled"
                assert f"error[{code}]" in rejected.stderr, rejected.stderr
                assert expected_type in rejected.stderr, rejected.stderr
