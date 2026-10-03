from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader(
    "preview_package", str(ROOT / "scripts/translator-preview-package")
)
spec = importlib.util.spec_from_loader(loader.name, loader)
package = importlib.util.module_from_spec(spec)
loader.exec_module(package)


class PreviewPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _model(self):
        document = json.loads((ROOT / "models/manifest.json").read_text())
        model = document["models"][0]
        document["models"] = [model]
        model["files"] = [
            {
                "path": "model.bin",
                "size_bytes": 7,
                "sha256": hashlib.sha256(b"payload").hexdigest(),
            }
        ]
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(document))
        cache = self.root / "cache"
        source = cache / model["cache_path"] / "model.bin"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"payload")
        with patch.dict(os.environ, {"TRANSLATOR_MODEL_CACHE_ROOT": str(cache)}):
            return package.load_manifest(manifest), source

    def test_models_are_hash_verified_and_source_is_preserved(self) -> None:
        manifest, source = self._model()
        mode = source.stat().st_mode
        with patch.object(package, "MODEL_IDS", ("faster-whisper-small",)):
            count = package.copy_models(manifest, self.root / "copied")
        copied = next((self.root / "copied").rglob("model.bin"))
        self.assertEqual(count, 7)
        self.assertEqual(copied.read_bytes(), b"payload")
        self.assertEqual(copied.stat().st_mode & 0o777, 0o600)
        self.assertEqual(source.stat().st_mode, mode)

    def test_corrupt_or_missing_model_cannot_be_packaged(self) -> None:
        manifest, source = self._model()
        with patch.object(package, "MODEL_IDS", ("faster-whisper-small",)):
            source.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "pinned manifest"):
                package.copy_models(manifest, self.root / "corrupt")
            source.unlink()
            with self.assertRaisesRegex(RuntimeError, "runtime model file is missing"):
                package.copy_models(manifest, self.root / "missing")

    def _cuda(self) -> Path:
        root = self.root / "site-packages"
        for name, version, component, library in (
            ("nvidia-cudnn-cu12", "9.10.2.21", "cudnn", "libcudnn.so.9"),
            ("nvidia-cuda-nvrtc-cu12", "12.8.93", "cuda_nvrtc", "libnvrtc.so.12"),
        ):
            relative = f"nvidia/{component}/lib/{library}"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"library")
            info = root / f"{name.replace('-', '_')}-{version}.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
            checksum = base64.urlsafe_b64encode(hashlib.sha256(b"library").digest())
            (info / "RECORD").write_text(
                f"{relative},sha256={checksum.decode().rstrip('=')},7\n"
            )
        return root

    def test_cuda_copies_use_vendor_record_and_private_permissions(self) -> None:
        root = self._cuda()
        result = package.copy_cuda(root, self.root / "cuda")
        self.assertEqual(len(result), 2)
        for path in (self.root / "cuda").rglob("*.so.*"):
            self.assertEqual(path.read_bytes(), b"library")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(len(list(root.rglob("*.so.*"))), 2)

    def test_cuda_corruption_is_rejected(self) -> None:
        root = self._cuda()
        next(root.rglob("libcudnn.so.9")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "CUDA.*RECORD"):
            package.copy_cuda(root, self.root / "cuda")

    def test_output_inside_checkout_is_rejected_before_writes(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside.*checkout"):
            package.package(ROOT / "preview-test-output", self.root, "bad", self.root)
        self.assertFalse((ROOT / "preview-test-output").exists())

    def _checkout(self, ui_script: str) -> tuple[Path, str]:
        checkout = self.root / "checkout"
        binaries = checkout / "target/release"
        binaries.mkdir(parents=True)
        for name, script in (
            ("translator-daemon", "exit 0"),
            ("translator-ui", ui_script),
        ):
            binary = binaries / name
            binary.write_text("#!/bin/sh\n" + script + "\n")
            binary.chmod(0o700)
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        tree = subprocess.check_output(
            ["git", "write-tree"], cwd=checkout, text=True
        ).strip()
        return checkout, tree

    def test_development_ui_is_rejected_before_output_or_model_access(self) -> None:
        checkout, tree = self._checkout('test "$1" = --check-bundled-ui; exit 78')
        output = self.root / "output"
        with (
            patch.object(package, "ROOT", checkout),
            patch.object(package, "load_manifest") as load,
            self.assertRaises(subprocess.CalledProcessError) as failed,
        ):
            package.package(output, self.root / "cache", tree, self.root / "cuda")
        self.assertEqual(failed.exception.returncode, 78)
        self.assertEqual(failed.exception.cmd[-1], "--check-bundled-ui")
        load.assert_not_called()
        self.assertFalse(output.exists())

    def test_success_without_ui_asset_receipt_is_rejected(self) -> None:
        checkout, tree = self._checkout("exit 0")
        output = self.root / "output"
        with (
            patch.object(package, "ROOT", checkout),
            patch.object(package, "load_manifest") as load,
            self.assertRaisesRegex(ValueError, "bundled UI"),
        ):
            package.package(output, self.root / "cache", tree, self.root / "cuda")
        load.assert_not_called()
        self.assertFalse(output.exists())

    def test_hanging_ui_probe_cannot_create_package(self) -> None:
        checkout, tree = self._checkout("exit 0")
        output = self.root / "output"
        original_run = subprocess.run

        def run(command, **kwargs):
            if command[-1] == "--check-bundled-ui":
                self.assertLessEqual(kwargs["timeout"], 10)
                self.assertNotIn("DISPLAY", kwargs["env"])
                self.assertNotIn("WAYLAND_DISPLAY", kwargs["env"])
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return original_run(command, **kwargs)

        with (
            patch.object(package, "ROOT", checkout),
            patch.object(package.subprocess, "run", side_effect=run),
            patch.object(package, "load_manifest") as load,
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            package.package(output, self.root / "cache", tree, self.root / "cuda")
        load.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
