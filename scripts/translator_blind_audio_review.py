"""Create an offline, audio-only review packet for frozen critical ASR cases."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path

from translator_mdc_asr_run import sha256, validate_manifest

REPO = Path(__file__).resolve().parents[1]
MAX_TOTAL_AUDIO_BYTES = 16 * 1024 * 1024


def private_new_dir(path: Path) -> None:
    if (
        not path.is_absolute()
        or path.resolve().is_relative_to(REPO)
        or any(parent.is_symlink() for parent in (path, *path.parents))
        or any((parent / ".git").exists() for parent in path.parents)
        or not path.parent.is_dir()
        or stat.S_IMODE(path.parent.stat().st_mode) & 0o077
        or path.exists()
    ):
        raise ValueError("output must be a new private directory outside Git")


def prepare(
    manifest: Path,
    manifest_hash: str,
    screen: Path,
    screen_hash: str,
    review_dir: Path,
    key_dir: Path,
) -> dict:
    screen_bytes = screen.read_bytes()
    if hashlib.sha256(screen_bytes).hexdigest() != screen_hash:
        raise ValueError("frozen screen changed")
    samples = validate_manifest(manifest, manifest_hash)
    by_key = {(sample["origin_id"], sample["condition"]): sample for sample in samples}
    if len(by_key) != len(samples):
        raise ValueError("duplicate corpus identity")
    screen_data = json.loads(screen_bytes)
    if screen_data["schema"] != 1 or screen_data["manifest_sha256"] != manifest_hash:
        raise ValueError("critical screen does not match frozen corpus")
    selected = screen_data["cases"]
    keys = [(case["origin_id"], case["condition"]) for case in selected]
    if (
        not keys
        or len(keys) > 12
        or len(keys) != len(set(keys))
        or any(key not in by_key for key in keys)
    ):
        raise ValueError("invalid critical-case selection")

    private_new_dir(review_dir)
    private_new_dir(key_dir)
    if (
        review_dir == key_dir
        or review_dir.is_relative_to(key_dir)
        or key_dir.is_relative_to(review_dir)
    ):
        raise ValueError("review and answer-key directories must be separate")

    shuffled = keys.copy()
    secrets.SystemRandom().shuffle(shuffled)
    pack_id = secrets.token_hex(16)
    cards = []
    mapping = []
    total_audio_bytes = 0
    for index, key in enumerate(shuffled, start=1):
        sample = by_key[key]
        audio_path = manifest.parent / sample["audio_file"]
        if audio_path.is_symlink():
            raise ValueError("audio symlink refused")
        remaining = MAX_TOTAL_AUDIO_BYTES - total_audio_bytes
        with audio_path.open("rb") as stream:
            audio = stream.read(remaining + 1)
            too_large = len(audio) > remaining or bool(stream.read(1))
        if too_large:
            raise ValueError("listening packet exceeds audio byte limit")
        total_audio_bytes += len(audio)
        if hashlib.sha256(audio).hexdigest() != sample["sha256"]:
            raise ValueError("frozen audio changed after manifest validation")
        label = f"{index:02d}"
        cards.append(
            f'<section><h2>Clip {label}</h2><audio controls preload="none" '
            f'src="data:audio/wav;base64,{base64.b64encode(audio).decode("ascii")}">'
            '</audio><label>Exact words heard<textarea rows="3" '
            f'data-label="{label}" spellcheck="false"></textarea></label></section>'
        )
        mapping.append(
            {
                "label": label,
                "origin_id": sample["origin_id"],
                "condition": sample["condition"],
                "language": sample["language"],
                "audio_sha256": sample["sha256"],
                "written_reference": sample["reference"],
            }
        )

    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; media-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Blind critical audio review</title>
<style>body{{max-width:48rem;margin:2rem auto;font:1rem sans-serif}}section{{margin:1rem 0;padding:1rem;border:1px solid #aaa}}audio,textarea{{display:block;width:100%;margin:.5rem 0}}textarea{{box-sizing:border-box}}button{{font-size:1rem;padding:.5rem}}</style>
<h1>Blind critical audio review</h1>
<p>Listen before looking at any reference or model output. Transcribe exact words, including names, numbers, and negations. Replay as needed. Enter UNINTELLIGIBLE if speech cannot be resolved. This page works offline; its embedded audio is private.</p>
{"".join(cards)}
<button id="export">Export responses</button>
<script>document.getElementById('export').addEventListener('click',()=>{{
 const responses=[...document.querySelectorAll('textarea[data-label]')].map(x=>({{label:x.dataset.label,heard:x.value.trim()}}));
 if(responses.some(x=>!x.heard)){{alert('Transcribe every clip or enter UNINTELLIGIBLE.');return;}}
 const blob=new Blob([JSON.stringify({{schema:1,pack_id:{json.dumps(pack_id)},responses}},null,2)],{{type:'application/json'}});
 const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='blind-audio-responses.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}});</script></html>
"""
    # The review directory contains audio only; the operator keeps the key in
    # a separate directory. Neither directory is handed over as a combined set.
    key_dir.mkdir(mode=0o700)
    review_dir.mkdir(mode=0o700)
    key_file = key_dir / "answer-key.json"
    page_file = review_dir / "review.html"
    key = {
        "schema": 1,
        "pack_id": pack_id,
        "manifest_sha256": manifest_hash,
        "screen_sha256": screen_hash,
        "review_html_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(),
        "cases": mapping,
    }
    for path, content in (
        (key_file, json.dumps(key, ensure_ascii=False, indent=2)),
        (page_file, page),
    ):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
    return {
        "cases": len(mapping),
        "pack_id": pack_id,
        "review_html_sha256": sha256(page_file),
        "answer_key_sha256": sha256(key_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--screen-sha256", required=True)
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--key-dir", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(
        args.manifest,
        args.manifest_sha256,
        args.screen,
        args.screen_sha256,
        args.review_dir,
        args.key_dir,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
