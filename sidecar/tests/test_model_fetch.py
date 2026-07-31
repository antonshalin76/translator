from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread
import traceback

import pytest

from translator_sidecar.local.model_fetch import (
    FetchError,
    ModelFetcher,
    UrllibTransport,
)
from translator_sidecar.local.model_manifest import (
    FilesystemOps,
    ManifestError,
    ModelDownloader,
    load_manifest,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(tmp_path: Path, payload: bytes = b"payload") -> Path:
    revision = "a" * 40
    document = {
        "schema_version": 1,
        "policy": {
            "download_budget_bytes": 2 * 1024**3,
            "post_download_free_floor_bytes": 1,
            "usage_mode": "personal_noncommercial",
            "redistribution": False,
            "certified_or_safety_critical": False,
            "staging_path": str(tmp_path / "cache" / ".staging"),
            "redirect_hosts": [
                "cdn-lfs.huggingface.co",
                "us.aws.cdn.hf.co",
            ],
        },
        "models": [
            {
                "id": "selected-mt",
                "role": "mt",
                "source": {
                    "repository": "owner/model",
                    "revision": revision,
                    "license": "CC-BY-NC-4.0",
                },
                "languages": ["ru", "en"],
                "cache_path": str(tmp_path / "cache"),
                "acquisition": "download",
                "files": [
                    {
                        "path": "model.bin",
                        "size_bytes": len(payload),
                        "sha256": _sha256(payload),
                        "source_url": (
                            "https://huggingface.co/owner/model/resolve/"
                            f"{revision}/model.bin"
                        ),
                        "source_path": "model.bin",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: dict[str, str],
        payload: bytes = b"",
        fail_read_at: int | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.status = status
        self.headers = headers
        self.payload = payload
        self.fail_read_at = fail_read_at
        self.close_error = close_error
        self.offset = 0
        self.read_sizes: list[int] = []
        self.closed = False

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if self.fail_read_at == len(self.read_sizes):
            raise OSError("private read detail")
        block = self.payload[self.offset : self.offset + size]
        self.offset += len(block)
        return block

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakeTransport:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, str], float]] = []

    def open(
        self, url: str, *, headers: dict[str, str], timeout_seconds: float
    ) -> FakeResponse:
        self.requests.append((url, headers, timeout_seconds))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingDownloader(ModelDownloader):
    def __init__(self, manifest: object) -> None:
        super().__init__(manifest)  # type: ignore[arg-type]
        self.chunk_sizes: list[int] = []

    def install_bytes(
        self, model_id: str, file_path: str, chunks: object
    ) -> Path:
        def recording_chunks() -> object:
            for chunk in chunks:  # type: ignore[union-attr]
                self.chunk_sizes.append(len(chunk))
                yield chunk

        return super().install_bytes(
            model_id,
            file_path,
            recording_chunks(),  # type: ignore[arg-type]
        )


def _add_config_file(manifest_path: Path, payload: bytes = b"{}") -> None:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = document["models"][0]
    revision = model["source"]["revision"]
    model["files"].append(
        {
            "path": "config.json",
            "size_bytes": len(payload),
            "sha256": _sha256(payload),
            "source_url": (
                "https://huggingface.co/owner/model/resolve/"
                f"{revision}/config.json"
            ),
            "source_path": "config.json",
        }
    )
    manifest_path.write_text(json.dumps(document), encoding="utf-8")


def _add_missing_reused_model(manifest_path: Path, tmp_path: Path) -> None:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["models"].append(
        {
            "id": "reused-asr",
            "role": "asr",
            "source": {
                "repository": "owner/reused",
                "revision": "b" * 40,
                "license": "MIT",
            },
            "languages": ["ru", "en"],
            "cache_path": str(tmp_path / "missing-reused"),
            "acquisition": "reuse",
            "files": [
                {
                    "path": "model.bin",
                    "size_bytes": 1,
                    "sha256": _sha256(b"x"),
                }
            ],
        }
    )
    manifest_path.write_text(json.dumps(document), encoding="utf-8")


def _fetcher(
    tmp_path: Path,
    transport: FakeTransport,
    *,
    payload: bytes = b"payload",
    chunk_size: int = 3,
) -> tuple[ModelFetcher, RecordingDownloader]:
    manifest = load_manifest(_manifest(tmp_path, payload))
    downloader = RecordingDownloader(manifest)
    return (
        ModelFetcher(
            manifest,
            downloader=downloader,
            transport=transport,
            chunk_size=chunk_size,
            timeout_seconds=17,
        ),
        downloader,
    )


def test_default_transport_does_not_automatically_follow_redirects() -> None:
    observed_paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            observed_paths.append(self.path)
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "/followed")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        transport = UrllibTransport()
        response = transport.open(
            f"http://127.0.0.1:{server.server_port}/start",
            headers={"Accept-Encoding": "identity"},
            timeout_seconds=2,
        )
        try:
            assert response.status == 302
        finally:
            response.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert observed_paths == ["/start"]


def test_fetch_validates_redirect_before_requesting_and_streams_bounded_chunks(
    tmp_path: Path,
) -> None:
    payload = b"payload"
    redirect = (
        "https://cdn-lfs.huggingface.co/repos/owner/model/"
        "signed-model.bin?token=opaque"
    )
    first = FakeResponse(status=302, headers={"Location": redirect})
    final = FakeResponse(
        status=200,
        headers={
            "Content-Length": str(len(payload)),
            "Content-Encoding": "identity",
        },
        payload=payload,
    )
    transport = FakeTransport([first, final])
    fetcher, downloader = _fetcher(tmp_path, transport)

    installed = fetcher.fetch("selected-mt", "model.bin")

    assert installed.read_bytes() == payload
    assert [request[0] for request in transport.requests] == [
        (
            "https://huggingface.co/owner/model/resolve/"
            f"{'a' * 40}/model.bin"
        ),
        redirect,
    ]
    assert all(
        request[1]["Accept-Encoding"] == "identity"
        for request in transport.requests
    )
    assert all(request[2] == 17 for request in transport.requests)
    assert downloader.chunk_sizes == [3, 3, 1]
    assert final.read_sizes == [3, 3, 3, 3]
    for _, headers, _ in transport.requests:
        names = {name.lower() for name in headers}
        assert names.isdisjoint(
            {"authorization", "cookie", "proxy-authorization", "referer"}
        )
    assert first.closed
    assert final.closed


def test_fetch_accepts_exact_pinned_huggingface_relative_cache_route(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    cache_route = (
        f"/api/resolve-cache/models/owner/model/{revision}/model.bin"
        "?etag=private-token"
    )
    cdn_route = (
        "https://cdn-lfs.huggingface.co/model.bin?token=private-cdn-token"
    )
    first = FakeResponse(status=307, headers={"Location": cache_route})
    second = FakeResponse(status=302, headers={"Location": cdn_route})
    final = FakeResponse(
        status=200,
        headers={"Content-Length": "7"},
        payload=b"payload",
    )
    transport = FakeTransport([first, second, final])
    fetcher, _ = _fetcher(tmp_path, transport)

    installed = fetcher.fetch("selected-mt", "model.bin")

    assert installed.read_bytes() == b"payload"
    assert [request[0] for request in transport.requests] == [
        (
            "https://huggingface.co/owner/model/resolve/"
            f"{revision}/model.bin"
        ),
        (
            "https://huggingface.co/api/resolve-cache/models/"
            f"owner/model/{revision}/model.bin?etag=private-token"
        ),
        cdn_route,
    ]


def test_fetch_accepts_observed_us_aws_huggingface_cdn_host(
    tmp_path: Path,
) -> None:
    cdn_route = (
        "https://us.aws.cdn.hf.co/repos/owner/model.bin"
        "?token=private-token"
    )
    first = FakeResponse(status=302, headers={"Location": cdn_route})
    final = FakeResponse(
        status=200,
        headers={"Content-Length": "7"},
        payload=b"payload",
    )
    transport = FakeTransport([first, final])
    fetcher, _ = _fetcher(tmp_path, transport)

    assert fetcher.fetch("selected-mt", "model.bin").read_bytes() == b"payload"
    assert [request[0] for request in transport.requests] == [
        (
            "https://huggingface.co/owner/model/resolve/"
            f"{'a' * 40}/model.bin"
        ),
        cdn_route,
    ]


def test_huggingface_cache_route_uses_nested_source_path_not_local_path(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    path = _manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    model_file = document["models"][0]["files"][0]
    model_file["source_path"] = "nested/source/model.bin"
    model_file["source_url"] = (
        "https://huggingface.co/owner/model/resolve/"
        f"{revision}/nested/source/model.bin"
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    manifest = load_manifest(path)
    cache_route = (
        "/api/resolve-cache/models/owner/model/"
        f"{revision}/nested%2Fsource%2Fmodel.bin?etag=private-token"
    )
    response = FakeResponse(status=307, headers={"Location": cache_route})
    final = FakeResponse(
        status=200,
        headers={"Content-Length": "7"},
        payload=b"payload",
    )
    transport = FakeTransport([response, final])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    installed = fetcher.fetch("selected-mt", "model.bin")

    assert installed.name == "model.bin"
    assert transport.requests[1][0].endswith(cache_route)


def test_huggingface_cache_route_rejects_nested_source_path_mutation(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    path = _manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    model_file = document["models"][0]["files"][0]
    model_file["source_path"] = "nested/source/model.bin"
    model_file["source_url"] = (
        "https://huggingface.co/owner/model/resolve/"
        f"{revision}/nested/source/model.bin"
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    manifest = load_manifest(path)
    response = FakeResponse(
        status=307,
        headers={
            "Location": (
                "/api/resolve-cache/models/owner/model/"
                f"{revision}/nested%2Fother%2Fmodel.bin?token=private-token"
            )
        },
    )
    transport = FakeTransport([response])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="redirect"):
        fetcher.fetch("selected-mt", "model.bin")

    assert len(transport.requests) == 1


def test_huggingface_cache_route_rejects_unencoded_nested_source_path(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    path = _manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    model_file = document["models"][0]["files"][0]
    model_file["source_path"] = "nested/source/model.bin"
    model_file["source_url"] = (
        "https://huggingface.co/owner/model/resolve/"
        f"{revision}/nested/source/model.bin"
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    manifest = load_manifest(path)
    response = FakeResponse(
        status=307,
        headers={
            "Location": (
                "/api/resolve-cache/models/owner/model/"
                f"{revision}/nested/source/model.bin?token=private-token"
            )
        },
    )
    transport = FakeTransport([response])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="redirect"):
        fetcher.fetch("selected-mt", "model.bin")

    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "cache_route",
    [
        (
            "/api/resolve-cache/models/attacker/model/"
            f"{'a' * 40}/model.bin"
        ),
        (
            "/api/resolve-cache/models/owner/model/"
            f"{'b' * 40}/model.bin"
        ),
        (
            "/api/resolve-cache/models/owner/model/"
            f"{'a' * 40}/other.bin"
        ),
        "/api/resolve-cache/models/owner/model",
    ],
)
def test_fetch_rejects_unpinned_huggingface_cache_route_before_request(
    tmp_path: Path, cache_route: str
) -> None:
    first = FakeResponse(
        status=307,
        headers={"Location": f"{cache_route}?token=private-token"},
    )
    transport = FakeTransport([first])
    fetcher, _ = _fetcher(tmp_path, transport)

    with pytest.raises(ManifestError, match="redirect") as raised:
        fetcher.fetch("selected-mt", "model.bin")

    assert "private-token" not in str(raised.value)
    assert len(transport.requests) == 1
    assert first.closed


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_fetch_rejects_unapproved_redirect_without_requesting_it(
    tmp_path: Path, status: int
) -> None:
    private_url = "https://attacker.invalid/model.bin?token=private-token"
    first = FakeResponse(
        status=status,
        headers={"Location": private_url},
    )
    transport = FakeTransport([first])
    fetcher, _ = _fetcher(tmp_path, transport)

    with pytest.raises(ManifestError, match="redirect") as raised:
        fetcher.fetch("selected-mt", "model.bin")

    assert private_url not in str(raised.value)
    assert "private-token" not in str(raised.value)
    assert len(transport.requests) == 1
    assert first.closed


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_fetch_accepts_standard_redirect_statuses_after_url_validation(
    tmp_path: Path, status: int
) -> None:
    redirect = (
        "https://cdn-lfs.huggingface.co/model.bin?"
        f"status={status}&token=private-token"
    )
    first = FakeResponse(status=status, headers={"Location": redirect})
    final = FakeResponse(
        status=200,
        headers={"Content-Length": "7"},
        payload=b"payload",
    )
    transport = FakeTransport([first, final])
    fetcher, _ = _fetcher(tmp_path, transport)

    assert fetcher.fetch("selected-mt", "model.bin").read_bytes() == b"payload"
    assert [request[0] for request in transport.requests] == [
        (
            "https://huggingface.co/owner/model/resolve/"
            f"{'a' * 40}/model.bin"
        ),
        redirect,
    ]
    for _, headers, _ in transport.requests:
        names = {name.lower() for name in headers}
        assert names.isdisjoint(
            {"authorization", "cookie", "proxy-authorization", "referer"}
        )
    assert first.closed
    assert final.closed


@pytest.mark.parametrize(
    ("status", "headers", "match"),
    [
        (200, {}, "Content-Length"),
        (200, {"Content-Length": "6"}, "Content-Length"),
        (200, {"Content-Length": "-7"}, "Content-Length"),
        (200, {"Content-Length": "not-an-int"}, "Content-Length"),
        (200, {"Content-Length": str(2**65)}, "Content-Length"),
        (
            200,
            {"Content-Length": "7", "Content-Encoding": "gzip"},
            "Content-Encoding",
        ),
        (206, {"Content-Length": "7"}, "status"),
        (300, {"Content-Length": "7"}, "status"),
        (304, {"Content-Length": "7"}, "status"),
        (305, {"Location": "https://private.example/secret"}, "status"),
        (306, {"Location": "https://private.example/secret"}, "status"),
        (302, {}, "Location"),
        (404, {"Content-Length": "21"}, "status"),
    ],
)
def test_fetch_rejects_ambiguous_or_unexpected_response_metadata(
    tmp_path: Path,
    status: int,
    headers: dict[str, str],
    match: str,
) -> None:
    response = FakeResponse(
        status=status,
        headers=headers,
        payload=b"private-spoken-marker",
    )
    transport = FakeTransport([response])
    fetcher, _ = _fetcher(tmp_path, transport)

    with pytest.raises(FetchError, match=match) as raised:
        fetcher.fetch("selected-mt", "model.bin")

    rendered = str(raised.value)
    assert "private-spoken-marker" not in rendered
    assert "private.example" not in rendered
    assert "secret" not in rendered
    assert response.closed
    assert response.read_sizes == []
    assert not (tmp_path / "cache" / "model.bin").exists()


def test_fetch_bounds_redirect_count_and_closes_every_response(
    tmp_path: Path,
) -> None:
    statuses = [301, 303, 307]
    secrets = ["first-secret", "second-secret", "third-secret"]
    responses = [
        FakeResponse(
            status=status,
            headers={
                "Location": (
                    "https://cdn-lfs.huggingface.co/model.bin?"
                    f"redirect={index}&token={secret}"
                )
            },
        )
        for index, (status, secret) in enumerate(zip(statuses, secrets))
    ]
    transport = FakeTransport(responses.copy())
    manifest = load_manifest(_manifest(tmp_path))
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
        max_redirects=2,
    )

    with pytest.raises(FetchError, match="redirect") as raised:
        fetcher.fetch("selected-mt", "model.bin")

    assert len(transport.requests) == 3
    assert "redirect=2" not in str(raised.value)
    assert all(secret not in str(raised.value) for secret in secrets)
    assert all(response.closed for response in responses)


def test_fetch_maps_connection_failure_without_creating_target(
    tmp_path: Path,
) -> None:
    transport = FakeTransport([OSError("private transport detail")])
    fetcher, _ = _fetcher(tmp_path, transport)

    with pytest.raises(FetchError, match="transport") as raised:
        fetcher.fetch("selected-mt", "model.bin")

    assert "private transport detail" not in str(raised.value)
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private transport detail" not in rendered
    assert not (tmp_path / "cache" / "model.bin").exists()


def test_fetch_closes_response_and_cleans_staging_after_midstream_failure(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        status=200,
        headers={"Content-Length": "7"},
        payload=b"payload",
        fail_read_at=2,
    )
    fetcher, _ = _fetcher(tmp_path, FakeTransport([response]))

    with pytest.raises(ManifestError, match="transport") as raised:
        fetcher.fetch("selected-mt", "model.bin")

    assert "private read detail" not in str(raised.value)
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private read detail" not in rendered
    assert response.closed
    assert list((tmp_path / "cache" / ".staging").glob("*.part")) == []
    assert not (tmp_path / "cache" / "model.bin").exists()


def test_fetch_closes_response_and_cleans_staging_after_checksum_rejection(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        status=200,
        headers={"Content-Length": "7"},
        payload=b"corrupt",
    )
    fetcher, _ = _fetcher(tmp_path, FakeTransport([response]))

    with pytest.raises(ManifestError, match="checksum"):
        fetcher.fetch("selected-mt", "model.bin")

    assert response.closed
    assert list((tmp_path / "cache" / ".staging").glob("*.part")) == []
    assert not (tmp_path / "cache" / "model.bin").exists()


def test_fetch_ignores_sanitized_close_failure_after_successful_commit(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        status=200,
        headers={"Content-Length": "7"},
        payload=b"payload",
        close_error=OSError("private close detail"),
    )
    fetcher, _ = _fetcher(tmp_path, FakeTransport([response]))

    installed = fetcher.fetch("selected-mt", "model.bin")

    assert installed.read_bytes() == b"payload"
    assert response.closed


def test_fetch_preserves_primary_error_when_response_close_also_fails(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        status=404,
        headers={"Content-Length": "21"},
        payload=b"private-spoken-marker",
        close_error=OSError("private close detail"),
    )
    fetcher, _ = _fetcher(tmp_path, FakeTransport([response]))

    with pytest.raises(FetchError, match="status") as raised:
        fetcher.fetch("selected-mt", "model.bin")

    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private close detail" not in rendered
    assert "private-spoken-marker" not in rendered


class NoSpaceFilesystemOps(FilesystemOps):
    def available_bytes_fd(self, descriptor: int) -> int:
        return 0


class CrossDeviceFilesystemOps(FilesystemOps):
    def device_id_fd(self, descriptor: int) -> int:
        path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        return 1 if path.name == ".staging" else 2


def test_fetch_runs_real_downloader_free_space_gate_before_network(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(_manifest(tmp_path))
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(
            manifest,
            filesystem=NoSpaceFilesystemOps(),
        ),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="space"):
        fetcher.fetch("selected-mt", "model.bin")

    assert transport.requests == []


def test_fetch_runs_persistent_ledger_budget_gate_before_network(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(_manifest(tmp_path))
    staging = manifest.policy.staging_path
    staging.mkdir(parents=True)
    (staging / ".download-ledger.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transferred_bytes": manifest.policy.download_budget_bytes,
            }
        ),
        encoding="ascii",
    )
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="budget"):
        fetcher.fetch("selected-mt", "model.bin")

    assert transport.requests == []


def test_fetch_runs_filesystem_identity_gate_before_network(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(_manifest(tmp_path))
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(
            manifest,
            filesystem=CrossDeviceFilesystemOps(),
        ),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="filesystem"):
        fetcher.fetch("selected-mt", "model.bin")

    assert transport.requests == []


@pytest.mark.parametrize("chunk_size", [0, -1, 8 * 1024 * 1024 + 1])
def test_fetch_rejects_invalid_chunk_size_before_network(
    tmp_path: Path, chunk_size: int
) -> None:
    transport = FakeTransport([])
    manifest = load_manifest(_manifest(tmp_path))

    with pytest.raises(ValueError, match="chunk"):
        ModelFetcher(
            manifest,
            downloader=ModelDownloader(manifest),
            transport=transport,
            chunk_size=chunk_size,
        )

    assert transport.requests == []


@pytest.mark.parametrize(
    ("model_id", "file_path"),
    [("unknown", "model.bin"), ("selected-mt", "unknown.bin")],
)
def test_fetch_rejects_unknown_allowlist_identity_before_network(
    tmp_path: Path, model_id: str, file_path: str
) -> None:
    transport = FakeTransport([])
    fetcher, _ = _fetcher(tmp_path, transport)

    with pytest.raises(ManifestError, match="unknown|allowlist"):
        fetcher.fetch(model_id, file_path)

    assert transport.requests == []


def test_fetch_rejects_reused_model_before_network(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    model = document["models"][0]
    model["acquisition"] = "reuse"
    model["files"][0].pop("source_url")
    model["files"][0].pop("source_path")
    path.write_text(json.dumps(document), encoding="utf-8")
    manifest = load_manifest(path)
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="download"):
        fetcher.fetch("selected-mt", "model.bin")

    assert transport.requests == []


def test_fetch_all_skips_only_an_integrity_valid_existing_target(
    tmp_path: Path,
) -> None:
    payload = b"payload"
    manifest = load_manifest(_manifest(tmp_path, payload))
    target = tmp_path / "cache" / "model.bin"
    target.parent.mkdir()
    target.write_bytes(payload)
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    installed = fetcher.fetch_all()

    assert installed == (target.resolve(),)
    assert transport.requests == []


def test_direct_fetch_skips_integrity_valid_existing_target(
    tmp_path: Path,
) -> None:
    payload = b"payload"
    manifest = load_manifest(_manifest(tmp_path, payload))
    target = tmp_path / "cache" / "model.bin"
    target.parent.mkdir()
    target.write_bytes(payload)
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    installed = fetcher.fetch("selected-mt", "model.bin")

    assert installed == target.resolve()
    assert transport.requests == []


def test_direct_fetch_rejects_corrupt_existing_target_before_network(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(_manifest(tmp_path))
    target = tmp_path / "cache" / "model.bin"
    target.parent.mkdir()
    target.write_bytes(b"corrupt")
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="checksum"):
        fetcher.fetch("selected-mt", "model.bin")

    assert transport.requests == []


def test_fetch_all_rejects_invalid_existing_target_without_network(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(_manifest(tmp_path))
    target = tmp_path / "cache" / "model.bin"
    target.parent.mkdir()
    target.write_bytes(b"corrupt")
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="checksum"):
        fetcher.fetch_all()

    assert transport.requests == []


def test_fetch_all_preflights_every_existing_target_before_network(
    tmp_path: Path,
) -> None:
    path = _manifest(tmp_path)
    _add_config_file(path)
    manifest = load_manifest(path)
    corrupt = tmp_path / "cache" / "config.json"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"xx")
    transport = FakeTransport([])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    with pytest.raises(ManifestError, match="checksum"):
        fetcher.fetch_all()

    assert transport.requests == []
    assert not (tmp_path / "cache" / "model.bin").exists()


def test_fetch_all_downloads_only_missing_files_after_global_preflight(
    tmp_path: Path,
) -> None:
    path = _manifest(tmp_path)
    _add_config_file(path)
    manifest = load_manifest(path)
    existing = tmp_path / "cache" / "model.bin"
    existing.parent.mkdir()
    existing.write_bytes(b"payload")
    response = FakeResponse(
        status=200,
        headers={"Content-Length": "2"},
        payload=b"{}",
    )
    transport = FakeTransport([response])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    installed = fetcher.fetch_all()

    assert installed == (
        existing.resolve(),
        (tmp_path / "cache" / "config.json").resolve(),
    )
    assert len(transport.requests) == 1
    assert transport.requests[0][0].endswith("/config.json")


def test_fetch_all_filters_reused_assets_in_mixed_manifest(
    tmp_path: Path,
) -> None:
    path = _manifest(tmp_path)
    _add_missing_reused_model(path, tmp_path)
    manifest = load_manifest(path)
    response = FakeResponse(
        status=200,
        headers={"Content-Length": "7"},
        payload=b"payload",
    )
    transport = FakeTransport([response])
    fetcher = ModelFetcher(
        manifest,
        downloader=ModelDownloader(manifest),
        transport=transport,
    )

    installed = fetcher.fetch_all()

    assert installed == ((tmp_path / "cache" / "model.bin").resolve(),)
    assert len(transport.requests) == 1
    assert "owner/model" in transport.requests[0][0]
