"""Allowlisted HTTPS transport for model manifest installations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Protocol
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from translator_sidecar.local.model_manifest import (
    ManifestError,
    ModelDownloader,
    ModelFile,
    ModelManifest,
)


_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_CONTENT_LENGTH_RE = re.compile(r"^\d+$")
_MAX_CHUNK_SIZE = 8 * 1024 * 1024


class FetchError(ManifestError):
    """A model response failed a transport safety check."""


class ResponseHeaders(Protocol):
    def get_all(self, name: str, failobj: object = None) -> list[str] | None: ...


class HttpResponse(Protocol):
    status: int
    headers: ResponseHeaders | Mapping[str, str]

    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...


class HttpTransport(Protocol):
    def open(
        self, url: str, *, headers: dict[str, str], timeout_seconds: float
    ) -> HttpResponse: ...


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


@dataclass(slots=True)
class _UrllibResponse:
    _response: object

    @property
    def status(self) -> int:
        status = getattr(self._response, "status", None)
        if status is None:
            status = getattr(self._response, "code")
        return int(status)

    @property
    def headers(self) -> ResponseHeaders:
        return getattr(self._response, "headers")

    def read(self, size: int) -> bytes:
        return getattr(self._response, "read")(size)

    def close(self) -> None:
        getattr(self._response, "close")()


class UrllibTransport:
    """urllib transport that exposes redirects instead of following them."""

    def __init__(self) -> None:
        self._opener = build_opener(_NoRedirectHandler())

    def open(
        self, url: str, *, headers: dict[str, str], timeout_seconds: float
    ) -> HttpResponse:
        request = Request(url, headers=headers, method="GET")
        try:
            response = self._opener.open(request, timeout=timeout_seconds)
        except HTTPError as error:
            response = error
        return _UrllibResponse(response)


class ModelFetcher:
    def __init__(
        self,
        manifest: ModelManifest,
        *,
        downloader: ModelDownloader,
        transport: HttpTransport | None = None,
        chunk_size: int = 1024 * 1024,
        timeout_seconds: float = 30,
        max_redirects: int = 5,
    ) -> None:
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
            or chunk_size > _MAX_CHUNK_SIZE
        ):
            raise ValueError("chunk size is outside the supported range")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if (
            isinstance(max_redirects, bool)
            or not isinstance(max_redirects, int)
            or max_redirects < 0
        ):
            raise ValueError("redirect limit must be non-negative")
        self.manifest = manifest
        self.downloader = downloader
        self.transport = transport or UrllibTransport()
        self.chunk_size = chunk_size
        self.timeout_seconds = float(timeout_seconds)
        self.max_redirects = max_redirects

    def fetch(self, model_id: str, file_path: str) -> Path:
        model, model_file = self.manifest.model_file(model_id, file_path)
        if model.acquisition != "download" or model_file.source_url is None:
            raise ManifestError("model file is not approved for download")
        target = model.cache_path / model_file.path
        if target.exists() or target.is_symlink():
            return self.manifest.resolve_runtime_file(model_id, file_path)
        return self._fetch_missing(model_id, model_file)

    def fetch_all(self) -> tuple[Path, ...]:
        resolved: dict[tuple[str, str], Path] = {}
        missing: list[tuple[str, ModelFile]] = []
        for model in self.manifest.models.values():
            if model.acquisition != "download":
                continue
            for model_file in model.files:
                identity = (model.id, model_file.path)
                target = model.cache_path / model_file.path
                if target.exists() or target.is_symlink():
                    resolved[identity] = self.manifest.resolve_runtime_file(
                        model.id, model_file.path
                    )
                else:
                    missing.append((model.id, model_file))

        for model_id, model_file in missing:
            resolved[(model_id, model_file.path)] = self._fetch_missing(
                model_id, model_file
            )

        return tuple(
            resolved[(model.id, model_file.path)]
            for model in self.manifest.models.values()
            if model.acquisition == "download"
            for model_file in model.files
        )

    def _fetch_missing(self, model_id: str, model_file: ModelFile) -> Path:
        return self.downloader.install_bytes(
            model_id,
            model_file.path,
            self._download_chunks(model_id, model_file),
        )

    def _download_chunks(self, model_id: str, model_file: ModelFile):
        source_url = model_file.source_url
        if source_url is None:
            raise ManifestError("model file is not approved for download")
        chain = [source_url]
        current_url = source_url
        redirects = 0
        headers = {
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "translator-local-model-fetch/1",
        }

        while True:
            try:
                response = self.transport.open(
                    current_url,
                    headers=headers,
                    timeout_seconds=self.timeout_seconds,
                )
            except Exception:
                raise FetchError("download transport failed") from None
            try:
                if response.status in _REDIRECT_STATUSES:
                    location = _single_header(response, "Location")
                    if location is None:
                        raise FetchError("redirect Location is missing")
                    redirects += 1
                    if redirects > self.max_redirects:
                        raise FetchError("download redirect limit exceeded")
                    next_url = urljoin(current_url, location)
                    next_chain = [*chain, next_url]
                    self.manifest.validate_download_chain(
                        model_id, model_file.path, next_chain
                    )
                    current_url = next_url
                    chain = next_chain
                    continue

                if response.status != 200:
                    raise FetchError("download response status is not allowed")
                self.manifest.validate_download_chain(model_id, model_file.path, chain)
                _validate_response_metadata(response, model_file)
                while True:
                    try:
                        block = response.read(self.chunk_size)
                    except Exception:
                        raise FetchError("download transport failed") from None
                    if not block:
                        return
                    yield block
            finally:
                _close_response(response)


def _single_header(response: HttpResponse, name: str) -> str | None:
    values = _header_values(response.headers, name)
    if not values:
        return None
    if len(values) != 1 or not values[0]:
        raise FetchError(f"{name} header is ambiguous")
    return values[0]


def _header_values(
    headers: ResponseHeaders | Mapping[str, str], name: str
) -> list[str]:
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        values = get_all(name, [])
        return list(values or [])
    return [
        value
        for header_name, value in headers.items()
        if header_name.lower() == name.lower()
    ]


def _validate_response_metadata(response: HttpResponse, model_file: ModelFile) -> None:
    content_length = _single_header(response, "Content-Length")
    if (
        content_length is None
        or not _CONTENT_LENGTH_RE.fullmatch(content_length)
        or int(content_length) != model_file.size_bytes
    ):
        raise FetchError("response Content-Length does not match manifest")
    encodings = _header_values(response.headers, "Content-Encoding")
    if len(encodings) > 1 or (encodings and encodings[0].strip().lower() != "identity"):
        raise FetchError("response Content-Encoding is not supported")


def _close_response(response: HttpResponse) -> None:
    try:
        response.close()
    except Exception:
        return
