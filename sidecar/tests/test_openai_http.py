from __future__ import annotations

import asyncio
import io
import json
import socket
import wave
from types import SimpleNamespace

import pytest

from translator_sidecar import openai_http as http
from translator_sidecar.provider_contract import SafeErrorCode


class Content:
    def __init__(self, body):
        self.body = body

    async def iter_chunked(self, size):
        for offset in range(0, len(self.body), size):
            yield self.body[offset : offset + size]


class Response:
    def __init__(self, body=b"", status=200, *, content_type="audio/pcm"):
        self.status = status
        self.headers = {} if content_type is None else {"Content-Type": content_type}
        self.content_type = content_type or "application/octet-stream"
        self.content = Content(
            body if isinstance(body, bytes) else json.dumps(body).encode()
        )
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(http.urllib.request, "getproxies", lambda: {})
    monkeypatch.setattr(socket, "if_nametoindex", lambda name: 20)
    monkeypatch.setattr(
        http.aiohttp, "TCPConnector", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    class Client:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.replies = []
            self.requests = []
            self.closed = False

        def request(self, method, url, **kwargs):
            self.requests.append((method, url, kwargs))
            return self.replies.pop(0)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(http.aiohttp, "ClientSession", Client)
    return http.OpenAIHttpPipeline("synthetic-not-a-key")


def test_exact_three_stage_requests_and_bounded_raw_speech(client):
    async def scenario():
        replies = [
            Response({"id": http.ASR_MODEL}),
            Response({"text": "Source fact."}),
            Response(
                {
                    "model": "gpt-4.1-2025-04-14",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "Translated fact."},
                        }
                    ],
                }
            ),
            Response(b"\1\2" * 1000),
        ]
        client._client.replies = replies.copy()
        await client.warmup()
        text = await client.transcribe(b"\1\2" * 480, 24000, "en")
        translated = await client.translate(text, "ru")
        chunks = [chunk async for chunk in client.speech(translated, "ru")]
        assert b"".join(chunks) == b"\1\2" * 1000 and max(map(len, chunks)) <= 960
        requests = client._client.requests
        assert [method for method, _, _ in requests] == ["GET", "POST", "POST", "POST"]
        assert all(
            url.startswith(http.API_ORIGIN + "/") and not options["allow_redirects"]
            for _, url, options in requests
        )
        form = requests[1][2]["data"]
        fields = {field[0]["name"]: field[2] for field in form._fields}
        assert fields["model"] == "gpt-transcribe" and fields["languages[]"] == "en"
        assert fields["prompt"] == http._ASR_PROMPTS["en"]
        with wave.open(io.BytesIO(fields["file"]), "rb") as audio:
            assert audio.getparams()[:3] == (1, 2, 24000)
            assert audio.readframes(480) == b"\1\2" * 480
        mt = requests[2][2]["json"]
        assert mt["store"] is False and mt["messages"][1]["content"] == text
        assert "numbers and times in words" in mt["messages"][0]["content"]
        assert "leading zeros and separators" in mt["messages"][0]["content"]
        speech = requests[3][2]["json"]
        assert requests[3][2]["timeout"].total == 36
        assert requests[3][2]["timeout"].sock_read == 6
        assert speech["voice"] == "marin" and speech["input"] == translated
        assert (
            "Russian" in speech["instructions"] and "number" in speech["instructions"]
        )
        assert all(reply.closed for reply in replies)
        await client.close()
        assert client._client.closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("status", "body", "retryable", "resource", "code"),
    [
        (401, {}, False, False, SafeErrorCode.PROVIDER_AUTH_FAILED),
        (403, {}, False, False, SafeErrorCode.PROVIDER_AUTH_FAILED),
        (429, {}, False, True, SafeErrorCode.PROVIDER_UNAVAILABLE),
        (
            429,
            {"error": {"code": "unknown"}},
            False,
            True,
            SafeErrorCode.PROVIDER_UNAVAILABLE,
        ),
        (429, {"error": {"code": []}}, False, True, SafeErrorCode.PROVIDER_UNAVAILABLE),
        (429, {"error": None}, False, True, SafeErrorCode.PROVIDER_UNAVAILABLE),
        (
            429,
            {"error": {"type": "rate_limit_error"}},
            True,
            True,
            SafeErrorCode.PROVIDER_UNAVAILABLE,
        ),
        (
            429,
            {"error": {"code": "insufficient_quota", "type": "rate_limit_error"}},
            False,
            True,
            SafeErrorCode.PROVIDER_UNAVAILABLE,
        ),
        (
            429,
            {"error": {"code": "insufficient_quota", "message": "DO_NOT_EXPOSE"}},
            False,
            True,
            SafeErrorCode.PROVIDER_UNAVAILABLE,
        ),
        (
            429,
            {"error": {"code": "rate_limit_exceeded"}},
            True,
            True,
            SafeErrorCode.PROVIDER_UNAVAILABLE,
        ),
        (429, b"malformed", False, True, SafeErrorCode.PROVIDER_UNAVAILABLE),
        (503, {}, True, False, SafeErrorCode.PROVIDER_UNAVAILABLE),
        (307, {}, False, False, SafeErrorCode.PROVIDER_UNAVAILABLE),
    ],
)
def test_failures_do_not_retry_redirect_or_expose_response(
    client, status, body, retryable, resource, code
):
    async def scenario():
        reply = Response(body, status)
        client._client.replies = [reply]
        with pytest.raises(http.OpenAIHttpFailure) as caught:
            await client.translate("actual text", "ru")
        assert caught.value.code == code and caught.value.retryable is retryable
        assert caught.value.resource_exhausted is resource
        assert str(caught.value) == code.value and len(client._client.requests) == 1
        assert reply.closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [b"", b"\0" * 960, b"\1" * 961, b"\1" * (http.MAX_PCM + 2)],
    ids=["empty", "silent", "odd", "over-capacity"],
)
def test_invalid_or_excess_speech_fails_and_closes(client, body):
    async def scenario():
        reply = Response(body)
        client._client.replies = [reply]
        with pytest.raises(http.OpenAIHttpFailure):
            async for _ in client.speech("unchanged MT", "en"):
                pass
        assert reply.closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "reply",
    [
        {},
        [],
        {"text": ""},
        {"text": "x" * 20001},
        {"text": " " * 20001},
        {"text": "ok", "model": "other"},
        b"malformed",
        b" " * (http._MAX_JSON + 1),
    ],
    ids=[
        "missing",
        "list",
        "empty",
        "long",
        "long-blank",
        "model",
        "invalid-json",
        "large-json",
    ],
)
def test_invalid_asr_never_reaches_mt(client, reply):
    async def scenario():
        client._client.replies = [Response(reply)]
        with pytest.raises(http.OpenAIHttpFailure) as raised:
            await client.transcribe(b"\1\2" * 480, 24000, "ru")
        if reply != {"text": ""}:
            assert raised.value.code is SafeErrorCode.PROVIDER_UNAVAILABLE
        assert len(client._client.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_valid_empty_asr_is_no_speech_not_provider_failure(client, text):
    async def scenario():
        response = Response({"text": text})
        client._client.replies = [response]
        with pytest.raises(http.OpenAIHttpFailure) as raised:
            await client.transcribe(b"\1\2" * 480, 24000, "ru")
        assert raised.value.code is SafeErrorCode.NO_SPEECH
        assert not raised.value.retryable and not raised.value.transport
        assert response.closed and len(client._client.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("interface", ["", "eth0", "lo"])
def test_transport_refuses_non_vpn_interface_before_client_creation(client, interface):
    with pytest.raises(http.OpenAIHttpFailure):
        http.OpenAIHttpPipeline("synthetic-not-a-key", vpn_interface=interface)


def test_cancel_speech_closes_active_response_before_return(client):
    async def scenario():
        waiting = asyncio.Event()

        class SlowContent:
            async def iter_chunked(self, size):
                yield b"\1\2" * 480
                waiting.set()
                await asyncio.Future()

        reply = Response()
        reply.content = SlowContent()
        client._client.replies = [reply]

        async def read():
            async for _ in client.speech("text", "en"):
                pass

        task = asyncio.create_task(read())
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reply.closed

    asyncio.run(scenario())


def test_vpn_binding_is_set_on_new_socket_before_return(client, monkeypatch):
    calls = []
    fake = SimpleNamespace(
        setsockopt=lambda *args: calls.append(args),
        close=lambda: calls.append("closed"),
    )
    monkeypatch.setattr(socket, "socket", lambda *args: fake)
    assert (
        client._socket((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443)))
        is fake
    )
    assert calls == [(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, b"amn0\0")]
    monkeypatch.setattr(socket, "if_nametoindex", lambda name: 21)
    with pytest.raises(http.OpenAIHttpFailure):
        client._socket((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443)))


def test_bind_failure_closes_socket_without_fallback(client, monkeypatch):
    calls = []

    def bind(*args):
        raise PermissionError("private host error")

    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args: SimpleNamespace(
            setsockopt=bind, close=lambda: calls.append("closed")
        ),
    )
    with pytest.raises(PermissionError):
        client._socket((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443)))
    assert calls == ["closed"]


@pytest.mark.parametrize("link", [20, 21])
def test_dns_must_return_explicit_vpn_link(client, monkeypatch, link):
    async def scenario():
        process = SimpleNamespace(returncode=0)

        async def communicate():
            return json.dumps(
                {
                    "type": "a(iiay)st",
                    "data": [[[link, 2, [1, 2, 3, 4]]], "api.openai.com", 1],
                }
            ).encode(), None

        process.communicate = communicate

        async def spawn(*args, **kwargs):
            assert args[-4:] == ("20", "api.openai.com", "2", "1")
            assert args[0] == "/usr/bin/busctl"
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        if link == 21:
            with pytest.raises(http.OpenAIHttpFailure) as raised:
                await client._resolver.resolve("api.openai.com", 443, socket.AF_INET)
            assert not raised.value.retryable and not raised.value.transport
        else:
            records = await client._resolver.resolve(
                "api.openai.com", 443, socket.AF_INET
            )
            assert records[0]["host"] == "1.2.3.4" and records[0]["port"] == 443

    asyncio.run(scenario())


def test_proxy_configuration_never_silently_bypassed(monkeypatch):
    monkeypatch.setattr(
        http.urllib.request, "getproxies", lambda: {"https": "private-proxy"}
    )
    with pytest.raises(http.OpenAIHttpFailure):
        http.OpenAIHttpPipeline("test-key")


def test_cancel_dns_kills_and_joins_owned_process(client, monkeypatch):
    async def scenario():
        waiting = asyncio.Event()
        killed = asyncio.Event()
        process = SimpleNamespace(returncode=None, joins=0)

        def kill():
            process.returncode = -9
            killed.set()

        async def communicate():
            waiting.set()
            await killed.wait()
            process.joins += 1
            return b"", None

        process.kill = kill
        process.communicate = communicate

        async def spawn(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(
            client._resolver.resolve("api.openai.com", 443, socket.AF_INET)
        )
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.returncode == -9 and process.joins == 1

    asyncio.run(scenario())


def test_missing_native_resolver_fails_without_dns_fallback(client, monkeypatch):
    async def scenario():
        async def spawn(*args, **kwargs):
            raise FileNotFoundError("native DNS unavailable")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        with pytest.raises(http.OpenAIHttpFailure):
            await client._resolver.resolve("api.openai.com", 443, socket.AF_INET)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "finish_reason",
        "refusal",
        "content",
        "choices",
        "null-message",
        "list-message",
        "string-message",
    ],
)
def test_mt_rejects_mismatched_model_truncation_refusal_or_empty(client, mutation):
    async def scenario():
        reply = {
            "model": "gpt-4.1",
            "choices": [{"finish_reason": "stop", "message": {"content": "fact"}}],
        }
        if mutation == "model":
            reply["model"] = "unexpected"
        elif mutation == "choices":
            reply["choices"] = []
        elif mutation == "finish_reason":
            reply["choices"][0]["finish_reason"] = "length"
        elif mutation.endswith("-message"):
            reply["choices"][0]["message"] = {
                "null-message": None,
                "list-message": [],
                "string-message": "invalid",
            }[mutation]
        else:
            reply["choices"][0]["message"][mutation] = (
                "refused" if mutation == "refusal" else ""
            )
        response = Response(reply)
        client._client.replies = [response]
        with pytest.raises(http.OpenAIHttpFailure) as raised:
            await client.translate("unchanged source", "ru")
        assert not raised.value.retryable and not raised.value.transport
        assert response.closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "content_type", [None, "application/json", "text/plain", "audio/wav", "audio/mpeg"]
)
def test_speech_rejects_incompatible_media_before_any_audio(client, content_type):
    async def scenario():
        response = Response(b'{"error":"oops"}', content_type=content_type)
        client._client.replies = [response]
        received = []
        with pytest.raises(http.OpenAIHttpFailure) as raised:
            async for chunk in client.speech("unchanged text", "en"):
                received.append(chunk)
        assert not received and response.closed
        assert not raised.value.retryable and not raised.value.transport

    asyncio.run(scenario())


@pytest.mark.parametrize("content_type", ["audio/pcm", "application/octet-stream"])
def test_speech_accepts_declared_raw_media(client, content_type):
    async def scenario():
        response = Response(b"\1\2" * 480, content_type=content_type)
        client._client.replies = [response]
        assert (
            b"".join([chunk async for chunk in client.speech("fact", "en")])
            == b"\1\2" * 480
        )
        assert response.closed

    asyncio.run(scenario())
