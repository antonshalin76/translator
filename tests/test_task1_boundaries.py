from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Task1BoundaryTests(unittest.TestCase):
    def test_protobuf_declares_bidirectional_stream_and_discriminated_messages(self) -> None:
        proto_root = ROOT / "proto"
        proto_path = proto_root / "translator/provider/v1/provider.proto"
        schema = proto_path.read_text()

        with tempfile.TemporaryDirectory() as temporary_directory:
            descriptor = Path(temporary_directory) / "provider.pb"
            subprocess.run(
                [
                    "protoc",
                    f"--proto_path={proto_root}",
                    f"--descriptor_set_out={descriptor}",
                    "--include_imports",
                    str(proto_path.relative_to(proto_root)),
                ],
                cwd=proto_root,
                check=True,
                capture_output=True,
                text=True,
            )
            decoded = subprocess.run(
                [
                    "protoc",
                    "--proto_path=/usr/include",
                    "--decode=google.protobuf.FileDescriptorSet",
                    "google/protobuf/descriptor.proto",
                ],
                input=descriptor.read_bytes(),
                check=True,
                capture_output=True,
            ).stdout.decode()

        self.assertIn('name: "Stream"', decoded)
        self.assertIn("client_streaming: true", decoded)
        self.assertIn("server_streaming: true", decoded)
        probe_method = re.search(
            r'method\s*\{\s*name:\s*"Probe"(?P<body>.*?)\n\s*\}',
            decoded,
            re.DOTALL,
        )
        self.assertIsNotNone(probe_method)
        self.assertIn(
            'input_type: ".translator.provider.v1.ProviderProbeRequest"',
            probe_method.group("body"),
        )
        self.assertIn(
            'output_type: ".translator.provider.v1.ProviderProbeResponse"',
            probe_method.group("body"),
        )
        self.assertNotIn("client_streaming:", probe_method.group("body"))
        self.assertNotIn("server_streaming:", probe_method.group("body"))
        request_variants = (
            "open_session",
            "input_frame",
            "cancel_utterance",
            "close_session",
            "update_debug_text",
        )
        event_variants = (
            "session_opened",
            "audio_delta",
            "transcript_delta",
            "translation_delta",
            "utterance_final",
            "session_closed",
            "health",
            "latency",
            "error",
        )
        for variant in request_variants + event_variants:
            self.assertRegex(
                decoded,
                rf'field\s*\{{[^}}]*name: "{variant}"[^}}]*oneof_index: 0[^}}]*\}}',
            )
        update_field = re.search(
            r'field\s*\{\s*name:\s*"update_debug_text"(?P<body>.*?)\n\s*\}',
            decoded,
            re.DOTALL,
        )
        self.assertIsNotNone(update_field)
        self.assertRegex(update_field.group("body"), r"(?m)^\s*number:\s*5\s*$")
        self.assertRegex(
            update_field.group("body"), r"(?m)^\s*type:\s*TYPE_MESSAGE\s*$"
        )
        self.assertIn(
            'type_name: ".translator.provider.v1.UpdateDebugText"',
            update_field.group("body"),
        )
        self.assertRegex(
            update_field.group("body"), r"(?m)^\s*oneof_index:\s*0\s*$"
        )
        request_oneof = re.search(r"oneof\s+request\s*\{([^}]*)\}", schema, re.DOTALL)
        event_oneof = re.search(r"oneof\s+event\s*\{([^}]*)\}", schema, re.DOTALL)
        self.assertIsNotNone(request_oneof)
        self.assertIsNotNone(event_oneof)
        for variant in request_variants:
            self.assertIn(variant, request_oneof.group(1))
        for variant in event_variants:
            self.assertIn(variant, event_oneof.group(1))
        for message in (
            "LatencyPolicyState",
            "ProviderProbeRequest",
            "ProviderProbeResponse",
            "ProviderInputFrame",
            "ProviderAudioDelta",
            "ProviderLatency",
            "ProviderHealth",
            "UpdateDebugText",
        ):
            self.assertIn(f'message {message} ', schema)
        def message_body(name: str) -> str:
            match = re.search(
                rf"message\s+{name}\s*\{{(?P<body>.*?)\n\}}",
                schema,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            return match.group("body")

        self.assertRegex(
            message_body("ProviderSessionOpened"),
            r"\buint64\s+event_sequence\s*=\s*7\s*;",
        )
        update_body = message_body("UpdateDebugText")
        self.assertRegex(update_body, r"\bstring\s+schema_version\s*=\s*1\s*;")
        self.assertRegex(update_body, r"\bstring\s+session_id\s*=\s*2\s*;")
        self.assertRegex(update_body, r"\bbool\s+enabled\s*=\s*3\s*;")
        self.assertRegex(
            message_body("ProviderProbeRequest"),
            r"\bstring\s+schema_version\s*=\s*1\s*;",
        )
        probe_response_body = message_body("ProviderProbeResponse")
        self.assertRegex(
            probe_response_body, r"\bstring\s+schema_version\s*=\s*1\s*;"
        )
        self.assertRegex(
            probe_response_body, r"\bstring\s+generation_id\s*=\s*2\s*;"
        )
        for field in (
            "schema_version",
            "session_id",
            "direction_id",
            "event_sequence",
        ):
            self.assertIn(field, schema)

    def test_wire_literals_are_present_in_all_language_contracts(self) -> None:
        rust = "\n".join(
            path.read_text()
            for path in sorted((ROOT / "crates/translator-core/src").glob("*.rs"))
        )
        python = (
            ROOT / "sidecar/translator_sidecar/provider_contract.py"
        ).read_text()
        protobuf = (ROOT / "proto/translator/provider/v1/provider.proto").read_text()

        for literal in ("microphone", "speaker", "quality_first", "streaming_first"):
            self.assertIn(literal, rust)
            self.assertIn(literal, python)
        for enum_value in (
            "AUDIO_DIRECTION_MICROPHONE",
            "AUDIO_DIRECTION_SPEAKER",
            "TRANSLATION_MODE_QUALITY_FIRST",
            "TRANSLATION_MODE_STREAMING_FIRST",
        ):
            self.assertIn(enum_value, protobuf)

        message_versions = {
            "OpenProviderSession": "translator.provider.open_session.v1",
            "ProviderSessionOpened": "translator.provider.session_opened.v1",
            "ProviderProbeRequest": "translator.provider.probe_request.v1",
            "ProviderProbeResponse": "translator.provider.probe_response.v1",
            "CloseProviderSession": "translator.provider.close_session.v1",
            "CancelUtterance": "translator.provider.cancel_utterance.v1",
            "ProviderInputFrame": "translator.provider.input.v1",
            "ProviderAudioDelta": "translator.provider.audio_delta.v1",
            "ProviderTranscriptDelta": "translator.provider.transcript_delta.v1",
            "ProviderTranslationDelta": "translator.provider.translation_delta.v1",
            "ProviderUtteranceFinal": "translator.provider.utterance_final.v1",
            "ProviderSessionClosed": "translator.provider.session_closed.v1",
            "ProviderHealth": "translator.provider.health.v1",
            "ProviderLatency": "translator.provider.latency.v1",
            "PrivacySafeProviderError": "translator.provider.error.v1",
            "UpdateDebugText": "translator.provider.update_debug_text.v1",
        }
        for message, version in message_versions.items():
            self.assertEqual(rust.count(version), 1)
            self.assertEqual(python.count(version), 2)
            class_body = re.search(
                rf"class\s+{message}\(ContractModel\):(?P<body>.*?)(?=\nclass\s|\ndef\s|\Z)",
                python,
                re.DOTALL,
            )
            self.assertIsNotNone(class_body)
            self.assertIn(version, class_body.group("body"))

    def test_close_request_reasons_are_narrower_than_session_close_reasons(self) -> None:
        schema = (ROOT / "proto/translator/provider/v1/provider.proto").read_text()

        def enum_values(name: str) -> set[str]:
            match = re.search(rf"enum\s+{name}\s*\{{([^}}]*)\}}", schema, re.DOTALL)
            self.assertIsNotNone(match)
            return set(re.findall(r"\b([A-Z][A-Z0-9_]+)\s*=", match.group(1)))

        request_values = enum_values("CloseRequestReason")
        event_values = enum_values("SessionCloseReason")
        self.assertEqual(
            request_values,
            {
                "CLOSE_REQUEST_REASON_UNSPECIFIED",
                "CLOSE_REQUEST_REASON_USER_STOP",
                "CLOSE_REQUEST_REASON_ROUTE_REMOVED",
                "CLOSE_REQUEST_REASON_DEVICE_UNAVAILABLE",
                "CLOSE_REQUEST_REASON_PROVIDER_SWITCH",
                "CLOSE_REQUEST_REASON_DAEMON_SHUTDOWN",
            },
        )
        self.assertIn("SESSION_CLOSE_REASON_PROVIDER_FAILURE", event_values)
        self.assertIn("SESSION_CLOSE_REASON_CLOSE_TIMEOUT", event_values)

    def test_frontend_sources_do_not_define_or_transport_raw_pcm(self) -> None:
        source_root = ROOT / "apps/translator-ui/src"
        forbidden = re.compile(
            r"\b(?:Uint8Array|ArrayBuffer|ProviderInputFrame|ProviderAudioDelta"
            r"|raw_pcm|pcm_bytes)\b"
        )
        violations: list[str] = []
        checked_files = 0

        self.assertTrue(source_root.is_dir(), "frontend source root must exist")

        for path in source_root.rglob("*"):
            if path.suffix not in {".ts", ".tsx", ".js", ".jsx"}:
                continue
            checked_files += 1
            match = forbidden.search(path.read_text())
            if match:
                violations.append(f"{path.relative_to(ROOT)}:{match.group(0)}")

        self.assertGreater(checked_files, 0, "at least one frontend source must be checked")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
