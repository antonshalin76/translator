# Privacy-Safe Logging

Normal logs contain operational metadata only:

- fixed event names and machine-readable error codes;
- direction, provider state, queue duration and latency counters;
- opaque session, stream and utterance identifiers when correlation is needed.

Normal logs never contain PCM, transcripts, translations, model prompts,
credentials, sidecar tokens or arbitrary exception text. Provider errors map a
closed error code to a fixed safe message. Log projection drops that message and
retains only the code and retryability.

`debug_text` requires explicit enablement and remains in a bounded in-memory
buffer. It is never written to logs, local storage, telemetry, debug-capture
metadata or error messages.

Debug audio capture is a separate explicit mode. Its bounded files are written
only to the private user-state debug directory and are excluded from version
control.
