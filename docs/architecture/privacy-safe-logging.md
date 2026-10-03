# Privacy-Safe Logging

Normal logs contain operational metadata only:

- fixed event names and machine-readable error codes;
- direction, provider state, queue duration and latency counters;
- opaque session, stream and utterance identifiers when correlation is needed.

Normal logs never contain PCM, transcripts, translations, model prompts,
credentials, sidecar tokens or arbitrary exception text. Provider errors map a
closed error code to a fixed safe message. Log projection drops that message and
retains only the code and retryability.

The daemon and Task 7 runtime bridge install a permanent panic hook before
creating their Tokio runtimes. It emits only the fixed `runtime_panic` event
and `internal_error` code, without payload, source location, thread name, or
backtrace. It does not chain or temporarily replace another hook. Catching an
unwind alone is not a logging privacy boundary: Rust invokes the hook first.
This protection does not cover pre-main failures, process aborts, or output
from child processes.

`debug_text` requires explicit enablement and remains in a bounded in-memory
buffer. It is never written to logs, local storage, telemetry, debug-capture
metadata or error messages.

Debug audio capture is a separate explicit mode. Its bounded files are written
only to the private user-state debug directory and are excluded from version
control.
