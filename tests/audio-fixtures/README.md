# Audio Fixtures

This directory owns deterministic synthetic audio used by routing, latency and
privacy tests in later tasks. Fixtures must contain generated or explicitly
licensed speech, fixed sample metadata and no private conversation recordings.

Each fixture set records sample rate, channels, sample format, frame duration,
language direction, expected event order and the privacy marker used by scans.
Raw captures and ad hoc debug recordings stay outside the repository.
