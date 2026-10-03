//! Synthetic PCM conservation at the real native service seam, never physical audio.

use super::*;
use std::{
    io::{Read, Write},
    os::unix::fs::FileTypeExt,
    process::{Child, Command, Stdio},
};

const PACKET_BYTES: usize = 1_024;
const HEADER: [i16; 4] = [-30_000, 30_000, -28_000, 28_000];
const WINDOW: Duration = Duration::from_secs(180);
const PCM_LIMIT: usize = 48_000 * FRAME_BYTES * 190;

fn packet(sequence: u32) -> Vec<u8> {
    let mut samples = Vec::with_capacity(PACKET_BYTES / 2);
    samples.extend(HEADER);
    samples.extend([sequence as i16, (sequence >> 16) as i16]);
    let mut state = sequence.wrapping_add(1);
    while samples.len() < PACKET_BYTES / 2 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        samples.push((state % 24_001) as i16 - 12_000);
    }
    samples.into_iter().flat_map(i16::to_le_bytes).collect()
}

fn packets(pcm: &[u8]) -> Vec<(u32, usize)> {
    let header: Vec<u8> = HEADER.into_iter().flat_map(i16::to_le_bytes).collect();
    let mut result = Vec::new();
    let mut offset = 0;
    while offset + PACKET_BYTES <= pcm.len() {
        if pcm[offset..offset + header.len()] != header {
            offset += FRAME_BYTES;
            continue;
        }
        let sequence = u32::from(u16::from_le_bytes([pcm[offset + 8], pcm[offset + 9]]))
            | (u32::from(u16::from_le_bytes([pcm[offset + 10], pcm[offset + 11]])) << 16);
        assert!(
            pcm[offset..offset + PACKET_BYTES] == packet(sequence),
            "corrupt packet {sequence}"
        );
        result.push((sequence, offset));
        offset += PACKET_BYTES;
    }
    for pair in result.windows(2) {
        assert_eq!(
            pair[1].0,
            pair[0].0 + 1,
            "producer packet loss/duplicate/reorder"
        );
        assert_eq!(
            pair[1].1,
            pair[0].1 + PACKET_BYTES,
            "inserted/lost samples between packets"
        );
    }
    if let (Some(first), Some(last)) = (result.first(), result.last()) {
        assert!(
            first.1 < PACKET_BYTES,
            "cannot exclude a complete first packet"
        );
        assert!(
            pcm.len() - last.1 - PACKET_BYTES < PACKET_BYTES,
            "cannot exclude a complete final packet"
        );
    } else {
        assert!(
            pcm.len() < PACKET_BYTES,
            "complete packets must be identifiable"
        );
    }
    result
}

#[derive(Default)]
struct CollectedPcm {
    bytes: Vec<u8>,
    receipts: Vec<(usize, Instant)>,
}

struct PcmChild {
    child: Child,
    done: Arc<AtomicBool>,
    worker: Option<JoinHandle<CollectedPcm>>,
}

impl PcmChild {
    fn source() -> Self {
        let mut child = Command::new("pacat")
            .args([
                "--playback",
                "--raw",
                "--rate=48000",
                "--channels=1",
                "--format=s16le",
                "--device=translator_test_mic",
                "--latency-msec=20",
                "--process-time-msec=5",
                "--property=media.name=synthetic-numbered-input",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        let done = Arc::new(AtomicBool::new(false));
        let source_done = done.clone();
        let worker = thread::spawn(move || {
            let mut sequence = 0;
            while !source_done.load(Ordering::Acquire) {
                if input.write_all(&packet(sequence)).is_err() {
                    break;
                }
                sequence += 1;
            }
            CollectedPcm::default()
        });
        Self {
            child,
            done,
            worker: Some(worker),
        }
    }

    fn observer() -> Self {
        let mut child = Command::new("parec")
            .args([
                "--raw",
                "--rate=48000",
                "--channels=1",
                "--format=s16le",
                "--device=translator_mic_out.monitor",
                "--latency-msec=20",
                "--process-time-msec=5",
                "--property=media.name=synthetic-numbered-observer",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut output = child.stdout.take().unwrap();
        let worker = thread::spawn(move || {
            let mut pcm = CollectedPcm::default();
            let mut buffer = [0; 4_096];
            loop {
                let length = output.read(&mut buffer).unwrap();
                if length == 0 {
                    break;
                }
                assert!(
                    pcm.bytes.len() + length <= PCM_LIMIT,
                    "observer collection bounded"
                );
                pcm.bytes.extend_from_slice(&buffer[..length]);
                pcm.receipts.push((pcm.bytes.len(), Instant::now()));
            }
            pcm
        });
        Self {
            child,
            done: Arc::new(AtomicBool::new(false)),
            worker: Some(worker),
        }
    }

    fn stop(mut self) -> CollectedPcm {
        self.done.store(true, Ordering::Release);
        let _ = self.child.kill();
        self.child.wait().unwrap();
        self.worker.take().unwrap().join().unwrap()
    }
}

impl Drop for PcmChild {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn inventory(kind: &str) -> Vec<serde_json::Value> {
    let result = Command::new("pactl")
        .args(["--format=json", "list", kind])
        .output()
        .unwrap();
    assert!(result.status.success());
    serde_json::from_slice(&result.stdout).unwrap()
}

fn fixture() -> (
    OriginalMicrophoneRegistry,
    NativeTransport,
    OriginalMicrophoneRegistration,
    PcmChild,
) {
    let server = std::env::var("PULSE_SERVER").expect("private PULSE_SERVER required");
    assert!(server.starts_with("unix:/tmp/translator-loopback-") && server.ends_with("/native"));
    assert!(
        std::fs::symlink_metadata(server.strip_prefix("unix:").unwrap())
            .unwrap()
            .file_type()
            .is_socket()
    );
    assert!(inventory("cards").is_empty(), "no physical fixture devices");
    let microphone = "translator_test_mic.monitor";
    let endpoint = |kind, name| {
        inventory(kind)
            .into_iter()
            .find(|v| v["name"] == name)
            .unwrap()["index"]
            .as_u64()
            .unwrap() as u32
    };
    let source = PcmChild::source();
    let registry = OriginalMicrophoneRegistry::default();
    let session = Uuid::new_v4();
    registry.claim(session).unwrap();
    let lease = Arc::new(Lease::default());
    let mut native = NativeTransport::new(session).unwrap();
    let identity = native
        .connect(
            Route {
                source: microphone.into(),
                source_index: endpoint("sources", microphone),
                sink_index: endpoint("sinks", MIC_OUT_SINK),
            },
            session,
            lease,
            Instant::now() + STARTUP_LIMIT,
        )
        .unwrap();
    registry.publish(identity.clone()).unwrap();
    (registry, native, identity, source)
}

fn service(
    native: &mut NativeTransport,
    identity: &OriginalMicrophoneRegistration,
    duration: Duration,
) {
    let end = Instant::now() + duration;
    while Instant::now() < end {
        native.iterate_streams(identity, true).unwrap();
        thread::sleep(POLL_INTERVAL);
    }
}

fn cork_playback(native: &mut NativeTransport) {
    let corked = Rc::new(Cell::new(None));
    let callback = corked.clone();
    let _cork = native
        .playback
        .as_mut()
        .unwrap()
        .cork(Some(Box::new(move |success| callback.set(Some(success)))));
    let deadline = Instant::now() + STARTUP_LIMIT;
    while corked.get().is_none() {
        assert!(Instant::now() < deadline);
        native.iterate().unwrap();
        thread::sleep(POLL_INTERVAL);
    }
    assert_eq!(corked.get(), Some(true));
    assert_eq!(native.playback.as_ref().unwrap().is_corked(), Ok(true));
}

// N2 mechanical exhaustion, not the full-service N5 corked-consumer lifecycle.
fn exhaust_capture(native: &mut NativeTransport, identity: &OriginalMicrophoneRegistration) {
    cork_playback(native);
    let deadline = Instant::now() + Duration::from_secs(1);
    let error = loop {
        native.iterate().unwrap();
        if let Err(error) = native.pump(true, &identity.lease) {
            break error;
        }
        assert!(
            Instant::now() < deadline,
            "bounded pump must fail: pending={} readable={:?} credit={:?} captured={} submitted={} corked={:?}",
            native.pending.len(),
            native.capture.as_ref().unwrap().readable_size(),
            native.playback.as_ref().unwrap().writable_size(),
            native.captured_bytes,
            native.forwarded_bytes,
            native.playback.as_ref().unwrap().is_corked()
        );
        thread::sleep(POLL_INTERVAL);
    };
    assert!(!identity.is_live());
    let failure = *identity.lease.failure.lock().unwrap();
    assert!(
        matches!(
            failure,
            Some((
                OriginalMicrophoneError::Buffer,
                "capture_readable_bound"
                    | "capture_turn_bound"
                    | "stream_overflow"
                    | "capture_backpressure_exhausted"
            ))
        ),
        "real buffer exhaustion required: {failure:?}"
    );
    eprintln!(
        "PRIVATE_NATIVE_PUMP_EXHAUSTION error={error:?} automatic_buffer_revocation=true PASS"
    );
}

#[test]
fn numbered_pcm_oracle_rejects_corruption_loss_duplicate_and_reordering() {
    let good = [packet(1), packet(2), packet(3)].concat();
    assert_eq!(
        packets(&good).iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    let mut corrupt = good.clone();
    corrupt[64] ^= 1;
    let mut corrupt_first_header = good.clone();
    corrupt_first_header[0] ^= 1;
    let mut corrupt_last_header = good.clone();
    corrupt_last_header[PACKET_BYTES * 2] ^= 1;
    for invalid in [
        corrupt,
        corrupt_first_header,
        corrupt_last_header,
        [packet(1), packet(3)].concat(),
        [packet(1), packet(1)].concat(),
        [packet(2), packet(1)].concat(),
        [packet(1), vec![0; 2], packet(2)].concat(),
    ] {
        assert!(std::panic::catch_unwind(|| packets(&invalid)).is_err());
    }
}

#[test]
#[ignore = "requires a disposable private audio server; runs 180 seconds of synthetic PCM"]
fn private_pulse_native_numbered_pcm_conservation_and_stall() {
    let (registry, mut native, identity, source) = fixture();
    let lease = identity.lease.clone();
    let session = identity.session_id();

    let capture_deadline = Instant::now() + Duration::from_millis(100);
    while native.capture.as_ref().unwrap().readable_size() == Some(0) {
        assert!(Instant::now() < capture_deadline);
        native.iterate().unwrap();
        thread::sleep(POLL_INTERVAL);
    }
    let fragment = match native.capture.as_mut().unwrap().peek().unwrap() {
        PeekResult::Data(data) => data.to_vec(),
        _ => panic!("synthetic whole fragment required"),
    };
    let playback = native.playback.as_mut().unwrap();
    let credit = playback.writable_size().unwrap();
    if credit > 0 {
        playback
            .write_copy(&vec![0; credit], 0, SeekMode::Relative)
            .unwrap();
    }
    native.pending.resize(PENDING_BYTES, 0);
    native.accepted_pcm = Some(Vec::new());
    let captured_before_deferral = native.captured_bytes;
    let readable_before_deferral = native.capture.as_ref().unwrap().readable_size();
    for _ in 0..3 {
        assert_eq!(native.pump(true, &lease), Ok(0));
        assert_eq!(native.captured_bytes, captured_before_deferral);
        assert_eq!(
            native.capture.as_ref().unwrap().readable_size(),
            readable_before_deferral
        );
        assert!(native.accepted_pcm.as_ref().unwrap().is_empty());
        assert_eq!(native.pending.len(), PENDING_BYTES);
    }
    native.pending.clear(); // Fixture-inserted silence, not accepted microphone PCM.
    let accepted_after_deferral = native.pump(true, &lease).unwrap();
    let resumed = native.accepted_pcm.take().unwrap();
    assert_eq!(&resumed[..fragment.len()], fragment);
    assert_eq!(resumed.len(), accepted_after_deferral);
    assert_eq!(
        native.captured_bytes - captured_before_deferral,
        resumed.len() as u64
    );
    assert_eq!(native.capture.as_ref().unwrap().readable_size(), Some(0));
    eprintln!(
        "PRIVATE_NATIVE_DEFER repeated_peeks=3 accepted_once={} discarded_once=true PASS",
        resumed.len()
    );

    let absent_before = (native.captured_bytes, native.forwarded_bytes);
    service(&mut native, &identity, Duration::from_millis(250));
    assert!(
        native.captured_bytes > absent_before.0 && native.forwarded_bytes > absent_before.1,
        "progress without a reader"
    );
    let late = PcmChild::observer();
    let late_before = (native.captured_bytes, native.forwarded_bytes);
    service(&mut native, &identity, Duration::from_millis(250));
    let zero = late.stop();
    assert!(
        !zero.bytes.is_empty() && zero.bytes.iter().all(|byte| *byte == 0),
        "late reader sees first-frame zero"
    );
    assert!(native.captured_bytes > late_before.0 && native.forwarded_bytes > late_before.1);
    let removed_before = (native.captured_bytes, native.forwarded_bytes);
    service(&mut native, &identity, Duration::from_millis(250));
    assert!(
        identity.is_live()
            && native.captured_bytes > removed_before.0
            && native.forwarded_bytes > removed_before.1,
        "healthy driver without downstream reader"
    );
    let observer = PcmChild::observer();
    service(&mut native, &identity, Duration::from_millis(250));
    let gain = Command::new("pactl")
        .args([
            "set-sink-input-volume",
            &identity.playback_index().to_string(),
            "100%",
        ])
        .status()
        .unwrap();
    assert!(gain.success());
    native
        .inspect(&identity, false, Instant::now() + STARTUP_LIMIT)
        .unwrap();
    native.accepted_pcm = Some(Vec::new());
    let captured_before = native.captured_bytes;
    let started = Instant::now();
    let mut maximum_pending = 0;
    while started.elapsed() < WINDOW {
        native.iterate_streams(&identity, true).unwrap();
        maximum_pending = maximum_pending.max(native.pending.len());
        assert!(native.accepted_pcm.as_ref().unwrap().len() <= PCM_LIMIT);
        thread::sleep(POLL_INTERVAL);
    }
    native
        .inspect(&identity, false, Instant::now() + STARTUP_LIMIT)
        .unwrap();
    let cutoff = Instant::now();
    let accepted = native.accepted_pcm.take().unwrap();
    assert_eq!(
        accepted.len() as u64,
        native.captured_bytes - captured_before,
        "only successfully discarded fragments are counted once"
    );
    let captured_end = native.captured_bytes;
    let submitted_end = native.forwarded_bytes;
    let pending_end = native.pending.len();
    let readable_end = native.capture.as_ref().unwrap().readable_size();
    service(&mut native, &identity, Duration::from_millis(250));
    let observed = observer.stop();
    let accepted_packets = packets(&accepted);
    assert!(
        accepted_packets.len() > 15_000,
        "full 180-second nonzero accepted range"
    );
    let &(first, first_offset) = accepted_packets.first().unwrap();
    let &(last, last_offset) = accepted_packets.last().unwrap();
    let first_packet = packet(first);
    let observed_first = observed
        .bytes
        .windows(PACKET_BYTES)
        .step_by(FRAME_BYTES)
        .position(|data| data == first_packet)
        .expect("first accepted packet consumed")
        * FRAME_BYTES;
    let observed_packets = packets(&observed.bytes[observed_first..]);
    let observed_last = observed_packets
        .iter()
        .find(|(id, _)| *id == last)
        .expect("last accepted packet consumed within250ms")
        .1
        + observed_first;
    assert!(
        accepted[first_offset..last_offset + PACKET_BYTES]
            == observed.bytes[observed_first..observed_last + PACKET_BYTES],
        "exact PCM conservation, first={first} last={last}"
    );
    let consumed_at = observed
        .receipts
        .iter()
        .find(|(end, _)| *end >= observed_last + PACKET_BYTES)
        .unwrap()
        .1;
    let drain = consumed_at.saturating_duration_since(cutoff);
    assert!(
        drain <= Duration::from_millis(250),
        "independent final packet receipt exceeds drain bound"
    );
    assert!(identity.is_live() && maximum_pending <= PENDING_BYTES);
    eprintln!(
        "PRIVATE_NATIVE_CONSERVATION duration_ms={} first={} last={} packets={} accepted_bytes={} submitted_bytes={} pending_end={} readable_end={:?} peak_pending={} drain_us={} consumption=independent-numbered-pcm PASS",
        cutoff.duration_since(started).as_millis(),
        first,
        last,
        accepted_packets.len(),
        captured_end,
        submitted_end,
        pending_end,
        readable_end,
        maximum_pending,
        drain.as_micros()
    );

    exhaust_capture(&mut native, &identity);
    registry.revoke(session);
    native.disconnect(Instant::now() + CLEANUP_LIMIT).unwrap();
    registry.release(session).unwrap();
    source.stop();
    for kind in ["sink-inputs", "source-outputs"] {
        assert!(
            inventory(kind)
                .iter()
                .all(|v| v["properties"][SESSION_PROPERTY] != session.to_string())
        );
    }
    eprintln!("PRIVATE_NATIVE_CLEANUP joined=true owned_streams_absent=true PASS");
}

#[test]
#[ignore = "requires a disposable private audio server and virtual fixture sinks"]
fn private_pulse_native_pump_exhaustion_revokes() {
    let (registry, mut native, identity, source) = fixture();
    service(&mut native, &identity, Duration::from_millis(100));
    exhaust_capture(&mut native, &identity);
    registry.revoke(identity.session_id());
    native.disconnect(Instant::now() + CLEANUP_LIMIT).unwrap();
    registry.release(identity.session_id()).unwrap();
    source.stop();
}

#[test]
#[ignore = "requires a disposable private audio server and virtual fixture sinks"]
fn private_pulse_native_corked_endpoint_revokes() {
    let (registry, mut native, identity, source) = fixture();
    service(&mut native, &identity, Duration::from_millis(100));
    cork_playback(&mut native);
    assert!(native.iterate_streams(&identity, true).is_err());
    assert!(!identity.is_live());
    assert_eq!(
        *identity.lease.failure.lock().unwrap(),
        Some((OriginalMicrophoneError::Transport, "stream_corked"))
    );
    registry.revoke(identity.session_id());
    native.disconnect(Instant::now() + CLEANUP_LIMIT).unwrap();
    registry.release(identity.session_id()).unwrap();
    source.stop();
}

#[test]
#[ignore = "requires a disposable private audio server and virtual fixture sinks"]
fn private_pulse_native_suspended_endpoint_revokes() {
    let (registry, mut native, identity, source) = fixture();
    service(&mut native, &identity, Duration::from_millis(100));
    assert!(
        Command::new("pactl")
            .args(["suspend-sink", MIC_OUT_SINK, "1"])
            .status()
            .unwrap()
            .success()
    );
    let deadline = Instant::now() + Duration::from_millis(500);
    while native.playback.as_ref().unwrap().is_suspended() != Ok(true) {
        assert!(
            Instant::now() < deadline,
            "server must report suspension for this test"
        );
        native.iterate().unwrap();
        thread::sleep(POLL_INTERVAL);
    }
    assert!(
        native.iterate_streams(&identity, true).is_err(),
        "suspended consumer cannot retain live raw forwarding"
    );
    assert!(
        !identity.is_live(),
        "native service must revoke without fixture cancellation"
    );
    registry.revoke(identity.session_id());
    native.disconnect(Instant::now() + CLEANUP_LIMIT).unwrap();
    registry.release(identity.session_id()).unwrap();
    source.stop();
    assert!(
        Command::new("pactl")
            .args(["suspend-sink", MIC_OUT_SINK, "0"])
            .status()
            .unwrap()
            .success()
    );
}
