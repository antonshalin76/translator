use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::Write;
use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use translator_audio::{AllowedApplication, RouteCandidate, RouteResolution, RoutingState};
use translator_daemon::{
    DebugCaptureLimits, DebugCaptureStopReason, DebugCaptureStore, DebugTextBuffer, DebugTextEvent,
    FreeSpaceProbe, RuntimeMutationError, RuntimeStore, SecureRuntimeErrorCode,
};

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Default)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);

struct LogWriter(Arc<Mutex<Vec<u8>>>);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuffer {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogWriter(self.0.clone())
    }
}

impl Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn debug_text_is_memory_only_bounded_and_cleared_when_disabled() {
    let marker = "private-spoken-marker";
    let mut buffer = DebugTextBuffer::default();
    assert!(!buffer.push(DebugTextEvent::new(marker, "translation")));

    buffer.set_enabled(true);
    for index in 0..250 {
        assert!(buffer.push(DebugTextEvent::new(
            format!("{marker}-{index}"),
            "translation"
        )));
    }
    assert_eq!(buffer.len(), 200);
    assert!(buffer.bytes_used() <= 1024 * 1024);
    assert!(
        !serde_json::to_string(&buffer.safe_status())
            .unwrap()
            .contains(marker)
    );

    for index in 0..200 {
        assert!(buffer.push(DebugTextEvent::new(
            format!("{index}-{}", "x".repeat(10_000)),
            "translation"
        )));
    }
    assert!(buffer.len() < 200);
    assert!(buffer.bytes_used() <= 1024 * 1024);
    let len_before = buffer.len();
    let bytes_before = buffer.bytes_used();
    assert!(!buffer.push(DebugTextEvent::new(
        "x".repeat(1024 * 1024 + 1),
        "translation"
    )));
    assert_eq!(buffer.len(), len_before);
    assert_eq!(buffer.bytes_used(), bytes_before);

    buffer.set_enabled(false);
    assert_eq!(buffer.len(), 0);
    assert_eq!(buffer.bytes_used(), 0);
}

#[test]
fn provider_switch_and_session_stop_clear_debug_text() {
    let mut buffer = DebugTextBuffer::default();
    buffer.set_enabled(true);
    buffer.push(DebugTextEvent::new("transcript", "translation"));
    buffer.clear_for_provider_switch();
    assert_eq!(buffer.len(), 0);
    buffer.push(DebugTextEvent::new("transcript", "translation"));
    buffer.clear_for_session_stop();
    assert_eq!(buffer.len(), 0);
}

#[test]
fn runtime_toggle_clears_text_without_enabling_debug_capture() {
    let store = RuntimeStore::default();
    store.set_debug_text_enabled(true);
    assert!(store.record_debug_text(DebugTextEvent::new("transcript", "translation")));
    assert_eq!(store.debug_text_status().event_count, 1);
    assert!(!store.snapshot().debug_capture_enabled);

    store.set_debug_text_enabled(false);
    assert_eq!(store.debug_text_status().event_count, 0);
    assert!(!store.snapshot().debug_capture_enabled);
}

#[test]
fn route_logs_expose_only_bounded_technical_fields() {
    let marker = "private-route-content-marker";
    let logs = LogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(logs.clone())
        .with_ansi(false)
        .without_time()
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        RuntimeStore::default().set_routes(RoutingState {
            candidates: vec![RouteCandidate {
                stream_id: 42,
                application: AllowedApplication::Firefox,
                stable_app_key: marker.to_owned(),
                application_name: marker.to_owned(),
                process_binary: marker.to_owned(),
                pipewire_node_name: None,
                media_role: Some(marker.to_owned()),
                description: Some(marker.to_owned()),
                current_sink_id: 1,
                current_sink_name: marker.to_owned(),
                call_like: true,
            }],
            source_outputs: Vec::new(),
            conflicting_stream_ids: Vec::new(),
            active_route: None,
            resolution: RouteResolution::AwaitingSelection,
        });
    });

    let output = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    assert!(output.contains("route_state_changed"));
    assert!(output.contains("candidate_count=1"));
    assert!(!output.contains(marker));
}

#[test]
fn capture_file_is_exclusive_user_only_and_off_in_new_store() {
    let temp = tempfile::tempdir().unwrap();
    let defaults = DebugCaptureLimits::default();
    assert_eq!(defaults.max_duration_ms(), 10 * 60 * 1_000);
    assert_eq!(defaults.max_bytes(), 500 * 1024 * 1024);
    assert_eq!(defaults.minimum_free_bytes(), 5 * GIB);
    let store = DebugCaptureStore::open(temp.path(), defaults).unwrap();
    assert!(!store.is_active());
    let session = store.start("capture-a", 0).unwrap();
    assert!(store.is_active());

    let directory = fs::metadata(store.directory_path()).unwrap();
    assert_eq!(directory.permissions().mode() & 0o777, 0o700);
    let metadata = fs::metadata(session.path()).unwrap();
    assert!(metadata.is_file());
    assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    assert_eq!(metadata.uid(), fs::metadata(temp.path()).unwrap().uid());

    drop(session);
    assert!(!store.is_active());
    let restarted = DebugCaptureStore::open(temp.path(), DebugCaptureLimits::default()).unwrap();
    assert!(!restarted.is_active());
}

#[test]
fn capture_symlink_is_rejected_without_touching_target() {
    let temp = tempfile::tempdir().unwrap();
    let store = DebugCaptureStore::open(temp.path(), DebugCaptureLimits::default()).unwrap();
    let target = temp.path().join("private-target");
    fs::write(&target, b"private-marker").unwrap();
    symlink(&target, store.directory_path().join("capture-a.pcm")).unwrap();

    let error = store.start("capture-a", 0).unwrap_err();

    assert_eq!(error.code(), SecureRuntimeErrorCode::UnsafePath);
    assert_eq!(fs::read(&target).unwrap(), b"private-marker");
}

#[test]
fn capture_rejects_parent_symlink_traversal_and_existing_regular_file() {
    let temp = tempfile::tempdir().unwrap();
    let target = tempfile::tempdir().unwrap();
    fs::create_dir(temp.path().join("translator")).unwrap();
    fs::set_permissions(
        temp.path().join("translator"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    symlink(target.path(), temp.path().join("translator/debug")).unwrap();
    let error = DebugCaptureStore::open(temp.path(), DebugCaptureLimits::default()).unwrap_err();
    assert_eq!(error.code(), SecureRuntimeErrorCode::UnsafePath);
    assert!(fs::read_dir(target.path()).unwrap().next().is_none());

    let temp = tempfile::tempdir().unwrap();
    let store = DebugCaptureStore::open(temp.path(), DebugCaptureLimits::default()).unwrap();
    assert_eq!(
        store.start("../escape", 0).unwrap_err().code(),
        SecureRuntimeErrorCode::UnsafePath
    );
    let existing = store.directory_path().join("existing.pcm");
    fs::write(&existing, b"existing-marker").unwrap();
    assert_eq!(
        store.start("existing", 0).unwrap_err().code(),
        SecureRuntimeErrorCode::UnsafePath
    );
    assert_eq!(fs::read(existing).unwrap(), b"existing-marker");
}

#[derive(Debug)]
struct FakeFreeSpace {
    values: Mutex<VecDeque<u64>>,
    calls: AtomicUsize,
}

impl FakeFreeSpace {
    fn new(values: impl IntoIterator<Item = u64>) -> Self {
        Self {
            values: Mutex::new(values.into_iter().collect()),
            calls: AtomicUsize::new(0),
        }
    }
}

impl FreeSpaceProbe for FakeFreeSpace {
    fn available_bytes(&self, directory: &File) -> std::io::Result<u64> {
        assert!(directory.metadata()?.is_dir());
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.values.lock().unwrap().pop_front().unwrap())
    }
}

#[test]
fn capture_stops_before_byte_time_or_projected_free_space_hard_bounds() {
    let temp = tempfile::tempdir().unwrap();
    let limits = DebugCaptureLimits::new(1_000, 10, 5 * GIB);
    let probe = Arc::new(FakeFreeSpace::new([6 * GIB; 8]));
    let store = DebugCaptureStore::open_with_probe(temp.path(), limits, probe.clone()).unwrap();

    let mut byte_limited = store.start("bytes", 0).unwrap();
    assert!(byte_limited.append(&[0; 8], 100).is_ok());
    assert_eq!(
        byte_limited.append(&[0; 3], 200).unwrap_err(),
        DebugCaptureStopReason::ByteLimit
    );
    assert!(!store.is_active());
    assert_eq!(fs::metadata(byte_limited.path()).unwrap().len(), 8);
    assert_eq!(
        byte_limited.append(&[0], 201).unwrap_err(),
        DebugCaptureStopReason::ByteLimit
    );
    assert_eq!(fs::metadata(byte_limited.path()).unwrap().len(), 8);
    let replacement = store.start("bytes-replacement", 202).unwrap();
    assert!(store.is_active());
    assert_eq!(
        byte_limited.append(&[0], 203).unwrap_err(),
        DebugCaptureStopReason::ByteLimit
    );
    drop(replacement);

    let mut time_limited = store.start("time", 0).unwrap();
    assert_eq!(
        time_limited.append(&[0], 1_000).unwrap_err(),
        DebugCaptureStopReason::TimeLimit
    );
    assert!(!store.is_active());
    assert_eq!(fs::metadata(time_limited.path()).unwrap().len(), 0);

    let exact_probe = Arc::new(FakeFreeSpace::new([6 * GIB, 5 * GIB + 4]));
    let exact_store =
        DebugCaptureStore::open_with_probe(temp.path(), limits, exact_probe.clone()).unwrap();
    let mut exact = exact_store.start("space-exact", 0).unwrap();
    assert!(exact.append(&[0; 4], 1).is_ok());
    assert_eq!(exact_probe.calls.load(Ordering::SeqCst), 2);
    drop(exact);

    let low_probe = Arc::new(FakeFreeSpace::new([6 * GIB, 5 * GIB + 3]));
    let low_store =
        DebugCaptureStore::open_with_probe(temp.path(), limits, low_probe.clone()).unwrap();
    let mut space_limited = low_store.start("space-low", 0).unwrap();
    assert_eq!(
        space_limited.append(&[0; 4], 1).unwrap_err(),
        DebugCaptureStopReason::LowFreeSpace
    );
    assert_eq!(low_probe.calls.load(Ordering::SeqCst), 2);
    assert!(!low_store.is_active());
    assert_eq!(fs::metadata(space_limited.path()).unwrap().len(), 0);
}

#[test]
fn runtime_capture_hard_stop_clears_enabled_state_and_rejects_more_audio() {
    let temp = tempfile::tempdir().unwrap();
    let limits = DebugCaptureLimits::new(1_000, 4, 0);
    let probe = Arc::new(FakeFreeSpace::new([1024; 2]));
    let capture_store = DebugCaptureStore::open_with_probe(temp.path(), limits, probe).unwrap();
    let store = RuntimeStore::default();
    store.configure_debug_capture(capture_store);
    store.set_debug_capture_enabled(true).unwrap();

    assert!(store.append_debug_capture(&[0; 4], 1).is_ok());
    assert_eq!(
        store.append_debug_capture(&[0], 2),
        Err(RuntimeMutationError::DebugCaptureStopped(
            DebugCaptureStopReason::ByteLimit
        ))
    );
    assert!(!store.snapshot().debug_capture_enabled);
    assert_eq!(
        store.append_debug_capture(&[0], 3),
        Err(RuntimeMutationError::DebugCaptureUnavailable)
    );
}
