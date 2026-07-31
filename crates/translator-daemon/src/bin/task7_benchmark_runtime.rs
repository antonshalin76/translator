use std::{
    io::{self, BufRead, BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{Arc, Mutex},
    thread,
};

use clap::Parser;
use translator_audio::{
    AudioGraph, MIC_OUT_SINK, PulseAudioGraph, REMOTE_IN_SINK, SystemCommandRunner,
    default_journal_path,
};
use translator_core::{AudioDirection, TranslationMode};
use translator_daemon::{
    DuplexAudioTargets, DuplexRuntimeObserver, ProcessDuplexConfig, ProcessDuplexRunner,
    RuntimeLatencyObserver, RuntimeLease, RuntimeStore, Task7BridgeEvent, Task7BridgeFailureStage,
};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Privacy-safe runtime bridge for the Task 7 full-duplex benchmark"
)]
struct Arguments {
    #[arg(long)]
    microphone_capture: String,

    #[arg(long)]
    speaker_playback: String,

    #[arg(long)]
    python: PathBuf,

    #[arg(long)]
    sidecar_root: PathBuf,

    #[arg(long)]
    socket_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct BridgeFailure {
    stage: Task7BridgeFailureStage,
    code: &'static str,
}

impl BridgeFailure {
    const fn new(stage: Task7BridgeFailureStage, code: &'static str) -> Self {
        Self { stage, code }
    }
}

struct NdjsonEmitter {
    output: Mutex<BufWriter<io::Stdout>>,
    output_failed: tokio::sync::watch::Sender<bool>,
}

impl NdjsonEmitter {
    fn new() -> Self {
        let (output_failed, _) = tokio::sync::watch::channel(false);
        Self {
            output: Mutex::new(BufWriter::new(io::stdout())),
            output_failed,
        }
    }

    fn emit(&self, event: &Task7BridgeEvent) -> io::Result<()> {
        let result = (|| {
            let mut output = self
                .output
                .lock()
                .map_err(|_| io::Error::other("bridge output lock poisoned"))?;
            serde_json::to_writer(&mut *output, event).map_err(io::Error::other)?;
            output.write_all(b"\n")?;
            output.flush()
        })();
        if result.is_err() {
            self.output_failed.send_replace(true);
        }
        result
    }

    fn subscribe_output_failure(&self) -> tokio::sync::watch::Receiver<bool> {
        self.output_failed.subscribe()
    }
}

impl DuplexRuntimeObserver for NdjsonEmitter {
    fn observe(&self, event: translator_daemon::DuplexRuntimeEvent) {
        if self.emit(&Task7BridgeEvent::from_runtime(event)).is_err() {
            tracing::error!(event = "task7_bridge_output_failed");
        }
    }
}

struct BridgeRuntimeObserver {
    emitter: Arc<NdjsonEmitter>,
    latency: RuntimeLatencyObserver,
}

impl BridgeRuntimeObserver {
    fn new(emitter: Arc<NdjsonEmitter>, store: RuntimeStore) -> Self {
        Self {
            emitter,
            latency: RuntimeLatencyObserver::new(store),
        }
    }
}

impl DuplexRuntimeObserver for BridgeRuntimeObserver {
    fn observe(&self, event: translator_daemon::DuplexRuntimeEvent) {
        self.latency.observe(event);
        self.emitter.observe(event);
    }

    fn requested_mode(&self, direction: AudioDirection) -> Option<TranslationMode> {
        self.latency.requested_mode(direction)
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_target(false)
        .without_time()
        .compact()
        .init();

    let arguments = Arguments::parse();
    let emitter = Arc::new(NdjsonEmitter::new());
    match run_bridge(arguments, emitter.clone()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            let _ = emitter.emit(&Task7BridgeEvent::failure(failure.stage, failure.code));
            ExitCode::FAILURE
        }
    }
}

async fn run_bridge(
    arguments: Arguments,
    emitter: Arc<NdjsonEmitter>,
) -> Result<(), BridgeFailure> {
    let runtime_parent = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            BridgeFailure::new(
                Task7BridgeFailureStage::RuntimeLease,
                "runtime_directory_unavailable",
            )
        })?;
    let lease = RuntimeLease::acquire(&runtime_parent).map_err(|error| {
        BridgeFailure::new(Task7BridgeFailureStage::RuntimeLease, error.code().as_str())
    })?;
    validate_socket_parent(&arguments.socket_path, lease.token_path())?;

    let journal_path = default_journal_path().map_err(|_| {
        BridgeFailure::new(
            Task7BridgeFailureStage::AudioGraphEnsure,
            "journal_path_unavailable",
        )
    })?;
    let mut graph = PulseAudioGraph::new(SystemCommandRunner, journal_path);
    let graph_state = match graph.ensure_endpoints() {
        Ok(state) => state,
        Err(error) => {
            let failure = BridgeFailure::new(
                Task7BridgeFailureStage::AudioGraphEnsure,
                error.code().safe_code(),
            );
            if graph.cleanup_owned().is_err() {
                return Err(BridgeFailure::new(
                    Task7BridgeFailureStage::AudioGraphCleanup,
                    "audio_graph_cleanup_failed",
                ));
            }
            return Err(failure);
        }
    };

    let runtime_result = run_runtime(arguments, graph_state, emitter.clone()).await;
    let cleanup_result = graph.cleanup_owned().map_err(|_| {
        BridgeFailure::new(
            Task7BridgeFailureStage::AudioGraphCleanup,
            "audio_graph_cleanup_failed",
        )
    });
    match (runtime_result, cleanup_result) {
        (_, Err(cleanup_failure)) => return Err(cleanup_failure),
        (Err(runtime_failure), Ok(_)) => return Err(runtime_failure),
        (Ok(_), Ok(_)) => {}
    }
    drop(graph);
    drop(lease);
    emitter
        .emit(&Task7BridgeEvent::stopped())
        .map_err(|_| BridgeFailure::new(Task7BridgeFailureStage::Output, "bridge_output_failed"))
}

async fn run_runtime(
    arguments: Arguments,
    graph_state: translator_audio::AudioGraphState,
    emitter: Arc<NdjsonEmitter>,
) -> Result<(), BridgeFailure> {
    let config = ProcessDuplexConfig::from_runtime(
        arguments.python,
        arguments.sidecar_root,
        arguments.socket_path,
    )
    .map_err(|_| {
        BridgeFailure::new(
            Task7BridgeFailureStage::RuntimeConfiguration,
            "runtime_configuration_invalid",
        )
    })?;
    let store = RuntimeStore::default();
    store.set_audio_graph(graph_state);
    let observer = Arc::new(BridgeRuntimeObserver::new(emitter.clone(), store.clone()));
    let runner = ProcessDuplexRunner::with_observer(config, observer);
    let snapshot = store.snapshot();
    let targets = DuplexAudioTargets {
        microphone_capture: arguments.microphone_capture,
        microphone_playback: MIC_OUT_SINK.to_owned(),
        speaker_capture: format!("{REMOTE_IN_SINK}.monitor"),
        speaker_playback: arguments.speaker_playback,
    };
    let mut active = runner
        .start_with_audio_targets(snapshot, targets)
        .map_err(|_| {
            BridgeFailure::new(
                Task7BridgeFailureStage::RuntimeStart,
                "runtime_start_failed",
            )
        })?;

    let ready_result = emitter
        .emit(&Task7BridgeEvent::ready(std::process::id()))
        .map_err(|_| BridgeFailure::new(Task7BridgeFailureStage::Output, "bridge_output_failed"));
    let control_result = if ready_result.is_ok() {
        wait_for_stop(emitter.subscribe_output_failure()).await
    } else {
        ready_result
    };
    let stop_result = active.stop().map_err(|_| {
        BridgeFailure::new(Task7BridgeFailureStage::RuntimeStop, "runtime_stop_failed")
    });
    control_result?;
    stop_result
}

async fn wait_for_stop(
    mut output_failure: tokio::sync::watch::Receiver<bool>,
) -> Result<(), BridgeFailure> {
    tokio::select! {
        result = wait_for_stop_line() => result,
        result = tokio::signal::ctrl_c() => result.map_err(|_| {
            BridgeFailure::new(
                Task7BridgeFailureStage::ControlInput,
                "signal_handler_failed",
            )
        }),
        _ = wait_for_output_failure(&mut output_failure) => Err(BridgeFailure::new(
            Task7BridgeFailureStage::Output,
            "bridge_output_failed",
        )),
    }
}

async fn wait_for_stop_line() -> Result<(), BridgeFailure> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    thread::Builder::new()
        .name("task7-bridge-stdin".to_owned())
        .spawn(move || {
            let result = read_stop_line(io::stdin().lock());
            let _ = sender.send(result);
        })
        .map_err(|_| {
            BridgeFailure::new(Task7BridgeFailureStage::ControlInput, "stdin_thread_failed")
        })?;
    receiver.await.map_err(|_| {
        BridgeFailure::new(Task7BridgeFailureStage::ControlInput, "stdin_thread_failed")
    })?
}

fn read_stop_line(mut input: impl BufRead) -> Result<(), BridgeFailure> {
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) => return Ok(()),
            Ok(_) if line.trim() == "stop" => return Ok(()),
            Ok(_) => {}
            Err(_) => {
                return Err(BridgeFailure::new(
                    Task7BridgeFailureStage::ControlInput,
                    "stdin_read_failed",
                ));
            }
        }
    }
}

async fn wait_for_output_failure(receiver: &mut tokio::sync::watch::Receiver<bool>) {
    if *receiver.borrow() {
        return;
    }
    let _ = receiver.changed().await;
}

fn validate_socket_parent(socket_path: &Path, token_path: &Path) -> Result<(), BridgeFailure> {
    let socket_parent = socket_path
        .parent()
        .and_then(|path| path.canonicalize().ok());
    let runtime_directory = token_path
        .parent()
        .and_then(|path| path.canonicalize().ok());
    if socket_parent.is_none() || socket_parent != runtime_directory {
        return Err(BridgeFailure::new(
            Task7BridgeFailureStage::RuntimeConfiguration,
            "socket_path_outside_runtime_directory",
        ));
    }
    Ok(())
}

trait AudioGraphErrorCodeExt {
    fn safe_code(self) -> &'static str;
}

impl AudioGraphErrorCodeExt for translator_audio::AudioGraphErrorCode {
    fn safe_code(self) -> &'static str {
        match self {
            Self::PactlMissing => "pactl_missing",
            Self::GraphInspectionFailed => "graph_inspection_failed",
            Self::ModuleLoadFailed => "module_load_failed",
            Self::DuplicateEndpoint => "duplicate_endpoint",
            Self::OwnershipJournalInvalid => "ownership_journal_invalid",
            Self::OwnershipJournalIo => "ownership_journal_io",
            Self::CleanupFailed => "cleanup_failed",
            Self::RollbackFailed => "rollback_failed",
            Self::EndpointVerificationFailed => "endpoint_verification_failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use translator_core::{AudioDirection, TranslationMode};
    use translator_daemon::DuplexRuntimeEvent;

    #[test]
    fn bridge_observer_exposes_the_latency_policy_it_updates() {
        let observer =
            BridgeRuntimeObserver::new(Arc::new(NdjsonEmitter::new()), RuntimeStore::default());

        for index in 0..3 {
            let utterance_id = uuid::Uuid::new_v4();
            let capture_monotonic_ns = 1_000_000_000 + index * 10_000_000_000;
            observer.observe(DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Microphone,
                utterance_id,
                capture_monotonic_ns,
            });
            observer.observe(DuplexRuntimeEvent::AudioFrame {
                direction: AudioDirection::Microphone,
                utterance_id,
                sequence: 0,
                provider_monotonic_ns: capture_monotonic_ns + 4_000_000_000,
                observed_monotonic_ns: capture_monotonic_ns + 4_000_000_000,
                queue_lag_ms: 20,
            });
            observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
                direction: AudioDirection::Microphone,
                utterance_id,
            });
        }

        assert_eq!(
            observer.requested_mode(AudioDirection::Microphone),
            Some(TranslationMode::Balanced)
        );
    }
}
