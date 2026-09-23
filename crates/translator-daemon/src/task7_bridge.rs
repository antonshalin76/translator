use std::{
    io::{self, BufRead, BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{Arc, Mutex},
    thread,
};

use crate::{
    DuplexRuntimeObserver, ProcessDuplexConfig, ProcessDuplexRunner, RuntimeLatencyObserver,
    RuntimeLease, RuntimeStore, Task7BridgeEvent, Task7BridgeFailureStage,
};
use clap::Parser;
use translator_audio::{
    AudioGraph, PulseAudioGraph, SystemCommandRunner, default_journal_path,
    inspect_task7_endpoints_until,
};
use translator_core::{AudioDirection, TranslationMode};

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
    output: Mutex<BufWriter<Box<dyn Write + Send>>>,
    output_failed: tokio::sync::watch::Sender<bool>,
}

impl NdjsonEmitter {
    fn new() -> Self {
        Self::with_output(io::stdout())
    }

    fn with_output(output: impl Write + Send + 'static) -> Self {
        let (output_failed, _) = tokio::sync::watch::channel(false);
        Self {
            output: Mutex::new(BufWriter::new(Box::new(output))),
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
    fn observe(&self, event: crate::DuplexRuntimeEvent) {
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
    fn observe(&self, event: crate::DuplexRuntimeEvent) {
        self.latency.observe(event);
        self.emitter.observe(event);
    }

    fn requested_mode(&self, direction: AudioDirection) -> Option<TranslationMode> {
        self.latency.requested_mode(direction)
    }

    fn reset_direction(&self, direction: AudioDirection) {
        self.latency.reset_direction(direction);
    }
}

pub fn run_task7_bridge() -> ExitCode {
    crate::install_private_panic_hook();
    match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime.block_on(async_main()),
        Err(_) => {
            let _ = std::io::stderr()
                .write_all(b"{\"event\":\"runtime_start_failed\",\"code\":\"internal_error\"}\n");
            ExitCode::FAILURE
        }
    }
}

async fn async_main() -> ExitCode {
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
    let deadline = tokio::time::Instant::now() + crate::DIRECTION_CLEANUP_BUDGET;
    let endpoint_facts = inspect_task7_endpoints_until(
        &SystemCommandRunner,
        &arguments.microphone_capture,
        &arguments.speaker_playback,
        deadline.into_std(),
    )
    .map_err(|_| {
        BridgeFailure::new(
            Task7BridgeFailureStage::RuntimeConfiguration,
            "audio_facts_unavailable",
        )
    })?;

    let journal_path = default_journal_path().map_err(|_| {
        BridgeFailure::new(
            Task7BridgeFailureStage::AudioGraphEnsure,
            "journal_path_unavailable",
        )
    })?;
    let graph = PulseAudioGraph::new(SystemCommandRunner, journal_path);
    let runtime_emitter = emitter.clone();
    run_owned_graph(
        lease,
        graph,
        emitter,
        deadline,
        move |lease, graph, graph_state| {
            let store = RuntimeStore::default();
            store.set_audio_graph(graph_state);
            let admitted = crate::acoustic_admission::admit_task7(
                store.snapshot(),
                endpoint_facts,
                lease,
                graph,
            );
            async move {
                match admitted {
                    Ok(admitted) => {
                        run_runtime(arguments, admitted, store, runtime_emitter, deadline).await
                    }
                    Err(_) => Err(BridgeFailure::new(
                        Task7BridgeFailureStage::RuntimeConfiguration,
                        "runtime_configuration_invalid",
                    )),
                }
            }
        },
    )
    .await
}

async fn run_owned_graph<G, F, Fut>(
    lease: RuntimeLease,
    mut graph: G,
    emitter: Arc<NdjsonEmitter>,
    deadline: tokio::time::Instant,
    run: F,
) -> Result<(), BridgeFailure>
where
    G: AudioGraph,
    F: FnOnce(&RuntimeLease, &G, translator_audio::AudioGraphState) -> Fut,
    Fut: std::future::Future<Output = Result<(), BridgeFailure>>,
{
    if tokio::time::Instant::now() >= deadline {
        return Err(BridgeFailure::new(
            Task7BridgeFailureStage::RuntimeConfiguration,
            "audio_facts_expired",
        ));
    }
    let runtime_result = match graph.ensure_endpoints_until(deadline.into_std()) {
        Ok(state) => run(&lease, &graph, state).await,
        Err(error) => Err(BridgeFailure::new(
            Task7BridgeFailureStage::AudioGraphEnsure,
            error.code().safe_code(),
        )),
    };
    let cleanup_result = drain_resource(|deadline| {
        graph
            .cleanup_owned_until(deadline.into_std())
            .map(|_| ())
            .map_err(|_| {
                BridgeFailure::new(
                    Task7BridgeFailureStage::AudioGraphCleanup,
                    "audio_graph_cleanup_failed",
                )
            })
    })
    .await;
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
    admitted: crate::AdmittedDuplex,
    store: RuntimeStore,
    emitter: Arc<NdjsonEmitter>,
    deadline: tokio::time::Instant,
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
    let observer = Arc::new(BridgeRuntimeObserver::new(emitter.clone(), store.clone()));
    let runner = ProcessDuplexRunner::with_observer(config, observer);
    let mut active = start_runtime(deadline, |deadline| {
        crate::DuplexRunner::start(&runner, admitted, deadline)
    })
    .await?;

    let ready_result = emitter
        .emit(&Task7BridgeEvent::ready(std::process::id()))
        .map_err(|_| BridgeFailure::new(Task7BridgeFailureStage::Output, "bridge_output_failed"));
    let control_result = if ready_result.is_ok() {
        wait_for_stop(emitter.subscribe_output_failure()).await
    } else {
        ready_result
    };
    let stop_result = drain_runtime(active.as_mut()).await;
    control_result?;
    stop_result
}

async fn start_runtime(
    deadline: tokio::time::Instant,
    start: impl FnOnce(tokio::time::Instant) -> crate::DuplexStartResult,
) -> Result<Box<dyn crate::ActiveDuplexRuntime>, BridgeFailure> {
    if tokio::time::Instant::now() >= deadline {
        return Err(BridgeFailure::new(
            Task7BridgeFailureStage::RuntimeStart,
            "runtime_start_failed",
        ));
    }
    match start(deadline) {
        Ok(runtime) => Ok(runtime),
        Err(failure) => {
            if let Some(mut cleanup) = failure.into_parts().1 {
                let _ = drain_runtime(cleanup.as_mut()).await;
            }
            Err(BridgeFailure::new(
                Task7BridgeFailureStage::RuntimeStart,
                "runtime_start_failed",
            ))
        }
    }
}

async fn drain_runtime(active: &mut dyn crate::ActiveDuplexRuntime) -> Result<(), BridgeFailure> {
    drain_resource(|deadline| {
        active.stop(deadline).map_err(|_| {
            BridgeFailure::new(Task7BridgeFailureStage::RuntimeStop, "runtime_stop_failed")
        })
    })
    .await
}

async fn drain_resource(
    mut attempt: impl FnMut(tokio::time::Instant) -> Result<(), BridgeFailure>,
) -> Result<(), BridgeFailure> {
    let mut failure = None;
    loop {
        let deadline = tokio::time::Instant::now() + crate::RUNTIME_CLEANUP_BUDGET;
        match attempt(deadline) {
            Ok(()) => return failure.map_or(Ok(()), Err),
            Err(error) => {
                failure.get_or_insert(error);
                tracing::error!(event = "task7_cleanup_pending", code = error.code);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
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
            Self::OwnershipJournalBusy => "ownership_journal_busy",
            Self::DeadlineExpired => "deadline_expired",
            Self::CleanupFailed => "cleanup_failed",
            Self::RollbackFailed => "rollback_failed",
            Self::EndpointVerificationFailed => "endpoint_verification_failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DuplexRuntimeEvent;
    use translator_core::{AudioDirection, TranslationMode};

    #[derive(Clone, Default)]
    struct RecordedOutput(Arc<Mutex<Vec<u8>>>);
    impl Write for RecordedOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct GraphProbe {
        ensure_calls: usize,
        cleanup_calls: Vec<(tokio::time::Instant, tokio::time::Instant, bool)>,
        dropped_after_cleanup: Option<usize>,
    }

    struct FixtureGraph {
        probe: Arc<Mutex<GraphProbe>>,
        lease_parent: PathBuf,
        fail_ensure: bool,
        fail_first_cleanup: bool,
    }

    fn graph_error() -> translator_audio::AudioGraphError {
        let parent = tempfile::tempdir().unwrap();
        PulseAudioGraph::new(SystemCommandRunner, parent.path().join("unused.json"))
            .inspect_until(std::time::Instant::now())
            .unwrap_err()
    }

    impl AudioGraph for FixtureGraph {
        fn ensure_endpoints_until(
            &mut self,
            _: std::time::Instant,
        ) -> Result<translator_audio::AudioGraphState, translator_audio::AudioGraphError> {
            self.probe.lock().unwrap().ensure_calls += 1;
            if self.fail_ensure {
                return Err(graph_error());
            }
            Ok(translator_audio::AudioGraphState {
                health: translator_audio::GraphHealth::Ready,
                endpoints: Vec::new(),
                owned_module_ids: Vec::new(),
                safe_error: None,
            })
        }
        fn inspect_until(
            &self,
            _: std::time::Instant,
        ) -> Result<translator_audio::AudioGraphState, translator_audio::AudioGraphError> {
            panic!("bridge cleanup must not inspect through another owner")
        }
        fn cleanup_owned_until(
            &mut self,
            deadline: std::time::Instant,
        ) -> Result<Vec<u32>, translator_audio::AudioGraphError> {
            let lease_held = RuntimeLease::acquire(&self.lease_parent).is_err();
            let mut probe = self.probe.lock().unwrap();
            probe.cleanup_calls.push((
                tokio::time::Instant::now(),
                tokio::time::Instant::from_std(deadline),
                lease_held,
            ));
            if self.fail_first_cleanup && probe.cleanup_calls.len() == 1 {
                Err(graph_error())
            } else {
                Ok(vec![73])
            }
        }
    }

    impl Drop for FixtureGraph {
        fn drop(&mut self) {
            let mut probe = self.probe.lock().unwrap();
            probe.dropped_after_cleanup = Some(probe.cleanup_calls.len());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn owned_graph_composition_preserves_success_and_original_errors() {
        for (fail_ensure, fail_runtime) in [(false, false), (true, false), (false, true)] {
            let parent = tempfile::tempdir().unwrap();
            let lease = RuntimeLease::acquire(parent.path()).unwrap();
            let probe = Arc::new(Mutex::new(GraphProbe::default()));
            let graph = FixtureGraph {
                probe: probe.clone(),
                lease_parent: parent.path().into(),
                fail_ensure,
                fail_first_cleanup: false,
            };
            let output = RecordedOutput::default();
            let runtime_calls = std::cell::Cell::new(0);
            let result = run_owned_graph(
                lease,
                graph,
                Arc::new(NdjsonEmitter::with_output(output.clone())),
                tokio::time::Instant::now() + crate::DIRECTION_CLEANUP_BUDGET,
                |_, _, _| {
                    runtime_calls.set(runtime_calls.get() + 1);
                    std::future::ready(if fail_runtime {
                        Err(BridgeFailure::new(
                            Task7BridgeFailureStage::RuntimeStart,
                            "runtime_start_failed",
                        ))
                    } else {
                        Ok(())
                    })
                },
            )
            .await;
            let reacquired = RuntimeLease::acquire(parent.path());
            assert!(reacquired.is_ok());
            drop(reacquired);
            assert_eq!(runtime_calls.get(), usize::from(!fail_ensure));
            let probe = probe.lock().unwrap();
            assert_eq!(probe.cleanup_calls.len(), 1);
            assert!(probe.cleanup_calls[0].2);
            assert_eq!(probe.dropped_after_cleanup, Some(1));
            let bytes = output.0.lock().unwrap();
            if fail_ensure {
                assert_eq!(result.unwrap_err().code, "deadline_expired");
                assert!(bytes.is_empty());
            } else if fail_runtime {
                assert_eq!(result.unwrap_err().code, "runtime_start_failed");
                assert!(bytes.is_empty());
            } else {
                result.unwrap();
                let event: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(event["event"], "stopped");
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn failed_graph_cleanup_retains_owner_and_lease_in_both_composition_branches() {
        for fail_ensure in [false, true] {
            let parent = tempfile::tempdir().unwrap();
            let lease = RuntimeLease::acquire(parent.path()).unwrap();
            let probe = Arc::new(Mutex::new(GraphProbe::default()));
            let graph = FixtureGraph {
                probe: probe.clone(),
                lease_parent: parent.path().into(),
                fail_ensure,
                fail_first_cleanup: true,
            };
            let output = RecordedOutput::default();
            let result = run_owned_graph(
                lease,
                graph,
                Arc::new(NdjsonEmitter::with_output(output.clone())),
                tokio::time::Instant::now() + crate::DIRECTION_CLEANUP_BUDGET,
                |_, _, _| {
                    std::future::ready(Err(BridgeFailure::new(
                        Task7BridgeFailureStage::RuntimeStart,
                        "runtime_start_failed",
                    )))
                },
            )
            .await;
            let reacquired = RuntimeLease::acquire(parent.path());
            assert!(reacquired.is_ok());
            drop(reacquired);
            let probe = probe.lock().unwrap();
            assert_eq!(result.unwrap_err().code, "audio_graph_cleanup_failed");
            assert!(output.0.lock().unwrap().is_empty());
            assert_eq!(
                probe.dropped_after_cleanup,
                Some(2),
                "ensure_failed={fail_ensure}"
            );
            assert_eq!(probe.cleanup_calls.len(), 2);
            for (admitted, deadline, lease_held) in &probe.cleanup_calls {
                assert_eq!(*deadline, *admitted + crate::RUNTIME_CLEANUP_BUDGET);
                assert!(*lease_held);
            }
            assert_eq!(
                probe.cleanup_calls[1].0 - probe.cleanup_calls[0].0,
                std::time::Duration::from_secs(1)
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn expired_after_facts_never_invokes_graph_maintenance() {
        let parent = tempfile::tempdir().unwrap();
        let lease = RuntimeLease::acquire(parent.path()).unwrap();
        let probe = Arc::new(Mutex::new(GraphProbe::default()));
        let graph = FixtureGraph {
            probe: probe.clone(),
            lease_parent: parent.path().into(),
            fail_ensure: false,
            fail_first_cleanup: false,
        };
        let output = RecordedOutput::default();
        let runtime_calls = std::cell::Cell::new(0);
        let result = run_owned_graph(
            lease,
            graph,
            Arc::new(NdjsonEmitter::with_output(output.clone())),
            tokio::time::Instant::now(),
            |_, _, _| {
                runtime_calls.set(runtime_calls.get() + 1);
                std::future::ready(Err(BridgeFailure::new(
                    Task7BridgeFailureStage::RuntimeStart,
                    "runtime_start_failed",
                )))
            },
        )
        .await;
        assert!(RuntimeLease::acquire(parent.path()).is_ok());
        assert!(result.is_err());
        assert!(output.0.lock().unwrap().is_empty());
        assert_eq!(runtime_calls.get(), 0);
        assert_eq!(probe.lock().unwrap().ensure_calls, 0);
        assert!(probe.lock().unwrap().cleanup_calls.is_empty());
    }

    #[derive(Default)]
    struct CleanupProbe {
        attempts: std::sync::atomic::AtomicUsize,
        dropped_unclean: std::sync::atomic::AtomicBool,
        deadlines: std::sync::Mutex<Vec<(tokio::time::Instant, tokio::time::Instant)>>,
    }

    struct CleanupRuntime(Arc<CleanupProbe>);

    impl crate::ActiveDuplexRuntime for CleanupRuntime {
        fn stop(
            &mut self,
            deadline: tokio::time::Instant,
        ) -> Result<(), crate::DuplexRuntimeError> {
            self.0
                .deadlines
                .lock()
                .unwrap()
                .push((tokio::time::Instant::now(), deadline));
            let attempt = self
                .0
                .attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if attempt == 0 {
                Err(crate::DuplexRuntimeError::StopFailed)
            } else {
                Ok(())
            }
        }
    }

    impl Drop for CleanupRuntime {
        fn drop(&mut self) {
            self.0.dropped_unclean.store(
                self.0.attempts.load(std::sync::atomic::Ordering::SeqCst) < 2,
                std::sync::atomic::Ordering::SeqCst,
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn successful_start_forwards_exact_admission_deadline_once() {
        let probe = Arc::new(CleanupProbe::default());
        let admitted = tokio::time::Instant::now();
        let mut observed = Vec::new();
        let mut active = start_runtime(admitted + crate::DIRECTION_CLEANUP_BUDGET, |deadline| {
            observed.push(deadline);
            Ok(Box::new(CleanupRuntime(probe.clone())))
        })
        .await
        .unwrap();
        let cleanup = drain_runtime(active.as_mut()).await;
        drop(active);
        assert_eq!(observed, vec![admitted + crate::DIRECTION_CLEANUP_BUDGET]);
        assert_eq!(cleanup.unwrap_err().code, "runtime_stop_failed");
        assert_cleanup_deadlines(&probe);
        assert!(
            !probe
                .dropped_unclean
                .load(std::sync::atomic::Ordering::SeqCst)
        );
    }

    fn assert_cleanup_deadlines(probe: &CleanupProbe) {
        let observed = probe.deadlines.lock().unwrap();
        assert_eq!(observed.len(), 2);
        for (admitted, deadline) in observed.iter() {
            assert_eq!(*deadline, *admitted + crate::RUNTIME_CLEANUP_BUDGET);
        }
        assert_eq!(
            observed[1].0 - observed[0].0,
            std::time::Duration::from_secs(1)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn failed_start_drains_owned_cleanup_before_returning_the_original_failure() {
        let probe = Arc::new(CleanupProbe::default());
        let failure = crate::DuplexStartFailure::cleanup_pending(
            crate::DuplexRuntimeError::StartFailed,
            Box::new(CleanupRuntime(Arc::clone(&probe))),
        );
        let started = tokio::time::Instant::now();
        let mut observed = Vec::new();
        let result = start_runtime(started + crate::DIRECTION_CLEANUP_BUDGET, |deadline| {
            observed.push(deadline);
            Err(failure)
        })
        .await;
        assert_eq!(observed, vec![started + crate::DIRECTION_CLEANUP_BUDGET]);
        assert!(matches!(
            result,
            Err(BridgeFailure {
                code: "runtime_start_failed",
                ..
            })
        ));
        assert_eq!(probe.attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(started.elapsed() >= std::time::Duration::from_secs(1));
        assert_cleanup_deadlines(&probe);
        assert!(
            !probe
                .dropped_unclean
                .load(std::sync::atomic::Ordering::SeqCst)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn failed_stop_drains_before_returning_and_never_turns_a_failed_run_green() {
        let probe = Arc::new(CleanupProbe::default());
        let mut runtime = CleanupRuntime(Arc::clone(&probe));
        let started = tokio::time::Instant::now();
        let result = drain_runtime(&mut runtime).await;
        let attempts_at_return = probe.attempts.load(std::sync::atomic::Ordering::SeqCst);
        // Keep the failing RED itself clean, without changing the observed result.
        if attempts_at_return < 2 {
            crate::ActiveDuplexRuntime::stop(
                &mut runtime,
                tokio::time::Instant::now() + crate::RUNTIME_CLEANUP_BUDGET,
            )
            .unwrap();
        }
        drop(runtime);
        assert_eq!(result.unwrap_err().code, "runtime_stop_failed");
        assert_eq!(attempts_at_return, 2);
        assert!(started.elapsed() >= std::time::Duration::from_secs(1));
        assert_cleanup_deadlines(&probe);
        assert!(
            !probe
                .dropped_unclean
                .load(std::sync::atomic::Ordering::SeqCst)
        );
    }

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

    #[test]
    fn bridge_forwards_direction_reset_without_erasing_peer_latency() {
        let observer =
            BridgeRuntimeObserver::new(Arc::new(NdjsonEmitter::new()), RuntimeStore::default());
        let direct = RuntimeLatencyObserver::new(RuntimeStore::default());
        let mut terminals = Vec::new();
        for direction in [AudioDirection::Microphone, AudioDirection::Speaker] {
            for index in 0..3 {
                let utterance_id = uuid::Uuid::new_v4();
                let capture_monotonic_ns = 1_000_000_000 + index * 10_000_000_000;
                for event in [
                    DuplexRuntimeEvent::SpeechStarted {
                        direction,
                        utterance_id,
                        capture_monotonic_ns,
                    },
                    DuplexRuntimeEvent::AudioFrame {
                        direction,
                        utterance_id,
                        sequence: 0,
                        provider_monotonic_ns: capture_monotonic_ns + 4_000_000_000,
                        observed_monotonic_ns: capture_monotonic_ns + 4_000_000_000,
                        queue_lag_ms: 20,
                    },
                ] {
                    observer.observe(event);
                    direct.observe(event);
                }
                terminals.push(DuplexRuntimeEvent::UtteranceTerminal {
                    direction,
                    utterance_id,
                });
            }
        }
        observer.reset_direction(AudioDirection::Microphone);
        direct.reset_direction(AudioDirection::Microphone);
        for event in terminals {
            observer.observe(event);
            direct.observe(event);
        }

        // The direct owner must work before a failure can identify the wrapper.
        assert_eq!(
            direct.requested_mode(AudioDirection::Microphone),
            Some(TranslationMode::QualityFirst)
        );
        assert_eq!(
            direct.requested_mode(AudioDirection::Speaker),
            Some(TranslationMode::Balanced)
        );
        assert_eq!(
            observer.requested_mode(AudioDirection::Speaker),
            Some(TranslationMode::Balanced)
        );
        assert_eq!(
            observer.requested_mode(AudioDirection::Microphone),
            Some(TranslationMode::QualityFirst),
            "a retired direction's terminal must not become a stale latency sample"
        );
    }
}
