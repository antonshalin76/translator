use std::{
    future,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};
use tokio::sync::watch;
use translator_audio::{
    AecCapability, AudioGraphState, DeviceHealth, DeviceSelectionState, GraphHealth, OutputMode,
    PcmFrame, PhysicalDevice, ProcessIdentity, RouteResolution, RoutingState, StreamPcmFormat,
    VirtualPeerCapability,
};
use translator_core::AudioDirection;
use translator_daemon::{
    AcousticSafety, ActiveDuplexRuntime, AdmittedDuplex, AudioOperationGate, AudioOperationState,
    DeviceState, DuplexRuntimeEvent, DuplexRuntimeObserver, RoundTripAudioWorker,
    RoundTripAudioWorkerFactory, RoundTripCheckpoint, RoundTripController, RoundTripDuplexFactory,
    RoundTripProcessError, RoundTripProcessRunner, RoundTripRuntimeHandle, RoundTripWorkerFuture,
    RuntimeSnapshot, RuntimeStore, SafeProviderErrorCode, TerminalOutcome,
    VirtualPeerRouteController, VirtualPeerRouteControllerFactory,
};
use uuid::Uuid;

#[path = "support/audio_facts.rs"]
mod audio_facts;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scenario {
    ActiveObserverReset,
    CleanupObserverReset,
    StartupFactoryExpiry,
    ActiveFactoryExpiry,
    StartupAudioFactoryExpiry,
    ActiveAudioFactoryExpiry,
    StartCleanupPending,
    PanicAfterRoute,
    PanicClearSensitive,
    PartialRouteFailure,
    RouteFailureBeforeAcquisition,
    Happy,
    Recursion,
    Timeout,
    SlowCleanupAfterRoute,
    Stop,
    ForgedCapability,
    StaleCapability,
    FailAfterRoute,
    DroppedReceipt,
    CorruptReceipt,
    PeerPersists,
    ProviderDrop,
    ProviderCancelled,
    IncomingProviderDrop,
    NoDebugTextStageEvents,
    AudioWithoutOutcome,
    IncomingDrainWait,
}

struct Shared {
    scenario: Scenario,
    actions: Mutex<Vec<&'static str>>,
    observer: Mutex<Option<Arc<dyn DuplexRuntimeObserver>>>,
    expected_frames: Vec<PcmFrame>,
    reinjected_frames: Mutex<Vec<PcmFrame>>,
    workers: AtomicUsize,
    max_workers: AtomicUsize,
    peer_alive: AtomicBool,
    outgoing_utterance: Mutex<Option<Uuid>>,
    incoming_utterance: Mutex<Option<Uuid>>,
    cleanup_failures: AtomicUsize,
    duplex_cleanup_failures: AtomicUsize,
    cleanup_panics: AtomicUsize,
    route_restore_failures: AtomicUsize,
    route_effect_pending: AtomicBool,
    route_events: Mutex<Vec<(Uuid, &'static str)>>,
    start_deadlines: Mutex<Vec<tokio::time::Instant>>,
    cleanup_deadlines: Mutex<Vec<tokio::time::Instant>>,
    startup_admission: Mutex<Option<Instant>>,
    short_startup_budget: AtomicBool,
}

impl Shared {
    fn new(scenario: Scenario) -> Arc<Self> {
        Arc::new(Self {
            scenario,
            actions: Mutex::new(Vec::new()),
            observer: Mutex::new(None),
            expected_frames: frames(),
            reinjected_frames: Mutex::new(Vec::new()),
            workers: AtomicUsize::new(0),
            max_workers: AtomicUsize::new(0),
            peer_alive: AtomicBool::new(false),
            outgoing_utterance: Mutex::new(None),
            incoming_utterance: Mutex::new(None),
            cleanup_failures: AtomicUsize::new(0),
            duplex_cleanup_failures: AtomicUsize::new(0),
            cleanup_panics: AtomicUsize::new(0),
            route_restore_failures: AtomicUsize::new(0),
            route_effect_pending: AtomicBool::new(false),
            route_events: Mutex::new(Vec::new()),
            start_deadlines: Mutex::new(Vec::new()),
            cleanup_deadlines: Mutex::new(Vec::new()),
            startup_admission: Mutex::new(None),
            short_startup_budget: AtomicBool::new(false),
        })
    }

    fn action(&self, action: &'static str) {
        self.actions.lock().unwrap().push(action);
    }

    fn emit_outgoing(&self) {
        let observer = self.observer.lock().unwrap().clone().unwrap();
        let utterance_id = Uuid::new_v4();
        *self.outgoing_utterance.lock().unwrap() = Some(utterance_id);
        let onset = monotonic_ns();
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Microphone,
            utterance_id,
            capture_monotonic_ns: onset,
        });
        observer.observe(audio_frame(AudioDirection::Microphone, Uuid::new_v4(), 0));
        observer.observe(audio_frame(AudioDirection::Speaker, utterance_id, 0));
        observer.observe(DuplexRuntimeEvent::ProviderError {
            direction: AudioDirection::Microphone,
            utterance_id: Some(utterance_id),
            code: SafeProviderErrorCode::ProviderUnavailable,
            retryable: true,
        });
        observer.observe(DuplexRuntimeEvent::UtteranceTerminalOutcome {
            direction: AudioDirection::Microphone,
            utterance_id: Uuid::new_v4(),
            outcome: TerminalOutcome::Dropped,
        });
        observer.observe(DuplexRuntimeEvent::UtteranceTerminalOutcome {
            direction: AudioDirection::Speaker,
            utterance_id,
            outcome: TerminalOutcome::Dropped,
        });
        if matches!(
            self.scenario,
            Scenario::ProviderDrop | Scenario::ProviderCancelled
        ) {
            if self.scenario == Scenario::ProviderDrop {
                observer.observe(audio_frame(AudioDirection::Microphone, utterance_id, 0));
            }
            observer.observe(DuplexRuntimeEvent::UtteranceTerminalOutcome {
                direction: AudioDirection::Microphone,
                utterance_id,
                outcome: if self.scenario == Scenario::ProviderDrop {
                    TerminalOutcome::Dropped
                } else {
                    TerminalOutcome::Cancelled
                },
            });
            return;
        }
        if self.scenario == Scenario::AudioWithoutOutcome {
            observer.observe(audio_frame(AudioDirection::Microphone, utterance_id, 0));
            return;
        }
        if self.scenario == Scenario::Recursion {
            for _ in 0..2 {
                observer.observe(DuplexRuntimeEvent::SpeechStarted {
                    direction: AudioDirection::Microphone,
                    utterance_id: Uuid::new_v4(),
                    capture_monotonic_ns: monotonic_ns(),
                });
            }
        }
        if self.scenario != Scenario::NoDebugTextStageEvents {
            observer.observe(DuplexRuntimeEvent::TranscriptFinal {
                direction: AudioDirection::Microphone,
                utterance_id,
            });
            observer.observe(DuplexRuntimeEvent::TranslationFinal {
                direction: AudioDirection::Microphone,
                utterance_id,
            });
        }
        observer.observe(audio_frame(AudioDirection::Microphone, utterance_id, 0));
        observer.observe(DuplexRuntimeEvent::UtteranceTerminalOutcome {
            direction: AudioDirection::Microphone,
            utterance_id,
            outcome: TerminalOutcome::Completed,
        });
        observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
            direction: AudioDirection::Microphone,
            utterance_id,
        });
    }

    fn emit_incoming(&self) {
        let observer = self.observer.lock().unwrap().clone().unwrap();
        let utterance_id = Uuid::new_v4();
        *self.incoming_utterance.lock().unwrap() = Some(utterance_id);
        let onset = monotonic_ns();
        observer.observe(DuplexRuntimeEvent::SpeechStarted {
            direction: AudioDirection::Speaker,
            utterance_id,
            capture_monotonic_ns: onset,
        });
        if self.scenario == Scenario::ActiveObserverReset {
            observer.reset_direction(AudioDirection::Speaker);
            self.action("active_observer_reset");
        }
        if self.scenario == Scenario::IncomingProviderDrop {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                observer.observe(DuplexRuntimeEvent::UtteranceTerminalOutcome {
                    direction: AudioDirection::Speaker,
                    utterance_id,
                    outcome: TerminalOutcome::Dropped,
                });
            });
            return;
        }
        if self.scenario == Scenario::Recursion {
            observer.observe(DuplexRuntimeEvent::SpeechStarted {
                direction: AudioDirection::Speaker,
                utterance_id: Uuid::new_v4(),
                capture_monotonic_ns: monotonic_ns(),
            });
        }
        if self.scenario != Scenario::NoDebugTextStageEvents {
            observer.observe(DuplexRuntimeEvent::TranscriptFinal {
                direction: AudioDirection::Speaker,
                utterance_id,
            });
            observer.observe(DuplexRuntimeEvent::TranslationFinal {
                direction: AudioDirection::Speaker,
                utterance_id,
            });
        }
        let frame = if self.scenario == Scenario::IncomingDrainWait {
            audio_frame_now(AudioDirection::Speaker, utterance_id, 0)
        } else {
            audio_frame(AudioDirection::Speaker, utterance_id, 0)
        };
        observer.observe(frame);
        observer.observe(DuplexRuntimeEvent::UtteranceTerminalOutcome {
            direction: AudioDirection::Speaker,
            utterance_id,
            outcome: TerminalOutcome::Completed,
        });
        self.action("incoming_terminal");
        observer.observe(DuplexRuntimeEvent::UtteranceTerminal {
            direction: AudioDirection::Speaker,
            utterance_id,
        });
    }

    fn worker_started(&self) {
        let active = self.workers.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_workers.fetch_max(active, Ordering::AcqRel);
    }
}

struct FakeDuplexFactory {
    shared: Arc<Shared>,
}

impl RoundTripDuplexFactory for FakeDuplexFactory {
    fn start(
        &self,
        _admitted: AdmittedDuplex,
        observer: Arc<dyn DuplexRuntimeObserver>,
        deadline: tokio::time::Instant,
    ) -> translator_daemon::DuplexStartResult {
        self.shared.start_deadlines.lock().unwrap().push(deadline);
        self.shared.action("duplex_start");
        *self.shared.observer.lock().unwrap() = Some(observer);
        let runtime = Box::new(FakeDuplex {
            shared: Arc::clone(&self.shared),
        });
        if self.shared.scenario == Scenario::StartCleanupPending {
            Err(translator_daemon::DuplexStartFailure::cleanup_pending(
                translator_daemon::DuplexRuntimeError::StartFailed,
                runtime,
            ))
        } else {
            Ok(runtime)
        }
    }
}

struct FakeDuplex {
    shared: Arc<Shared>,
}

impl ActiveDuplexRuntime for FakeDuplex {
    fn stop(
        &mut self,
        deadline: tokio::time::Instant,
    ) -> Result<(), translator_daemon::DuplexRuntimeError> {
        self.shared.cleanup_deadlines.lock().unwrap().push(deadline);
        self.shared.action("duplex_stop");
        if self.shared.scenario == Scenario::CleanupObserverReset {
            let observer = self.shared.observer.lock().unwrap().clone().unwrap();
            observer.reset_direction(AudioDirection::Microphone);
            observer.reset_direction(AudioDirection::Speaker);
            self.shared.action("cleanup_observer_reset");
        }
        if self
            .shared
            .duplex_cleanup_failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            return Err(translator_daemon::DuplexRuntimeError::StopFailed);
        }
        Ok(())
    }
}

impl Drop for FakeDuplex {
    fn drop(&mut self) {
        self.shared.action("duplex_drop");
    }
}

struct FakeAudioFactory {
    shared: Arc<Shared>,
}

impl RoundTripAudioWorkerFactory for FakeAudioFactory {
    fn create(
        &self,
        _session_id: Uuid,
        _physical_sink: &str,
    ) -> Result<Box<dyn RoundTripAudioWorker>, RoundTripProcessError> {
        self.shared.worker_started();
        self.shared.action("audio_create");
        if matches!(
            self.shared.scenario,
            Scenario::StartupAudioFactoryExpiry | Scenario::ActiveAudioFactoryExpiry
        ) {
            let until = if self.shared.scenario == Scenario::StartupAudioFactoryExpiry {
                self.shared.startup_admission.lock().unwrap().unwrap()
            } else {
                Instant::now() + Duration::from_millis(50)
            };
            std::thread::sleep(
                until.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
            );
            self.shared.action("audio_factory_returned");
        }
        Ok(Box::new(FakeAudio {
            shared: Arc::clone(&self.shared),
        }))
    }
}

struct FakeAudio {
    shared: Arc<Shared>,
}

impl Drop for FakeAudio {
    fn drop(&mut self) {
        self.shared.workers.fetch_sub(1, Ordering::AcqRel);
        self.shared.action("audio_drop");
    }
}

impl RoundTripAudioWorker for FakeAudio {
    fn capture_english_utterance<'a>(
        &'a mut self,
        outgoing_terminal: &'a mut watch::Receiver<bool>,
        stop: &'a mut watch::Receiver<bool>,
    ) -> RoundTripWorkerFuture<'a, Vec<PcmFrame>> {
        self.shared.action("capture_entered");
        Box::pin(async move {
            self.shared.action("capture");
            match self.shared.scenario {
                Scenario::Timeout => future::pending().await,
                Scenario::Stop => {
                    if !*stop.borrow() {
                        let _ = stop.changed().await;
                    }
                    Err(RoundTripProcessError::Stopped)
                }
                _ => {
                    self.shared.emit_outgoing();
                    if !*outgoing_terminal.borrow() {
                        outgoing_terminal
                            .changed()
                            .await
                            .map_err(|_| RoundTripProcessError::Progress)?;
                    }
                    assert!(*outgoing_terminal.borrow());
                    self.shared.action("outgoing_terminal");
                    self.shared.action("tap_drain");
                    Ok(self.shared.expected_frames.clone())
                }
            }
        })
    }

    fn monitor_english<'a>(
        &'a mut self,
        frames: &'a [PcmFrame],
        _stop: &'a mut watch::Receiver<bool>,
    ) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            assert_eq!(frames, self.shared.expected_frames);
            self.shared.action("monitor_start");
            self.shared.action("monitor_finish");
            Ok(())
        })
    }

    fn spawn_virtual_peer(&mut self) -> Result<ProcessIdentity, RoundTripProcessError> {
        self.shared.action("peer_spawn");
        self.shared.peer_alive.store(true, Ordering::Release);
        ProcessIdentity::inspect(std::process::id()).ok_or(RoundTripProcessError::Audio)
    }

    fn write_virtual_peer_frame<'a>(
        &'a mut self,
        frame: &'a PcmFrame,
    ) -> RoundTripWorkerFuture<'a, (u64, StreamPcmFormat, usize, [u8; 32])> {
        Box::pin(async move {
            assert!(
                self.shared.scenario != Scenario::PanicAfterRoute,
                "injected lifecycle panic"
            );
            self.shared.action("peer_write");
            if matches!(
                self.shared.scenario,
                Scenario::FailAfterRoute | Scenario::SlowCleanupAfterRoute
            ) {
                return Err(RoundTripProcessError::Audio);
            }
            self.shared
                .reinjected_frames
                .lock()
                .unwrap()
                .push(frame.clone());
            let bytes_written = if self.shared.scenario == Scenario::DroppedReceipt {
                0
            } else {
                frame.pcm().len()
            };
            let mut hash: [u8; 32] = Sha256::digest(frame.pcm()).into();
            if self.shared.scenario == Scenario::CorruptReceipt {
                hash[0] ^= 0xff;
            }
            Ok((frame.sequence(), frame.format(), bytes_written, hash))
        })
    }

    fn finish_virtual_peer<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            self.shared.action("peer_finish");
            if self.shared.scenario != Scenario::PeerPersists {
                self.shared.peer_alive.store(false, Ordering::Release);
            }
            self.shared.emit_incoming();
            Ok(())
        })
    }

    fn stop_writes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            self.shared.action("stop_writes");
            assert!(
                self.shared
                    .cleanup_panics
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| remaining
                        .checked_sub(1))
                    .is_err(),
                "injected cleanup panic"
            );
            if self
                .shared
                .cleanup_failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                return Err(RoundTripProcessError::Audio);
            }
            if self.shared.scenario == Scenario::SlowCleanupAfterRoute {
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            Ok(())
        })
    }

    fn finish_processes<'a>(&'a mut self) -> RoundTripWorkerFuture<'a, ()> {
        Box::pin(async move {
            self.shared.action("finish_processes");
            self.shared.peer_alive.store(false, Ordering::Release);
            Ok(())
        })
    }

    fn clear_sensitive(&mut self) {
        self.shared.action("clear_sensitive");
        assert!(
            self.shared.scenario != Scenario::PanicClearSensitive,
            "injected sensitive cleanup panic"
        );
    }
}

struct FakeRouteFactory {
    shared: Arc<Shared>,
}

impl VirtualPeerRouteControllerFactory for FakeRouteFactory {
    fn create(&self) -> Box<dyn VirtualPeerRouteController> {
        let id = Uuid::new_v4();
        self.shared
            .route_events
            .lock()
            .unwrap()
            .push((id, "create"));
        if matches!(
            self.shared.scenario,
            Scenario::StartupFactoryExpiry | Scenario::ActiveFactoryExpiry
        ) {
            let until = if self.shared.scenario == Scenario::StartupFactoryExpiry {
                self.shared.startup_admission.lock().unwrap().unwrap()
            } else {
                Instant::now() + Duration::from_millis(50)
            };
            std::thread::sleep(
                until.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
            );
            self.shared.action("route_factory_returned");
        }
        Box::new(FakeRoute {
            shared: Arc::clone(&self.shared),
            id,
            acquired: false,
        })
    }
}

struct FakeRoute {
    shared: Arc<Shared>,
    id: Uuid,
    acquired: bool,
}

impl FakeRoute {
    fn record(&self, event: &'static str) {
        self.shared.action(event);
        self.shared
            .route_events
            .lock()
            .unwrap()
            .push((self.id, event));
    }
}

impl Drop for FakeRoute {
    fn drop(&mut self) {
        self.record("route_drop");
    }
}

impl VirtualPeerRouteController for FakeRoute {
    fn route(
        &mut self,
        session_id: Uuid,
        process: ProcessIdentity,
        _expected_target: &str,
    ) -> Result<VirtualPeerCapability, RoundTripProcessError> {
        if self.shared.scenario == Scenario::RouteFailureBeforeAcquisition {
            return Err(RoundTripProcessError::Route);
        }
        self.acquired = true;
        self.shared
            .route_effect_pending
            .store(true, Ordering::SeqCst);
        self.record("route");
        if self.shared.scenario == Scenario::PartialRouteFailure {
            return Err(RoundTripProcessError::Route);
        }
        let session_id = if self.shared.scenario == Scenario::ForgedCapability {
            Uuid::new_v4()
        } else {
            session_id
        };
        let process = if self.shared.scenario == Scenario::StaleCapability {
            ProcessIdentity {
                start_time_ticks: process.start_time_ticks.saturating_add(1),
                ..process
            }
        } else {
            process
        };
        Ok(VirtualPeerCapability {
            session_id,
            stream_id: 41,
            object_serial: 42,
            process,
            process_binary: "pacat".to_owned(),
        })
    }

    fn validate(
        &mut self,
        _capability: &VirtualPeerCapability,
        _expected_target: &str,
    ) -> Result<(), RoundTripProcessError> {
        self.shared.action("route_validate");
        Ok(())
    }

    fn restore(&mut self) -> Result<(), RoundTripProcessError> {
        if matches!(
            self.shared.scenario,
            Scenario::StartupFactoryExpiry
                | Scenario::ActiveFactoryExpiry
                | Scenario::StartupAudioFactoryExpiry
                | Scenario::ActiveAudioFactoryExpiry
        ) {
            self.record("route_restore_called");
        }
        if !self.acquired {
            return Ok(());
        }
        self.record("route_restore");
        if self
            .shared
            .route_restore_failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok()
        {
            return Err(RoundTripProcessError::Route);
        }
        self.shared
            .route_effect_pending
            .store(false, Ordering::SeqCst);
        self.record("route_restored");
        Ok(())
    }

    fn ensure_absent(&mut self) -> Result<(), RoundTripProcessError> {
        if matches!(
            self.shared.scenario,
            Scenario::StartupFactoryExpiry
                | Scenario::ActiveFactoryExpiry
                | Scenario::StartupAudioFactoryExpiry
                | Scenario::ActiveAudioFactoryExpiry
        ) {
            self.record("route_absent_called");
        }
        if !self.acquired {
            return Ok(());
        }
        self.record("route_absent");
        if self.shared.peer_alive.load(Ordering::Acquire) {
            Err(RoundTripProcessError::Route)
        } else {
            self.record("route_absence_confirmed");
            Ok(())
        }
    }
}

#[test]
fn route_owner_survives_peer_absence_until_restoration() {
    let shared = Shared::new(Scenario::Happy);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));
    RoundTripController::start(&controller).unwrap();
    wait_for_gate(&gate, AudioOperationState::Idle);
    controller.shutdown().unwrap();
    assert_eq!(
        store.snapshot().self_test.status.checkpoint,
        Some(RoundTripCheckpoint::Completed)
    );
    let events = shared.actions.lock().unwrap();
    assert_before(&events, "route_absence_confirmed", "route_restored");
    assert_before(&events, "route_restored", "route_drop");
    assert!(!shared.route_effect_pending.load(Ordering::SeqCst));
}

#[test]
fn observer_reset_during_active_incoming_fails_run_and_preserves_cleanup() {
    assert_observer_reset_cleanup(Scenario::ActiveObserverReset, RoundTripCheckpoint::Failed);
}

#[test]
fn observer_reset_during_frozen_cleanup_preserves_completed_run() {
    assert_observer_reset_cleanup(
        Scenario::CleanupObserverReset,
        RoundTripCheckpoint::Completed,
    );
}

fn assert_observer_reset_cleanup(scenario: Scenario, expected: RoundTripCheckpoint) {
    let shared = Shared::new(scenario);
    let (store, gate, controller) = controller(shared.clone(), Duration::from_secs(2));
    let started = RoundTripController::start(&controller);
    let limit = Instant::now() + Duration::from_secs(2);
    while gate.state() != AudioOperationState::Idle && Instant::now() < limit {
        std::thread::yield_now();
    }
    let auto_cleaned = gate.state() == AudioOperationState::Idle;
    let stopped = RoundTripController::stop(&controller);
    controller.shutdown().unwrap();
    let status = store.snapshot().self_test.status;
    assert!(started.is_ok() && stopped.is_ok());
    assert!(
        auto_cleaned,
        "reset must finish through normal retained-owner cleanup"
    );
    assert_eq!(status.checkpoint, Some(expected));
    assert_eq!(
        *shared.reinjected_frames.lock().unwrap(),
        shared.expected_frames
    );
    assert_eq!(shared.workers.load(Ordering::Acquire), 0);
    assert!(!shared.route_effect_pending.load(Ordering::Acquire));
    let actions = shared.actions.lock().unwrap();
    assert!(
        actions.contains(&if scenario == Scenario::ActiveObserverReset {
            "active_observer_reset"
        } else {
            "cleanup_observer_reset"
        })
    );
    for action in [
        "audio_create",
        "audio_drop",
        "duplex_start",
        "duplex_stop",
        "duplex_drop",
    ] {
        assert_eq!(actions.iter().filter(|value| **value == action).count(), 1);
    }
    assert_before(&actions, "route_restored", "route_drop");
    assert_before(&actions, "finish_processes", "duplex_stop");
}

#[test]
fn route_restore_failure_retains_owner_after_successful_absence() {
    assert_route_owner_retry(Scenario::FailAfterRoute);
}

#[test]
fn route_partial_error_retains_owner_without_a_returned_capability() {
    assert_route_owner_retry(Scenario::PartialRouteFailure);
}

fn assert_route_owner_retry(scenario: Scenario) {
    let shared = Shared::new(scenario);
    shared.route_restore_failures.store(1, Ordering::SeqCst);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));
    let session_id = RoundTripController::start(&controller)
        .unwrap()
        .status
        .session_id
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !store.snapshot().self_test.status.cleanup_pending
        && gate.state() != AudioOperationState::Idle
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(2));
    }
    let before = store.snapshot();
    let gate_before = gate.state();
    let events_before = shared.actions.lock().unwrap().clone();
    let overlap = RoundTripController::start(&controller);
    RoundTripController::stop(&controller).unwrap();
    controller.shutdown().unwrap();
    assert!(before.self_test.status.cleanup_pending);
    assert_eq!(
        gate_before,
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert!(overlap.is_err());
    assert!(!events_before.contains(&"route_drop"));
    let events = shared.actions.lock().unwrap();
    for phase in [
        "stop_writes",
        "finish_processes",
        "duplex_stop",
        "route_absence_confirmed",
    ] {
        assert_eq!(
            events_before
                .iter()
                .filter(|value| **value == phase)
                .count(),
            1,
            "phase {phase} must already be complete"
        );
        assert_eq!(
            events.iter().filter(|value| **value == phase).count(),
            1,
            "completed phase {phase} repeated"
        );
    }
    assert_eq!(
        events
            .iter()
            .filter(|value| **value == "route_restore")
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|value| **value == "route_drop")
            .count(),
        1
    );
    let identities = shared.route_events.lock().unwrap();
    assert!(identities.iter().all(|event| event.0 == identities[0].0));
    assert!(!shared.route_effect_pending.load(Ordering::SeqCst));
    assert!(!store.snapshot().self_test.status.cleanup_pending);
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn route_cleanup_without_acquisition_has_no_route_effects() {
    let shared = Shared::new(Scenario::RouteFailureBeforeAcquisition);
    let (_store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));
    RoundTripController::start(&controller).unwrap();
    wait_for_gate(&gate, AudioOperationState::Idle);
    controller.shutdown().unwrap();
    let events = shared.route_events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].1, "create");
    assert_eq!(events[1], (events[0].0, "route_drop"));
    assert!(!shared.route_effect_pending.load(Ordering::SeqCst));
}

#[test]
fn happy_path_is_linear_and_reuses_exact_pcm_after_monitor_completion() {
    let shared = Shared::new(Scenario::Happy);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Completed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    let proof = snapshot.self_test.status.exact_pcm.unwrap();
    assert!(proof.exact_match);
    assert_eq!(proof.frame_count, shared.expected_frames.len() as u64);
    assert_eq!(
        *shared.reinjected_frames.lock().unwrap(),
        shared.expected_frames
    );
    assert_eq!(shared.max_workers.load(Ordering::Acquire), 1);
    assert_eq!(snapshot.self_test.status.recursion_count, 0);

    let actions = shared.actions.lock().unwrap();
    assert_before(&actions, "outgoing_terminal", "tap_drain");
    assert_before(&actions, "tap_drain", "monitor_start");
    assert_before(&actions, "monitor_finish", "route");
    assert_before(&actions, "route_validate", "peer_write");
    assert_before(&actions, "peer_finish", "route_absent");
    assert_teardown_order(&actions, false);
}

#[test]
fn audio_frames_advance_privacy_mode_without_text_delta_events() {
    let shared = Shared::new(Scenario::NoDebugTextStageEvents);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Completed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    assert!(!snapshot.debug_text_enabled);
    assert!(snapshot.self_test.status.debug_text.is_none());
    assert!(snapshot.self_test.status.exact_pcm.unwrap().exact_match);
    assert_eq!(snapshot.self_test.status.recursion_count, 0);
}

#[test]
fn audio_without_a_completed_outcome_does_not_invent_final_text_stages() {
    let shared = Shared::new(Scenario::AudioWithoutOutcome);
    let (store, _gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    wait_for_checkpoint(&store, RoundTripCheckpoint::OutgoingVad);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        store.snapshot().self_test.status.checkpoint,
        Some(RoundTripCheckpoint::OutgoingVad)
    );
    RoundTripController::stop(&controller).unwrap();
}

#[test]
fn repeated_outgoing_and_incoming_speech_starts_are_counted_as_recursion() {
    let shared = Shared::new(Scenario::Recursion);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Completed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    assert_eq!(snapshot.self_test.status.recursion_count, 3);
}

#[test]
fn completed_waits_for_incoming_playback_drain_before_teardown() {
    let shared = Shared::new(Scenario::IncomingDrainWait);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    let started = Instant::now();
    RoundTripController::start(&controller).unwrap();
    wait_for_checkpoint(&store, RoundTripCheckpoint::Completed);
    assert!(
        started.elapsed() >= Duration::from_millis(70),
        "completed checkpoint was published before incoming playback had time to drain"
    );
    wait_for_gate(&gate, AudioOperationState::Idle);

    let actions = shared.actions.lock().unwrap();
    assert_before(&actions, "incoming_terminal", "duplex_stop");
}

#[test]
fn hard_timeout_sets_safe_error_and_runs_teardown() {
    let shared = Shared::new(Scenario::Timeout);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_millis(20));

    RoundTripController::start(&controller).unwrap();
    let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Failed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    assert_eq!(
        snapshot.self_test.status.safe_error,
        Some(translator_daemon::RoundTripErrorCode::Timeout)
    );
    assert_teardown_order(&shared.actions.lock().unwrap(), false);
}

#[test]
fn dropped_provider_utterance_fails_fast_without_waiting_for_session_timeout() {
    for scenario in [Scenario::ProviderDrop, Scenario::ProviderCancelled] {
        let shared = Shared::new(scenario);
        let timeout = Duration::from_secs(4);
        let (store, gate, controller) = controller(Arc::clone(&shared), timeout);
        let started = Instant::now();

        RoundTripController::start(&controller).unwrap();
        let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Failed);
        wait_for_gate(&gate, AudioOperationState::Idle);

        assert_eq!(
            snapshot.self_test.status.safe_error,
            Some(translator_daemon::RoundTripErrorCode::RuntimeFailed)
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "provider terminal outcome waited for the session deadline: {:?}",
            started.elapsed()
        );
        let utterance_id = shared.outgoing_utterance.lock().unwrap().unwrap();
        shared.observer.lock().unwrap().clone().unwrap().observe(
            DuplexRuntimeEvent::UtteranceTerminal {
                direction: AudioDirection::Microphone,
                utterance_id,
            },
        );
        assert_eq!(
            store.snapshot().self_test.status.safe_error,
            Some(translator_daemon::RoundTripErrorCode::RuntimeFailed)
        );
        assert_teardown_order(&shared.actions.lock().unwrap(), false);
    }
}

#[test]
fn incoming_provider_drop_wakes_an_active_checkpoint_wait() {
    let shared = Shared::new(Scenario::IncomingProviderDrop);
    let timeout = Duration::from_secs(4);
    let (store, gate, controller) = controller(Arc::clone(&shared), timeout);
    let started = Instant::now();

    RoundTripController::start(&controller).unwrap();
    let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Failed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    assert_eq!(
        snapshot.self_test.status.safe_error,
        Some(translator_daemon::RoundTripErrorCode::RuntimeFailed)
    );
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn late_provider_outcome_cannot_replace_a_completed_checkpoint() {
    let shared = Shared::new(Scenario::Happy);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    wait_for_checkpoint(&store, RoundTripCheckpoint::Completed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    let utterance_id = shared.outgoing_utterance.lock().unwrap().unwrap();
    shared.observer.lock().unwrap().clone().unwrap().observe(
        DuplexRuntimeEvent::UtteranceTerminalOutcome {
            direction: AudioDirection::Microphone,
            utterance_id,
            outcome: TerminalOutcome::Dropped,
        },
    );
    let snapshot = store.snapshot();
    assert_eq!(
        snapshot.self_test.status.checkpoint,
        Some(RoundTripCheckpoint::Completed)
    );
    assert_eq!(snapshot.self_test.status.safe_error, None);
}

#[test]
fn lifecycle_panic_preserves_cleanup_owners_until_retry() {
    assert_panic_cleanup_isolated(
        "lifecycle_panic_preserves_cleanup_owners_until_retry",
        Scenario::PanicAfterRoute,
    );
}

#[test]
fn cleanup_panic_preserves_cleanup_owners_until_retry() {
    assert_panic_cleanup_isolated(
        "cleanup_panic_preserves_cleanup_owners_until_retry",
        Scenario::Happy,
    );
}

#[test]
fn sensitive_cleanup_panic_preserves_cleanup_owners_until_retry() {
    assert_panic_cleanup_isolated(
        "sensitive_cleanup_panic_preserves_cleanup_owners_until_retry",
        Scenario::PanicClearSensitive,
    );
}

fn assert_panic_cleanup_isolated(test_name: &str, scenario: Scenario) {
    if std::env::var("TRANSLATOR_ROUND_TRIP_PANIC_TEST")
        .ok()
        .as_deref()
        != Some(test_name)
    {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test_name, "--exact", "--nocapture"])
            .env("TRANSLATOR_ROUND_TRIP_PANIC_TEST", test_name)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let timed_out = child.try_wait().unwrap().is_none();
        if timed_out {
            child.kill().unwrap();
        }
        let output = child.wait_with_output().unwrap();
        assert!(
            !timed_out && output.status.success(),
            "isolated ownership probe failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let shared = Shared::new(scenario);
    if scenario == Scenario::Happy {
        shared.cleanup_panics.store(1, Ordering::SeqCst);
    } else {
        shared.cleanup_failures.store(1, Ordering::SeqCst);
    }
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));
    let session_id = RoundTripController::start(&controller)
        .unwrap()
        .status
        .session_id
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !store.snapshot().self_test.status.cleanup_pending {
        assert!(
            Instant::now() < deadline,
            "panic did not retain a cleanup projection"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let before = store.snapshot();
    let held_gate = gate.state();
    let held_workers = shared.workers.load(Ordering::SeqCst);
    let actions_before = shared.actions.lock().unwrap().clone();
    let state = RoundTripController::stop(&controller).unwrap();
    controller.shutdown().unwrap();
    assert_eq!(
        before.self_test.status.checkpoint,
        Some(RoundTripCheckpoint::Failed)
    );
    assert_eq!(
        before.self_test.status.safe_error,
        Some(translator_daemon::RoundTripErrorCode::RuntimeFailed)
    );
    assert_eq!(
        held_gate,
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert_eq!(held_workers, 1);
    assert_eq!(state.status.checkpoint, Some(RoundTripCheckpoint::Failed));
    assert!(!state.status.cleanup_pending);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(shared.workers.load(Ordering::SeqCst), 0);
    assert_eq!(shared.max_workers.load(Ordering::SeqCst), 1);
    assert!(!actions_before.contains(&"audio_drop"));
    assert_eq!(
        shared
            .actions
            .lock()
            .unwrap()
            .iter()
            .filter(|action| **action == "audio_drop")
            .count(),
        1
    );
}

#[test]
fn failed_inner_start_transfers_cleanup_into_the_same_outer_owner() {
    let shared = Shared::new(Scenario::StartCleanupPending);
    shared.duplex_cleanup_failures.store(1, Ordering::SeqCst);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));
    let session_id = RoundTripController::start(&controller)
        .unwrap()
        .status
        .session_id
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !store.snapshot().self_test.status.cleanup_pending {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let gate_before = gate.state();
    let actions_before = shared.actions.lock().unwrap().clone();
    let overlap = RoundTripController::start(&controller).unwrap_err();
    let cleaned = RoundTripController::stop(&controller).unwrap();
    controller.shutdown().unwrap();
    assert_eq!(
        gate_before,
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert_eq!(overlap.code, "self_test_already_running");
    assert!(!actions_before.contains(&"duplex_drop"));
    assert!(!actions_before.contains(&"audio_create"));
    assert_eq!(cleaned.status.checkpoint, Some(RoundTripCheckpoint::Failed));
    assert_eq!(gate.state(), AudioOperationState::Idle);
    let actions = shared.actions.lock().unwrap();
    assert_eq!(
        actions
            .iter()
            .filter(|action| **action == "duplex_start")
            .count(),
        1
    );
    assert_eq!(
        actions
            .iter()
            .filter(|action| **action == "duplex_stop")
            .count(),
        2
    );
    assert_eq!(
        actions
            .iter()
            .filter(|action| **action == "duplex_drop")
            .count(),
        1
    );
}

#[test]
fn failed_cleanup_keeps_exact_worker_and_lease_until_successful_retry() {
    let shared = Shared::new(Scenario::Stop);
    shared.cleanup_failures.store(1, Ordering::SeqCst);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));
    let started = RoundTripController::start(&controller).unwrap();
    let session_id = started.status.session_id.unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !shared.actions.lock().unwrap().contains(&"capture") {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let first = RoundTripController::stop(&controller);
    let retained_gate = gate.state();
    let retained_workers = shared.workers.load(Ordering::SeqCst);
    let status = serde_json::to_value(store.snapshot().self_test.status).unwrap();
    let overlap_rejected = RoundTripController::start(&controller).is_err();
    let retry = RoundTripController::stop(&controller);

    assert_eq!(first.unwrap_err().code, "self_test_stop_failed");
    assert_eq!(
        retained_gate,
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert_eq!(
        retained_workers, 1,
        "failed cleanup must retain the same audio owner"
    );
    assert_eq!(status["cleanup_pending"], true);
    assert!(overlap_rejected);
    let state = retry.expect("Stop must retry retained cleanup, not replay cached failure");
    assert_eq!(state.status.checkpoint, Some(RoundTripCheckpoint::Failed));
    assert_eq!(
        serde_json::to_value(state.status).unwrap()["cleanup_pending"],
        false
    );
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(shared.workers.load(Ordering::SeqCst), 0);
    assert_eq!(shared.max_workers.load(Ordering::SeqCst), 1);
    assert!(RoundTripController::stop(&controller).is_ok());
    controller.shutdown().unwrap();
    let actions = shared.actions.lock().unwrap();
    assert_eq!(
        actions
            .iter()
            .filter(|action| **action == "stop_writes")
            .count(),
        2
    );
    assert_eq!(
        actions
            .iter()
            .filter(|action| **action == "finish_processes")
            .count(),
        1
    );
    assert_eq!(
        actions
            .iter()
            .filter(|action| **action == "duplex_stop")
            .count(),
        1
    );
}

#[test]
fn async_cleanup_deadline_retains_worker_until_retry_without_repeating_route_restore() {
    let shared = Shared::new(Scenario::SlowCleanupAfterRoute);
    let timeout = Duration::from_millis(100);
    let (store, gate, controller) = controller(Arc::clone(&shared), timeout);
    RoundTripController::start(&controller).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !store.snapshot().self_test.status.cleanup_pending {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let snapshot = store.snapshot();
    let retained_gate = gate.state();
    let retained_worker = shared.workers.load(Ordering::Acquire);
    let actions_before_retry = shared.actions.lock().unwrap().clone();
    let stopped = RoundTripController::stop(&controller).unwrap();

    assert_eq!(
        snapshot.self_test.status.safe_error,
        Some(translator_daemon::RoundTripErrorCode::RuntimeFailed)
    );
    assert!(matches!(
        retained_gate,
        AudioOperationState::HumanRoundTrip { .. }
    ));
    assert_eq!(retained_worker, 1);
    assert!(actions_before_retry.contains(&"clear_sensitive"));
    assert!(!actions_before_retry.contains(&"audio_drop"));
    assert_eq!(stopped.status.checkpoint, Some(RoundTripCheckpoint::Failed));
    assert!(!stopped.status.cleanup_pending);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    let actions = shared.actions.lock().unwrap();
    assert!(actions.contains(&"stop_writes"));
    assert!(actions.contains(&"clear_sensitive"));
    assert!(actions.contains(&"audio_drop"));
    assert_eq!(
        actions
            .iter()
            .filter(|action| **action == "route_restore")
            .count(),
        1,
        "a confirmed route restoration must not be repeated by cleanup retry"
    );
    drop(actions);
    controller.shutdown().unwrap();
}

#[test]
fn autonomous_cleanup_failure_never_publishes_completed_and_retry_preserves_failure() {
    let shared = Shared::new(Scenario::Happy);
    shared.cleanup_failures.store(1, Ordering::SeqCst);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Failed);
    assert!(snapshot.self_test.status.cleanup_pending);
    assert!(RoundTripController::start(&controller).is_err());
    let cleaned = RoundTripController::stop(&controller).unwrap();
    assert_eq!(cleaned.status.checkpoint, Some(RoundTripCheckpoint::Failed));
    assert!(!cleaned.status.cleanup_pending);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert!(shared.actions.lock().unwrap().contains(&"stop_writes"));
    controller.shutdown().unwrap();
}

#[test]
fn explicit_stop_is_idempotently_torn_down_without_a_second_worker() {
    let shared = Shared::new(Scenario::Stop);
    let (_store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    assert!(RoundTripController::start(&controller).is_err());
    let stopped = RoundTripController::stop(&controller).unwrap();

    assert_eq!(
        stopped.status.checkpoint,
        Some(RoundTripCheckpoint::Stopped)
    );
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert_eq!(shared.max_workers.load(Ordering::Acquire), 1);
    assert_teardown_order(&shared.actions.lock().unwrap(), false);
}

#[test]
fn forged_or_stale_capability_is_rejected_before_reinjection_and_restored() {
    for scenario in [Scenario::ForgedCapability, Scenario::StaleCapability] {
        let shared = Shared::new(scenario);
        let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

        RoundTripController::start(&controller).unwrap();
        let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Failed);
        wait_for_gate(&gate, AudioOperationState::Idle);

        assert_eq!(
            snapshot.self_test.status.safe_error,
            Some(translator_daemon::RoundTripErrorCode::RuntimeFailed)
        );
        let actions = shared.actions.lock().unwrap();
        assert!(!actions.contains(&"peer_write"));
        assert!(actions.contains(&"route_restore"));
        assert_teardown_order(&actions, true);
    }
}

#[test]
fn failure_after_route_restores_route_before_releasing_worker() {
    let shared = Shared::new(Scenario::FailAfterRoute);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    wait_for_checkpoint(&store, RoundTripCheckpoint::Failed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    let actions = shared.actions.lock().unwrap();
    assert_before(&actions, "route_validate", "peer_write");
    assert_before(&actions, "route_restore", "stop_writes");
    assert_teardown_order(&actions, true);
}

#[test]
fn tap_keeps_second_speech_segment_after_more_than_300ms_pause() {
    let shared = Shared::new(Scenario::Happy);
    let expected = shared.expected_frames.clone();
    assert!(
        expected[3..19]
            .iter()
            .all(|frame| { frame.pcm().chunks_exact(2).all(|sample| sample == [0, 0]) })
    );
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Completed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    assert_eq!(
        snapshot.self_test.status.exact_pcm.unwrap().frame_count,
        expected.len() as u64
    );
    assert_eq!(*shared.reinjected_frames.lock().unwrap(), expected);
}

#[test]
fn dropped_or_corrupt_write_receipt_rejects_exact_pcm_proof() {
    for scenario in [Scenario::DroppedReceipt, Scenario::CorruptReceipt] {
        let shared = Shared::new(scenario);
        let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

        RoundTripController::start(&controller).unwrap();
        let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Failed);
        wait_for_gate(&gate, AudioOperationState::Idle);

        assert_eq!(
            snapshot.self_test.status.safe_error,
            Some(translator_daemon::RoundTripErrorCode::RuntimeFailed)
        );
        assert!(snapshot.self_test.status.exact_pcm.is_none());
        assert_before(
            &shared.actions.lock().unwrap(),
            "route_restore",
            "stop_writes",
        );
    }
}

#[test]
fn success_requires_exact_peer_absence_after_eof() {
    let shared = Shared::new(Scenario::PeerPersists);
    let (store, gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    let snapshot = wait_for_checkpoint(&store, RoundTripCheckpoint::Failed);
    wait_for_gate(&gate, AudioOperationState::Idle);

    assert_eq!(
        snapshot.self_test.status.safe_error,
        Some(translator_daemon::RoundTripErrorCode::RuntimeFailed)
    );
    let actions = shared.actions.lock().unwrap();
    assert_before(&actions, "peer_finish", "route_absent");
    assert_before(&actions, "route_absent", "route_restore");
    assert_before(&actions, "route_restore", "stop_writes");
}

struct DeadlineRunner {
    inner: RoundTripProcessRunner,
    deadline: Option<Instant>,
    shared: Arc<Shared>,
}

impl translator_daemon::RoundTripRunner for DeadlineRunner {
    fn start(
        &self,
        admitted: AdmittedDuplex,
        session_id: Uuid,
        progress: translator_daemon::RoundTripProgress,
        deadline: Instant,
    ) -> Result<
        Box<dyn translator_daemon::ActiveRoundTripRuntime>,
        translator_daemon::RoundTripRuntimeError,
    > {
        let deadline = self.deadline.unwrap_or_else(|| {
            let deadline = if self.shared.short_startup_budget.load(Ordering::Acquire) {
                deadline.min(Instant::now() + Duration::from_millis(250))
            } else {
                deadline
            };
            *self.shared.startup_admission.lock().unwrap() = Some(deadline);
            deadline
        });
        translator_daemon::RoundTripRunner::start(
            &self.inner,
            admitted,
            session_id,
            progress,
            deadline,
        )
    }
}

#[test]
fn expired_process_start_has_no_factory_or_audio_effects() {
    let positive = Shared::new(Scenario::Stop);
    let (_, _, positive_controller) = controller(positive.clone(), Duration::from_secs(300));
    let positive_start = RoundTripController::start(&positive_controller);
    let positive_wait = Instant::now() + Duration::from_secs(2);
    while positive.start_deadlines.lock().unwrap().is_empty() && Instant::now() < positive_wait {
        std::thread::yield_now();
    }
    let positive_stop = RoundTripController::stop(&positive_controller);
    positive_controller.shutdown().unwrap();
    let shared = Shared::new(Scenario::Stop);
    let (store, gate, controller) = controller_with_deadline(
        shared.clone(),
        Duration::from_secs(300),
        Some(Instant::now() - Duration::from_millis(1)),
    );
    let started = RoundTripController::start(&controller);
    let stopped = RoundTripController::stop(&controller);
    controller.shutdown().unwrap();
    assert!(positive_start.is_ok() && positive_stop.is_ok());
    assert_eq!(positive.start_deadlines.lock().unwrap().len(), 1);
    assert_eq!(positive.max_workers.load(Ordering::Acquire), 1);
    assert!(started.is_err());
    assert!(stopped.is_ok());
    assert!(shared.actions.lock().unwrap().is_empty());
    assert!(shared.start_deadlines.lock().unwrap().is_empty());
    assert_eq!(shared.max_workers.load(Ordering::Acquire), 0);
    assert_eq!(gate.state(), AudioOperationState::Idle);
    assert!(!store.snapshot().self_test.status.cleanup_pending);
}

#[test]
fn factory_crossing_startup_deadline_cannot_start_the_next_factory() {
    assert_factory_deadline(Scenario::StartupFactoryExpiry);
}

#[test]
fn factory_crossing_active_deadline_cannot_start_the_next_factory() {
    assert_factory_deadline(Scenario::ActiveFactoryExpiry);
}

#[test]
fn audio_factory_crossing_startup_deadline_cannot_start_capture() {
    assert_audio_factory_deadline(Scenario::StartupAudioFactoryExpiry);
}

#[test]
fn audio_factory_crossing_active_deadline_cannot_start_capture() {
    assert_audio_factory_deadline(Scenario::ActiveAudioFactoryExpiry);
}

fn assert_audio_factory_deadline(scenario: Scenario) {
    let shared = Shared::new(scenario);
    shared.short_startup_budget.store(
        scenario == Scenario::StartupAudioFactoryExpiry,
        Ordering::Release,
    );
    let timeout = if scenario == Scenario::ActiveAudioFactoryExpiry {
        Duration::from_millis(20)
    } else {
        Duration::from_secs(300)
    };
    let (_, _, controller) = controller(shared.clone(), timeout);
    let started = RoundTripController::start(&controller);
    let limit = Instant::now() + Duration::from_secs(2);
    while !shared
        .actions
        .lock()
        .unwrap()
        .contains(&"audio_factory_returned")
        && Instant::now() < limit
    {
        std::thread::yield_now();
    }
    let startup_still_live = scenario != Scenario::ActiveAudioFactoryExpiry
        || Instant::now() < shared.startup_admission.lock().unwrap().unwrap();
    let mut stopped = RoundTripController::stop(&controller);
    if stopped.is_err() {
        stopped = RoundTripController::stop(&controller);
    }
    controller.shutdown().unwrap();
    assert!(started.is_ok() && stopped.is_ok());
    assert!(startup_still_live);
    let actions = shared.actions.lock().unwrap();
    assert!(actions.contains(&"audio_factory_returned"));
    assert!(
        !actions.contains(&"capture_entered"),
        "capture must not be admitted after audio factory exhausted the inherited deadline"
    );
    assert!(shared.reinjected_frames.lock().unwrap().is_empty());
    for action in [
        "duplex_start",
        "audio_create",
        "stop_writes",
        "finish_processes",
        "audio_drop",
        "duplex_stop",
        "duplex_drop",
    ] {
        assert_eq!(
            actions.iter().filter(|event| **event == action).count(),
            1,
            "single created owner must complete {action}"
        );
    }
    assert_eq!(shared.workers.load(Ordering::Acquire), 0);
    let events = shared.route_events.lock().unwrap();
    let created: Vec<_> = events
        .iter()
        .filter(|(_, event)| *event == "create")
        .collect();
    assert_eq!(created.len(), 1);
    let owner = created[0].0;
    for event in ["route_restore_called", "route_absent_called", "route_drop"] {
        assert!(
            events.contains(&(owner, event)),
            "same created route must complete {event}"
        );
    }
}

fn assert_factory_deadline(scenario: Scenario) {
    let shared = Shared::new(scenario);
    shared.short_startup_budget.store(
        scenario == Scenario::StartupFactoryExpiry,
        Ordering::Release,
    );
    let timeout = if scenario == Scenario::ActiveFactoryExpiry {
        Duration::from_millis(20)
    } else {
        Duration::from_secs(300)
    };
    let (_, _, controller) = controller(shared.clone(), timeout);
    let started = RoundTripController::start(&controller);
    let limit = Instant::now() + Duration::from_secs(2);
    while !shared
        .actions
        .lock()
        .unwrap()
        .contains(&"route_factory_returned")
        && Instant::now() < limit
    {
        std::thread::yield_now();
    }
    let startup_still_live = scenario != Scenario::ActiveFactoryExpiry
        || Instant::now() < shared.startup_admission.lock().unwrap().unwrap();
    let mut stopped = RoundTripController::stop(&controller);
    if stopped.is_err() {
        stopped = RoundTripController::stop(&controller);
    }
    controller.shutdown().unwrap();
    assert!(started.is_ok() && stopped.is_ok());
    assert!(
        startup_still_live,
        "active-deadline case must retain an unexpired startup deadline"
    );
    assert!(
        shared.start_deadlines.lock().unwrap().is_empty(),
        "native factory must not begin after the inherited deadline"
    );
    assert_eq!(shared.max_workers.load(Ordering::Acquire), 0);
    let events = shared.route_events.lock().unwrap();
    let owner = events
        .iter()
        .find(|(_, event)| *event == "create")
        .unwrap()
        .0;
    for event in ["route_restore_called", "route_absent_called", "route_drop"] {
        assert!(
            events.contains(&(owner, event)),
            "created route owner must be retained through {event}"
        );
    }
}

#[test]
fn exact_short_startup_deadline_does_not_truncate_active_round_trip() {
    let shared = Shared::new(Scenario::Stop);
    shared.short_startup_budget.store(true, Ordering::Release);
    let (_, gate, controller) =
        controller_with_deadline(shared.clone(), Duration::from_secs(300), None);
    RoundTripController::start(&controller).unwrap();
    let handshake_limit = Instant::now() + Duration::from_secs(2);
    while shared.start_deadlines.lock().unwrap().is_empty() && Instant::now() < handshake_limit {
        std::thread::yield_now();
    }
    let start_deadline = *shared.startup_admission.lock().unwrap();
    if let Some(deadline) = start_deadline {
        std::thread::sleep(
            deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(20),
        );
    }
    let active = gate.state();
    let live_workers = shared.workers.load(Ordering::Acquire);
    let before_stop = Instant::now();
    let stopped = RoundTripController::stop(&controller);
    let after_stop = Instant::now();
    controller.shutdown().unwrap();
    assert!(matches!(active, AudioOperationState::HumanRoundTrip { .. }));
    assert_eq!(live_workers, 1);
    assert!(stopped.is_ok());
    assert_eq!(
        *shared.start_deadlines.lock().unwrap(),
        vec![tokio::time::Instant::from_std(start_deadline.unwrap())]
    );
    let stops = shared.cleanup_deadlines.lock().unwrap();
    assert_eq!(stops.len(), 1);
    assert!(stops[0].into_std() >= before_stop + translator_daemon::RUNTIME_CLEANUP_BUDGET);
    assert!(stops[0].into_std() <= after_stop + translator_daemon::RUNTIME_CLEANUP_BUDGET);
}

#[test]
fn autonomous_cleanup_admits_eight_seconds_instead_of_remaining_session_minutes() {
    let shared = Shared::new(Scenario::Happy);
    let (_, gate, controller) = controller(shared.clone(), Duration::from_secs(300));
    let before = Instant::now();
    RoundTripController::start(&controller).unwrap();
    wait_for_gate(&gate, AudioOperationState::Idle);
    let after = Instant::now();
    controller.shutdown().unwrap();
    let stops = shared.cleanup_deadlines.lock().unwrap();
    assert_eq!(stops.len(), 1);
    assert!(stops[0].into_std() >= before + translator_daemon::RUNTIME_CLEANUP_BUDGET);
    assert!(stops[0].into_std() <= after + translator_daemon::RUNTIME_CLEANUP_BUDGET);
}

fn controller(
    shared: Arc<Shared>,
    timeout: Duration,
) -> (RuntimeStore, AudioOperationGate, RoundTripRuntimeHandle) {
    controller_with_deadline(shared, timeout, None)
}

fn controller_with_deadline(
    shared: Arc<Shared>,
    timeout: Duration,
    deadline: Option<Instant>,
) -> (RuntimeStore, AudioOperationGate, RoundTripRuntimeHandle) {
    let inner = RoundTripProcessRunner::with_components(
        Arc::new(FakeDuplexFactory {
            shared: Arc::clone(&shared),
        }),
        Arc::new(FakeAudioFactory {
            shared: Arc::clone(&shared),
        }),
        Arc::new(FakeRouteFactory {
            shared: shared.clone(),
        }),
        timeout,
    );
    let runner = Arc::new(DeadlineRunner {
        inner,
        deadline,
        shared,
    });
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let controller = RoundTripRuntimeHandle::try_new(
        store.clone(),
        runner,
        gate.clone(),
        audio_facts::fixture(store.snapshot()),
    )
    .unwrap();
    (store, gate, controller)
}

fn ready_store() -> RuntimeStore {
    let store = RuntimeStore::default();
    store.set_audio_graph(AudioGraphState {
        health: GraphHealth::Ready,
        endpoints: Vec::new(),
        owned_module_ids: Vec::new(),
        safe_error: None,
    });
    store.set_devices(DeviceState {
        source: selection(1, "alsa_input.physical", "Mic", "Mic"),
        sink: selection(2, "alsa_output.headphones", "Headphones", "Headphones"),
        acoustic: AcousticSafety {
            mode: OutputMode::Headphones,
            aec_capability: AecCapability::Unavailable,
            full_duplex_allowed: true,
            warning: None,
        },
    });
    store.set_routes(RoutingState {
        candidates: Vec::new(),
        source_outputs: Vec::new(),
        conflicting_stream_ids: Vec::new(),
        active_route: None,
        resolution: RouteResolution::NoCandidate,
    });
    store
}

fn selection(id: u32, name: &str, description: &str, port_type: &str) -> DeviceSelectionState {
    let device = PhysicalDevice {
        id,
        name: name.to_owned(),
        description: description.to_owned(),
        active_port: Some("active".to_owned()),
        active_port_type: Some(port_type.to_owned()),
        available: true,
    };
    DeviceSelectionState {
        health: DeviceHealth::Available,
        pinned_name: Some(name.to_owned()),
        current_default: Some(name.to_owned()),
        pending_default: None,
        selected: Some(device),
    }
}

fn frames() -> Vec<PcmFrame> {
    let format = StreamPcmFormat::provider_default();
    let mut samples = vec![1_u8, 2, 3];
    samples.extend(std::iter::repeat_n(0, 16));
    samples.extend([4, 5, 6]);
    samples
        .into_iter()
        .enumerate()
        .map(|(sequence, sample)| {
            PcmFrame::try_new(
                sequence as u64,
                sequence as u64 * 20_000_000,
                format,
                vec![sample; format.frame_bytes()],
            )
            .unwrap()
        })
        .collect()
}

fn audio_frame(direction: AudioDirection, utterance_id: Uuid, sequence: u64) -> DuplexRuntimeEvent {
    let now = monotonic_ns();
    DuplexRuntimeEvent::AudioFrame {
        direction,
        utterance_id,
        sequence,
        provider_monotonic_ns: now,
        observed_monotonic_ns: now.saturating_sub(1_000_000_000),
        queue_lag_ms: 0,
    }
}

fn audio_frame_now(
    direction: AudioDirection,
    utterance_id: Uuid,
    sequence: u64,
) -> DuplexRuntimeEvent {
    let now = monotonic_ns();
    DuplexRuntimeEvent::AudioFrame {
        direction,
        utterance_id,
        sequence,
        provider_monotonic_ns: now,
        observed_monotonic_ns: now,
        queue_lag_ms: 0,
    }
}

fn wait_for_checkpoint(store: &RuntimeStore, expected: RoundTripCheckpoint) -> RuntimeSnapshot {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let snapshot = store.snapshot();
        if snapshot.self_test.status.checkpoint == Some(expected) {
            return snapshot;
        }
        assert!(
            Instant::now() < deadline,
            "checkpoint {expected:?} timed out"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn wait_for_gate(gate: &AudioOperationGate, expected: AudioOperationState) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while gate.state() != expected {
        assert!(Instant::now() < deadline, "audio gate did not become idle");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn assert_teardown_order(actions: &[&'static str], routed: bool) {
    assert_before(actions, "stop_writes", "finish_processes");
    assert_before(actions, "finish_processes", "duplex_stop");
    if routed {
        assert_before(actions, "route_restore", "stop_writes");
    }
    assert_before(actions, "clear_sensitive", "stop_writes");
    assert_before(actions, "clear_sensitive", "audio_drop");
}

fn assert_before(actions: &[&'static str], first: &'static str, second: &'static str) {
    let first_index = actions.iter().position(|action| *action == first).unwrap();
    let second_index = actions.iter().position(|action| *action == second).unwrap();
    assert!(
        first_index < second_index,
        "{first} must precede {second}: {actions:?}"
    );
}

fn monotonic_ns() -> u64 {
    let time = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    u64::try_from(time.tv_sec).unwrap_or(0) * 1_000_000_000
        + u64::try_from(time.tv_nsec).unwrap_or(0)
}
