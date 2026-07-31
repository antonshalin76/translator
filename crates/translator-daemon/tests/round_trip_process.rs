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
    AcousticSafety, AecCapability, AudioGraphState, DeviceHealth, DeviceSelectionState,
    DeviceState, GraphHealth, OutputMode, PcmFrame, PhysicalDevice, ProcessIdentity,
    RouteResolution, RoutingState, StreamPcmFormat, VirtualPeerCapability,
};
use translator_core::AudioDirection;
use translator_daemon::{
    ActiveDuplexRuntime, AudioOperationGate, AudioOperationState, DuplexRuntimeEvent,
    DuplexRuntimeObserver, RoundTripAudioWorker, RoundTripAudioWorkerFactory, RoundTripCheckpoint,
    RoundTripController, RoundTripDuplexFactory, RoundTripProcessError, RoundTripProcessRunner,
    RoundTripRuntimeHandle, RoundTripWorkerFuture, RuntimeSnapshot, RuntimeStore,
    SafeProviderErrorCode, TerminalOutcome, VirtualPeerRouteController,
    VirtualPeerRouteControllerFactory,
};
use uuid::Uuid;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scenario {
    Happy,
    Recursion,
    Timeout,
    SlowCleanupAfterRoute,
    TeardownFailure,
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
        _snapshot: RuntimeSnapshot,
        observer: Arc<dyn DuplexRuntimeObserver>,
    ) -> Result<Box<dyn ActiveDuplexRuntime>, RoundTripProcessError> {
        self.shared.action("duplex_start");
        *self.shared.observer.lock().unwrap() = Some(observer);
        Ok(Box::new(FakeDuplex {
            shared: Arc::clone(&self.shared),
        }))
    }
}

struct FakeDuplex {
    shared: Arc<Shared>,
}

impl ActiveDuplexRuntime for FakeDuplex {
    fn stop(&mut self) -> Result<(), translator_daemon::DuplexRuntimeError> {
        self.shared.action("duplex_stop");
        Ok(())
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
            if self.shared.scenario == Scenario::SlowCleanupAfterRoute {
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            if self.shared.scenario == Scenario::TeardownFailure {
                return Err(RoundTripProcessError::Audio);
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
    }
}

struct FakeRouteFactory {
    shared: Arc<Shared>,
}

impl VirtualPeerRouteControllerFactory for FakeRouteFactory {
    fn create(&self) -> Box<dyn VirtualPeerRouteController> {
        Box::new(FakeRoute {
            shared: Arc::clone(&self.shared),
        })
    }
}

struct FakeRoute {
    shared: Arc<Shared>,
}

impl VirtualPeerRouteController for FakeRoute {
    fn route(
        &mut self,
        session_id: Uuid,
        process: ProcessIdentity,
        _expected_target: &str,
    ) -> Result<VirtualPeerCapability, RoundTripProcessError> {
        self.shared.action("route");
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

    fn restore(
        &mut self,
        _capability: &VirtualPeerCapability,
    ) -> Result<(), RoundTripProcessError> {
        self.shared.action("route_restore");
        Ok(())
    }

    fn ensure_absent(
        &mut self,
        _capability: &VirtualPeerCapability,
    ) -> Result<(), RoundTripProcessError> {
        self.shared.action("route_absent");
        if self.shared.peer_alive.load(Ordering::Acquire) {
            Err(RoundTripProcessError::Route)
        } else {
            Ok(())
        }
    }
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
fn hard_deadline_includes_forced_cleanup_budget() {
    let shared = Shared::new(Scenario::SlowCleanupAfterRoute);
    let timeout = Duration::from_millis(100);
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
        started.elapsed() <= timeout + Duration::from_millis(50),
        "whole lifecycle exceeded configured timeout: {:?}",
        started.elapsed()
    );
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
        "force release must not retry a synchronous route restore after the deadline"
    );
    drop(actions);
    let stop_error = RoundTripController::stop(&controller).unwrap_err();
    assert_eq!(stop_error.code, "self_test_stop_failed");
}

#[test]
fn teardown_failure_is_reported_as_stop_failed_and_runtime_is_retained() {
    let shared = Shared::new(Scenario::TeardownFailure);
    let (store, _gate, controller) = controller(Arc::clone(&shared), Duration::from_secs(2));

    RoundTripController::start(&controller).unwrap();
    wait_for_checkpoint(&store, RoundTripCheckpoint::Completed);

    let stop_error = RoundTripController::stop(&controller).unwrap_err();

    assert_eq!(stop_error.code, "self_test_stop_failed");
    assert!(RoundTripController::start(&controller).is_err());
    assert!(shared.actions.lock().unwrap().contains(&"stop_writes"));
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

fn controller(
    shared: Arc<Shared>,
    timeout: Duration,
) -> (RuntimeStore, AudioOperationGate, RoundTripRuntimeHandle) {
    let runner = Arc::new(RoundTripProcessRunner::with_components(
        Arc::new(FakeDuplexFactory {
            shared: Arc::clone(&shared),
        }),
        Arc::new(FakeAudioFactory {
            shared: Arc::clone(&shared),
        }),
        Arc::new(FakeRouteFactory { shared }),
        timeout,
    ));
    let store = ready_store();
    let gate = AudioOperationGate::new();
    let controller = RoundTripRuntimeHandle::new(store.clone(), runner, gate.clone());
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
        assert_before(actions, "route_restore", "clear_sensitive");
    } else {
        assert_before(actions, "duplex_stop", "clear_sensitive");
    }
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
