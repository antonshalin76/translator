//! User-session daemon runtime and local control API.

mod acoustic_admission;
mod aec_admission;
mod aec_runtime_observer;
mod aec_validation;
mod aec_validation_control;
mod api;
mod audio_mix_application;
mod audio_operation_gate;
mod bounded_http;
mod control_application;
mod debug;
mod direction_session;
mod latency;
mod panic_report;
mod process_sidecar;
mod provider_audio_watchdog;
mod queues;
mod round_trip;
mod round_trip_process;
mod round_trip_runtime;
mod runtime_state;
mod secure_state;
mod sidecar_runtime;
mod sidecar_supervisor;
mod task7_bridge;
mod translation_runtime;

pub use panic_report::install_private_panic_hook;
pub use task7_bridge::run_task7_bridge;

pub use acoustic_admission::{
    AcousticSafety, AcousticWarning, AdmittedDuplex, DeviceState, FactsError, RuntimeFacts,
    RuntimeFactsSource,
};
pub use aec_admission::{AecProofInspector, AecRuntimeAuthority, AecStartReservation};
pub use aec_runtime_observer::{
    AecObservationCollectorError, AecObserverGenerationActivation, AecRuntimeObserver,
    DuplexRuntimeObserverFanout,
};
pub use aec_validation::{
    AEC_PROOF_LIFETIME_NS, AecAdmissionGuard, AecAdmissionReservation, AecCalibrationChallenge,
    AecCalibrationCoordinator, AecCoordinatorError, AecProofBinding, AecProofStatus,
};
pub use aec_validation_control::{
    AEC_CALIBRATION_BUDGET, AecCalibrationCancellation, AecCalibrationControlError,
    AecCalibrationControlStatus, AecCalibrationController, AecCalibrationEngine,
    AecCalibrationEngineError, AecCalibrationFuture, AecCalibrationPublication,
    AecCalibrationRequest, AecCleanupFuture,
};

pub use api::{
    ApiControllers, ApiLimits, AudioMixController, ControlFailure, ListenAddressError,
    ManualRouteController, RoundTripController, build_router, build_router_with_controllers,
    build_router_with_manual_routes, validate_listen_address,
};
pub use audio_mix_application::{AudioMixApplication, TranslationMixMode};
pub use audio_operation_gate::{
    AudioOperationAdmissionError, AudioOperationGate, AudioOperationLease, AudioOperationState,
};
pub use bounded_http::serve_control;
pub use control_application::{
    ControlApplication, ControlCommand, RuntimeMaintenance, RuntimeSupervisor,
};
pub use debug::{
    DebugCaptureLimits, DebugCaptureSession, DebugCaptureStopReason, DebugCaptureStore,
    DebugTextBuffer, DebugTextEvent, DebugTextStatus, FreeSpaceProbe,
};
pub use direction_session::{
    DirectionEffect, DirectionRuntimeConfig, DirectionSession, DirectionSessionError,
    DirectionWatchdogEffect, SafeProviderErrorCode, TerminalOutcome,
};
pub use latency::{DuplexLatencyPolicy, LatencySample, LatencyTransition, LatencyTransitionReason};
pub use process_sidecar::{GRACEFUL_SHUTDOWN_TIMEOUT, ProcessSidecarRuntime};
pub use provider_audio_watchdog::{
    CANCEL_FINAL_TIMEOUT, INTER_AUDIO_DELTA_TIMEOUT, ProviderAudioWatchdog,
    ProviderStreamCoordinator, ProviderStreamCoordinatorError, WatchdogAction, WatchdogError,
};
pub use queues::{DaemonQueues, QueueConsumeResult, QueueKind, QueuePushResult, QueueState};
pub use round_trip::{
    ExactPcmEvidence, ExactPcmEvidenceError, ExactPcmProof, RoundTripCheckpoint,
    RoundTripDebugText, RoundTripErrorCode, RoundTripLatency, RoundTripPreconditions,
    RoundTripSelfTest, RoundTripStatus,
};
pub use round_trip_process::{
    RoundTripAudioWorker, RoundTripAudioWorkerFactory, RoundTripDuplexFactory,
    RoundTripProcessError, RoundTripProcessRunner, RoundTripWorkerFuture,
    VirtualPeerRouteController, VirtualPeerRouteControllerFactory,
};
pub use round_trip_runtime::{
    ActiveRoundTripRuntime, RoundTripOwnerShutdownError, RoundTripOwnerStartError,
    RoundTripProgress, RoundTripRunner, RoundTripRuntimeError, RoundTripRuntimeHandle,
    RoundTripTerminal,
};
pub(crate) use runtime_state::RuntimeEvent;
pub use runtime_state::{
    AudioMixKnowledge, AudioMixPatch, AudioMixState, DirectionPatch, DirectionRuntimeFailure,
    DirectionRuntimeStatus, DirectionState, LatencyPolicyPatch, ProviderPatch,
    RoundTripSelfTestState, RuntimeMutationError, RuntimeSnapshot, RuntimeStatus, RuntimeStore,
    VoiceProfilePatch,
};
pub use secure_state::{ControlToken, RuntimeLease, SecureRuntimeError, SecureRuntimeErrorCode};
pub use sidecar_runtime::{
    QuarantinedStaleSocket, StaleSocketError, VerifiedStaleSocket, remove_stale_sidecar_socket,
};
pub use sidecar_supervisor::{
    CLOSE_ACK_TIMEOUT, ChildState, CloseOutcome, GenerationRetirement, MAX_START_ATTEMPTS,
    PROBE_TIMEOUT, SidecarLaunch, SidecarRuntime, SidecarStatus, SidecarSupervisor,
    SupervisorError,
};
pub use translation_runtime::{
    ActiveDuplexRuntime, CompletedCaptureFrame, DIRECTION_CLEANUP_BUDGET, DuplexCompletionObserver,
    DuplexRunner, DuplexRuntimeError, DuplexRuntimeEvent, DuplexRuntimeObserver,
    DuplexStartFailure, DuplexStartResult, ProcessDuplexConfig, ProcessDuplexRunner,
    ProviderEffectOrigin, RUNTIME_CLEANUP_BUDGET, RuntimeLatencyObserver,
    TASK7_BRIDGE_SCHEMA_VERSION, Task7BridgeEvent, Task7BridgeFailureStage,
};
