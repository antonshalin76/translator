//! User-session daemon runtime and local control API.

mod api;
mod audio_operation_gate;
mod debug;
mod direction_session;
mod latency;
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
mod translation_runtime;

pub use api::{
    ApiControllers, ApiLimits, AudioMixController, ControlFailure, ListenAddressError,
    ManualRouteController, RoundTripController, TranslationController, build_router,
    build_router_with_controllers, build_router_with_manual_routes, validate_listen_address,
};
pub use audio_operation_gate::{
    AudioOperationAdmissionError, AudioOperationGate, AudioOperationLease, AudioOperationState,
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
    ActiveRoundTripRuntime, RoundTripProgress, RoundTripRunner, RoundTripRuntimeError,
    RoundTripRuntimeHandle,
};
pub(crate) use runtime_state::RuntimeEvent;
pub use runtime_state::{
    AudioMixPatch, AudioMixState, DirectionPatch, DirectionState, LatencyPolicyPatch,
    ProviderPatch, RoundTripSelfTestState, RuntimeMutationError, RuntimeSnapshot, RuntimeStore,
    VoiceProfilePatch,
};
pub use secure_state::{ControlToken, RuntimeLease, SecureRuntimeError, SecureRuntimeErrorCode};
pub use sidecar_runtime::{
    QuarantinedStaleSocket, StaleSocketError, VerifiedStaleSocket, remove_stale_sidecar_socket,
};
pub use sidecar_supervisor::{
    CLOSE_ACK_TIMEOUT, ChildState, CloseOutcome, MAX_START_ATTEMPTS, PROBE_TIMEOUT, SidecarLaunch,
    SidecarRuntime, SidecarStatus, SidecarSupervisor, SupervisorError,
};
pub use translation_runtime::{
    ActiveDuplexRuntime, DuplexAudioTargets, DuplexRunner, DuplexRuntimeError, DuplexRuntimeEvent,
    DuplexRuntimeHandle, DuplexRuntimeObserver, ProcessDuplexConfig, ProcessDuplexRunner,
    RuntimeLatencyObserver, TASK7_BRIDGE_SCHEMA_VERSION, Task7BridgeEvent, Task7BridgeFailureStage,
    resolve_duplex_audio_targets,
};
