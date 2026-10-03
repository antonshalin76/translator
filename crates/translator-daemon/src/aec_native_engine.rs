use std::{
    collections::BTreeMap,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use sha2::{Digest, Sha256};
use tokio::{sync::Mutex as AsyncMutex, task::JoinHandle, time::Instant};
use translator_audio::{AecDeviceMetadata, NativeAecIdentity, NativeMeasurementReceiver};
use translator_core::AudioDirection;

use crate::{
    ActiveDuplexRuntime, AdmittedDuplex, AecCalibrationEngine, AecCalibrationEngineError,
    AecCalibrationFuture, AecCalibrationPublication, AecCalibrationRequest, AecCleanupFuture,
    AecCoordinatorError, AecProofBinding, AecProofInspector, DirectionRuntimeFailure,
    DirectionRuntimeStatus, DuplexCompletionObserver, DuplexRunner, DuplexRuntimeError,
    DuplexRuntimeObserver, DuplexRuntimeObserverFanout, DuplexStartFailure, DuplexStartResult,
    PlaybackMixAuthority, ProcessDuplexConfig, ProcessDuplexRunner, RuntimeSnapshot, RuntimeStore,
    aec_backend_session::{AecBackendAttempt, AecBackendSessionOwner, AecBackendSessionStatus},
    translation_runtime::NativeDuplexIo,
};

pub struct NativeAecPairFacts {
    pub audio_server_id: String,
    pub source_name: String,
    pub sink_name: String,
    pub source_hardware_id: String,
    pub sink_hardware_id: String,
}

pub trait NativeAecEnvironment: Send + Sync {
    fn inspect_pair(
        &self,
        identity: &NativeAecIdentity,
        deadline: std::time::Instant,
    ) -> Result<NativeAecPairFacts, AecCalibrationEngineError>;
    fn quarantine(&self) -> Result<(), AecCalibrationEngineError>;
}

pub struct NativeAecPositiveFixture {
    pcm: Vec<u8>,
    sha256: String,
}

impl NativeAecPositiveFixture {
    pub fn read(path: &Path, sha256: &str) -> Result<Self, AecCalibrationEngineError> {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(
                (rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK).bits() as i32,
            )
            .open(path)
            .map_err(|_| error("aec_positive_unavailable"))?;
        let metadata = file
            .metadata()
            .map_err(|_| error("aec_positive_unavailable"))?;
        if !metadata.is_file()
            || metadata.len() == 0
            || metadata.len() > 1_048_576
            || metadata.len() % 640 != 0
        {
            return Err(error("aec_positive_invalid"));
        }
        let mut pcm = Vec::new();
        file.by_ref()
            .take(1_048_577)
            .read_to_end(&mut pcm)
            .map_err(|_| error("aec_positive_unavailable"))?;
        if pcm.len() as u64 != metadata.len() {
            return Err(error("aec_positive_invalid"));
        }
        if digest(&pcm) != sha256
            || !pcm
                .chunks_exact(2)
                .any(|v| i16::from_le_bytes([v[0], v[1]]) != 0)
        {
            return Err(error("aec_positive_invalid"));
        }
        Ok(Self {
            pcm,
            sha256: sha256.into(),
        })
    }
}

struct NativeSession {
    guard: AecBackendAttempt,
    io: Arc<NativeDuplexIo>,
    measurement: Option<NativeMeasurementReceiver>,
    binding: AecProofBinding,
    snapshot: RuntimeSnapshot,
}

#[derive(Default)]
struct WarmRuntime {
    startup: Option<JoinHandle<DuplexStartResult>>,
    active: Option<Box<dyn ActiveDuplexRuntime>>,
    retirement: Option<JoinHandle<(Box<dyn ActiveDuplexRuntime>, bool)>>,
    retirement_panicked: bool,
}

struct NativeEngineInner {
    backend: AecBackendSessionOwner,
    session: Mutex<Option<NativeSession>>,
    runtime: Arc<AsyncMutex<WarmRuntime>>,
    config: ProcessDuplexConfig,
    store: RuntimeStore,
    environment: Arc<dyn NativeAecEnvironment>,
    mix: Arc<dyn PlaybackMixAuthority>,
    observer: Arc<dyn DuplexRuntimeObserver>,
    fixture: NativeAecPositiveFixture,
    completion: Arc<RetainedCompletion>,
}

#[derive(Default)]
struct RetainedCompletion {
    attached: Mutex<Option<(u64, Arc<dyn DuplexCompletionObserver>)>>,
    directions: Mutex<Vec<RetainedDirectionStatus>>,
}

type RetainedDirectionStatus = (
    AudioDirection,
    u64,
    DirectionRuntimeStatus,
    Option<DirectionRuntimeFailure>,
);

impl RetainedCompletion {
    fn attach(&self, generation: u64, observer: Arc<dyn DuplexCompletionObserver>) {
        let directions = self
            .directions
            .lock()
            .expect("retained completion mutex poisoned");
        *self
            .attached
            .lock()
            .expect("retained attachment mutex poisoned") = Some((generation, observer.clone()));
        for (direction, epoch, status, failure) in directions.iter() {
            observer.direction_status_changed(generation, *direction, *epoch, *status, *failure);
        }
    }
    fn detach(&self) {
        self.attached
            .lock()
            .expect("retained attachment mutex poisoned")
            .take();
    }
}

impl DuplexCompletionObserver for RetainedCompletion {
    fn cleanup_started(&self, _: u64) {
        if let Some((generation, observer)) = self
            .attached
            .lock()
            .expect("retained attachment mutex poisoned")
            .as_ref()
        {
            observer.cleanup_started(*generation);
        }
    }
    fn completed(&self, _: u64, result: Result<(), DuplexRuntimeError>) {
        if let Some((generation, observer)) = self
            .attached
            .lock()
            .expect("retained attachment mutex poisoned")
            .as_ref()
        {
            observer.completed(*generation, result);
        }
    }
    fn direction_status_changed(
        &self,
        _: u64,
        direction: AudioDirection,
        epoch: u64,
        status: DirectionRuntimeStatus,
        failure: Option<DirectionRuntimeFailure>,
    ) {
        let mut directions = self
            .directions
            .lock()
            .expect("retained completion mutex poisoned");
        directions.retain(|(old, _, _, _)| *old != direction);
        directions.push((direction, epoch, status, failure));
        drop(directions);
        if let Some((generation, observer)) = self
            .attached
            .lock()
            .expect("retained attachment mutex poisoned")
            .as_ref()
        {
            observer.direction_status_changed(*generation, direction, epoch, status, failure);
        }
    }
}

#[derive(Clone)]
pub struct NativeAecCalibrationEngine {
    inner: Arc<NativeEngineInner>,
}

impl NativeAecCalibrationEngine {
    pub fn new(
        config: ProcessDuplexConfig,
        store: RuntimeStore,
        environment: Arc<dyn NativeAecEnvironment>,
        mix: Arc<dyn PlaybackMixAuthority>,
        observer: Arc<dyn DuplexRuntimeObserver>,
        fixture: NativeAecPositiveFixture,
    ) -> Self {
        Self {
            inner: Arc::new(NativeEngineInner {
                backend: AecBackendSessionOwner::new(),
                session: Mutex::new(None),
                runtime: Arc::new(AsyncMutex::new(WarmRuntime::default())),
                config,
                store,
                environment,
                mix,
                observer,
                fixture,
                completion: Arc::new(RetainedCompletion::default()),
            }),
        }
    }

    fn binding(
        &self,
        identity: &NativeAecIdentity,
        snapshot: &RuntimeSnapshot,
        deadline: Instant,
    ) -> Result<AecProofBinding, AecCalibrationEngineError> {
        let facts = self
            .inner
            .environment
            .inspect_pair(identity, deadline.into_std())?;
        Ok(AecProofBinding {
            audio_server_id: facts.audio_server_id,
            source_hardware_id: facts.source_hardware_id,
            sink_hardware_id: facts.sink_hardware_id,
            source_name: facts.source_name,
            sink_name: facts.sink_name,
            source_port: identity.source_port.clone(),
            sink_port: identity.sink_port.clone(),
            source_channel_gains: identity.capture_gains.clone(),
            sink_channel_gains: identity.playback_gains.clone(),
            source_muted: identity.capture_muted,
            sink_muted: identity.playback_muted,
            source_geometry: format!(
                "48000:{}:{}",
                identity.capture_channels, identity.capture_buffer
            ),
            sink_geometry: format!(
                "48000:{}:{}",
                identity.playback_channels, identity.playback_buffer
            ),
            graph: identity.graph.clone(),
            aec_generation: format!("{:?}", identity.graph),
            aec_config_id: identity.control_fingerprint.clone(),
            vad_config_id: vad_fingerprint(),
            provider_config_id: config_fingerprint(snapshot)?,
        })
    }

    async fn calibrate_inner(
        &self,
        request: &AecCalibrationRequest,
    ) -> Result<AecCalibrationPublication, AecCalibrationEngineError> {
        let (io, measurement, snapshot, binding) = {
            let mut state = self
                .inner
                .session
                .lock()
                .map_err(|_| error("aec_custody_unknown"))?;
            let session = state
                .as_mut()
                .ok_or_else(|| error("aec_source_unavailable"))?;
            (
                session.io.clone(),
                session
                    .measurement
                    .take()
                    .ok_or_else(|| error("aec_measurement_consumed"))?,
                session.snapshot.clone(),
                session.binding.clone(),
            )
        };
        let (collector, activation) = crate::AecRuntimeObserver::new(
            uuid::Uuid::new_v4(),
            request.attempt_id,
            request.challenge.challenge_id(),
        );
        collector
            .activate_generation(activation)
            .map_err(|_| error("aec_observer_unavailable"))?;
        let collector = Arc::new(collector);
        let observer = Arc::new(DuplexRuntimeObserverFanout::new(
            self.inner.observer.clone(),
            collector.clone(),
        ));
        let runner = ProcessDuplexRunner::with_observer(self.inner.config.clone(), observer)
            .with_playback_mix_authority(self.inner.mix.clone())
            .with_native(io.clone());
        let startup = async {
            let mut warm = self.inner.runtime.lock().await;
            if warm.startup.is_some() || warm.active.is_some() {
                return Err(error("aec_runtime_busy"));
            }
            let deadline = request.deadline;
            let completion = self.inner.completion.clone();
            warm.startup = Some(tokio::task::spawn_blocking(move || {
                runner.start_native_calibration(&snapshot, completion, deadline)
            }));
            let result = warm
                .startup
                .as_mut()
                .expect("startup is retained before waiting")
                .await;
            warm.startup = None;
            match result {
                Ok(Ok(runtime)) => {
                    warm.active = Some(runtime);
                    Ok(())
                }
                Ok(Err(failure)) => {
                    warm.active = failure.into_parts().1;
                    Err(error("aec_runtime_unavailable"))
                }
                Err(_) => {
                    warm.retirement_panicked = true;
                    Err(error("aec_runtime_unavailable"))
                }
            }
        };
        startup.await?;
        // Model warm-up cannot fill the four-window acquisition queue. Capture
        // remains paused; the actual settling/baseline/ERLE sequence starts now.
        io.handle
            .start_measurement(request.deadline)
            .await
            .map_err(|_| error("aec_measurement_unavailable"))?;
        let measured = measurement
            .collect(request.deadline)
            .await
            .map_err(|_| error("aec_measurement_failed"))?;
        collector
            .start_positive_control(monotonic_ns())
            .map_err(|_| error("aec_observer_unavailable"))?;
        io.allow_positive(true);
        io.resume(request.deadline)
            .await
            .map_err(|_| error("aec_runtime_unavailable"))?;
        io.handle
            .inject_positive(
                &self.inner.fixture.pcm,
                &self.inner.fixture.sha256,
                request.deadline,
            )
            .await
            .map_err(|_| error("aec_positive_failed"))?;
        io.handle
            .positive_drained(request.deadline)
            .await
            .map_err(|_| error("aec_positive_failed"))?;
        loop {
            let progress = io.progress(AudioDirection::Microphone);
            if progress.failed {
                return Err(error("aec_positive_failed"));
            }
            if progress.completed > 0 && progress.drained && progress.last_capture_physical {
                break;
            }
            if Instant::now() >= request.deadline {
                return Err(error("aec_positive_not_drained"));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        io.allow_positive(false);
        let positive_end = monotonic_ns();
        collector
            .complete_positive_control(positive_end)
            .map_err(|_| error("aec_positive_failed"))?;
        let generation = io
            .progress(AudioDirection::Microphone)
            .runtime_generation
            .ok_or_else(|| error("aec_runtime_unavailable"))?;
        let scored_start = monotonic_ns();
        collector
            .start_scored_interval(request.challenge.interval_id(), scored_start, generation)
            .map_err(|_| error("aec_observer_unavailable"))?;
        while collector.scored_frame_count() < translator_audio::AEC_OBSERVATION_FRAME_COUNT {
            if io.progress(AudioDirection::Microphone).failed || Instant::now() >= request.deadline
            {
                return Err(error("aec_observation_failed"));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let observation = collector
            .complete_scored_interval(monotonic_ns())
            .map_err(|_| error("aec_observation_failed"))?;
        io.pause(request.deadline)
            .await
            .map_err(|_| error("aec_runtime_not_drained"))?;
        let identity = io
            .handle
            .check_fresh(request.deadline)
            .map_err(|_| error("aec_source_unavailable"))?;
        if self.binding(&identity, &self.inner.store.snapshot(), request.deadline)? != binding {
            return Err(error("aec_binding_changed"));
        }
        let metadata = AecDeviceMetadata {
            source_name: binding.source_name.clone(),
            sink_name: binding.sink_name.clone(),
            source_geometry: binding.source_geometry.clone(),
            sink_geometry: binding.sink_geometry.clone(),
            sink_port: binding.sink_port.clone(),
            sink_volume_percent: identity.playback_volume_percent,
        };
        let input = measured
            .finish(binding.measurement_binding(), metadata, observation)
            .map_err(|_| error("aec_measurement_failed"))?;
        Ok(AecCalibrationPublication {
            input: input.into(),
            probe_teardown_confirmed: true,
            graph_retained: true,
        })
    }

    async fn cleanup_inner(&self, deadline: Instant) -> bool {
        self.inner.completion.detach();
        let mut warm = self.inner.runtime.lock().await;
        if let Some(task) = warm.startup.as_mut() {
            match tokio::time::timeout_at(deadline, task).await {
                Ok(Ok(Ok(runtime))) => warm.active = Some(runtime),
                Ok(Ok(Err(failure))) => warm.active = failure.into_parts().1,
                Ok(Err(_)) => {
                    warm.startup = None;
                    warm.retirement_panicked = true;
                    return false;
                }
                Err(_) => return false,
            }
            warm.startup = None;
        }
        if warm.retirement_panicked {
            return false;
        }
        if warm.retirement.is_none()
            && let Some(mut runtime) = warm.active.take()
        {
            warm.retirement = Some(tokio::task::spawn_blocking(move || {
                let clean = runtime.stop(deadline).is_ok();
                (runtime, clean)
            }));
        }
        if let Some(task) = warm.retirement.as_mut() {
            match tokio::time::timeout_at(deadline, task).await {
                Ok(Ok((_, true))) => {
                    warm.retirement = None;
                }
                Ok(Ok((runtime, false))) => {
                    warm.retirement = None;
                    warm.active = Some(runtime);
                    return false;
                }
                Ok(Err(_)) => {
                    warm.retirement = None;
                    warm.retirement_panicked = true;
                    return false;
                }
                Err(_) => return false,
            }
        }
        drop(warm);
        let session = self
            .inner
            .session
            .lock()
            .ok()
            .and_then(|mut state| state.take());
        if let Some(session) = session {
            let _terminal = session.guard.cancel().await;
            if self.inner.backend.status().await != AecBackendSessionStatus::Idle {
                *self
                    .inner
                    .session
                    .lock()
                    .expect("native session mutex poisoned") = Some(session);
                return false;
            }
        }
        self.inner.backend.status().await == AecBackendSessionStatus::Idle
    }
}

impl AecCalibrationEngine for NativeAecCalibrationEngine {
    fn inspect_binding(
        &self,
        deadline: Instant,
    ) -> Result<AecProofBinding, AecCalibrationEngineError> {
        if let Some(session) = self
            .inner
            .session
            .lock()
            .map_err(|_| error("aec_custody_unknown"))?
            .as_ref()
        {
            let identity = session
                .io
                .handle
                .check_fresh(deadline)
                .map_err(|_| error("aec_source_unavailable"))?;
            return self.binding(&identity, &self.inner.store.snapshot(), deadline);
        }
        self.inner.environment.quarantine()?;
        let snapshot = self.inner.store.snapshot();
        if ![AudioDirection::Microphone, AudioDirection::Speaker]
            .iter()
            .all(|direction| {
                snapshot
                    .directions
                    .iter()
                    .any(|state| state.direction_id == *direction && state.enabled)
            })
            || snapshot.audio_mix.microphone_original_percent != 0
            || snapshot.audio_mix.speaker_original_percent != 0
        {
            return Err(error("aec_configuration_unavailable"));
        }
        let attempt = tokio::runtime::Handle::current()
            .block_on(tokio::time::timeout_at(
                deadline,
                self.inner.backend.start_retained_native(),
            ))
            .map_err(|_| error("aec_source_unavailable"))?
            .map_err(|_| error("aec_source_unavailable"))?;
        let io = NativeDuplexIo::new(attempt.handle, attempt.capture);
        let result = tokio::runtime::Handle::current().block_on(io.handle.pause_capture(deadline));
        if result.is_err() {
            let _ = tokio::runtime::Handle::current().block_on(attempt.guard.cancel());
            return Err(error("aec_source_unavailable"));
        }
        let identity = io
            .handle
            .check_fresh(deadline)
            .map_err(|_| error("aec_source_unavailable"))?;
        let binding = self.binding(&identity, &snapshot, deadline)?;
        *self
            .inner
            .session
            .lock()
            .map_err(|_| error("aec_custody_unknown"))? = Some(NativeSession {
            guard: attempt.guard,
            io,
            measurement: Some(attempt.measurement),
            binding: binding.clone(),
            snapshot,
        });
        Ok(binding)
    }

    fn calibrate(&self, request: AecCalibrationRequest) -> AecCalibrationFuture {
        let owner = self.clone();
        Box::pin(async move {
            let result = tokio::select! {
                biased;
                _ = request.cancellation.cancelled() => Err(error("aec_cancelled")),
                _ = tokio::time::sleep_until(request.deadline) => Err(error("aec_timed_out")),
                result = owner.calibrate_inner(&request) => result,
            };
            match result {
                Ok(publication) => Ok(publication),
                Err(mut failure) => {
                    failure.cleanup_confirmed = owner
                        .cleanup_inner(Instant::now() + crate::RUNTIME_CLEANUP_BUDGET)
                        .await;
                    Err(failure)
                }
            }
        })
    }

    fn cleanup(&self, deadline: Instant) -> AecCleanupFuture {
        let owner = self.clone();
        Box::pin(async move { owner.cleanup_inner(deadline).await })
    }
}

impl AecProofInspector for NativeAecCalibrationEngine {
    fn inspect_binding(
        &self,
        deadline: std::time::Instant,
    ) -> Result<AecProofBinding, AecCoordinatorError> {
        let state = self
            .inner
            .session
            .lock()
            .map_err(|_| AecCoordinatorError::GraphUnavailable)?;
        let session = state
            .as_ref()
            .ok_or(AecCoordinatorError::GraphUnavailable)?;
        if session.io.progress(AudioDirection::Microphone).failed
            || session
                .io
                .progress(AudioDirection::Microphone)
                .runtime_generation
                .is_none()
        {
            return Err(AecCoordinatorError::GraphUnavailable);
        }
        let warm = self
            .inner
            .runtime
            .try_lock()
            .map_err(|_| AecCoordinatorError::GraphUnavailable)?;
        if warm.active.is_none()
            || warm.startup.is_some()
            || warm.retirement.is_some()
            || warm.retirement_panicked
        {
            return Err(AecCoordinatorError::GraphUnavailable);
        }
        drop(warm);
        let identity = session
            .io
            .handle
            .check_fresh(deadline.into())
            .map_err(|_| AecCoordinatorError::GraphUnavailable)?;
        self.binding(&identity, &self.inner.store.snapshot(), deadline.into())
            .map_err(|_| AecCoordinatorError::BindingChanged)
    }
}

impl DuplexRunner for NativeAecCalibrationEngine {
    fn start(&self, admitted: AdmittedDuplex, deadline: Instant) -> DuplexStartResult {
        if !admitted.aec_reservation().is_some_and(|r| r.native_graph()) {
            return ProcessDuplexRunner::with_observer(
                self.inner.config.clone(),
                self.inner.observer.clone(),
            )
            .with_playback_mix_authority(self.inner.mix.clone())
            .start(admitted, deadline);
        }
        let reservation = admitted
            .aec_reservation()
            .filter(|r| r.native_graph())
            .ok_or_else(|| DuplexStartFailure::rejected(DuplexRuntimeError::StartFailed))?;
        let state = self
            .inner
            .session
            .lock()
            .map_err(|_| DuplexStartFailure::rejected(DuplexRuntimeError::StartFailed))?;
        let session = state
            .as_ref()
            .ok_or_else(|| DuplexStartFailure::rejected(DuplexRuntimeError::StartFailed))?;
        if config_fingerprint(admitted.snapshot()).ok().as_deref()
            != Some(session.binding.provider_config_id.as_str())
            || vad_fingerprint() != session.binding.vad_config_id
        {
            return Err(DuplexStartFailure::rejected(
                DuplexRuntimeError::StartFailed,
            ));
        }
        let io = session.io.clone();
        let reservation = reservation.clone();
        drop(state);
        reservation
            .consume_before_effects(deadline.into_std())
            .and_then(|()| reservation.confirm_before_pcm(deadline.into_std()))
            .map_err(|_| DuplexStartFailure::rejected(DuplexRuntimeError::StartFailed))?;
        tokio::runtime::Handle::current()
            .block_on(io.handle.stop_playback(deadline))
            .map_err(|_| DuplexStartFailure::rejected(DuplexRuntimeError::StartFailed))?;
        io.attach_reservation(reservation);
        tokio::runtime::Handle::current()
            .block_on(io.resume(deadline))
            .map_err(DuplexStartFailure::rejected)?;
        Ok(Box::new(NativeRuntimeAttachment {
            io,
            owner: self.clone(),
            bypass_confirmed: false,
        }))
    }

    fn start_supervised(
        &self,
        admitted: AdmittedDuplex,
        generation: u64,
        completion: Arc<dyn DuplexCompletionObserver>,
        deadline: Instant,
    ) -> DuplexStartResult {
        if !admitted.aec_reservation().is_some_and(|r| r.native_graph()) {
            return ProcessDuplexRunner::with_observer(
                self.inner.config.clone(),
                self.inner.observer.clone(),
            )
            .with_playback_mix_authority(self.inner.mix.clone())
            .start_supervised(admitted, generation, completion, deadline);
        }
        self.inner.completion.attach(generation, completion);
        let result = self.start(admitted, deadline);
        if result.is_err() {
            self.inner.completion.detach();
        }
        result
    }
}

struct NativeRuntimeAttachment {
    io: Arc<NativeDuplexIo>,
    owner: NativeAecCalibrationEngine,
    bypass_confirmed: bool,
}
impl Drop for NativeRuntimeAttachment {
    fn drop(&mut self) {
        if !self.bypass_confirmed {
            self.io.suspend_delivery();
            self.owner.inner.completion.detach();
        }
    }
}
impl ActiveDuplexRuntime for NativeRuntimeAttachment {
    fn stop(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        self.bypass_confirmed = false;
        let runtime = tokio::runtime::Handle::current();
        if runtime
            .block_on(async {
                self.io.pause(deadline).await?;
                self.io.begin_bypass(deadline).await
            })
            .is_ok()
        {
            self.bypass_confirmed = true;
            self.owner.inner.completion.detach();
            return Ok(());
        }
        if runtime.block_on(self.owner.cleanup_inner(deadline)) {
            Ok(())
        } else {
            Err(DuplexRuntimeError::StopFailed)
        }
    }
    fn reap(&mut self, deadline: Instant) -> Result<(), DuplexRuntimeError> {
        self.stop(deadline)
    }
    fn retained_native_bypass(&self) -> bool {
        self.bypass_confirmed
    }
}

fn error(code: &'static str) -> AecCalibrationEngineError {
    AecCalibrationEngineError {
        code,
        cleanup_confirmed: false,
    }
}
fn monotonic_ns() -> u64 {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    (now.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(now.tv_nsec as u64)
}
fn vad_fingerprint() -> String {
    let values: BTreeMap<_, _> = std::env::vars_os()
        .filter(|(name, _)| name.to_string_lossy().starts_with("TRANSLATOR_VAD_"))
        .collect();
    format!("webrtc-vad:{}", digest(format!("{values:?}").as_bytes()))
}
fn config_fingerprint(snapshot: &RuntimeSnapshot) -> Result<String, AecCalibrationEngineError> {
    let configs: Vec<_> = snapshot
        .directions
        .iter()
        .map(|state| {
            (
                state.direction_id,
                state.enabled,
                state.source_language,
                state.target_language,
                &state.voice_profile,
            )
        })
        .collect();
    let modes: Vec<_> = snapshot
        .latency_policy
        .iter()
        .map(|state| (state.direction_id, state.current_mode))
        .collect();
    let environment: BTreeMap<Vec<u8>, Vec<u8>> = std::env::vars_os()
        .filter(|(name, _)| name.to_string_lossy().starts_with("TRANSLATOR_"))
        .map(|(name, value)| {
            (
                name.as_encoded_bytes().to_vec(),
                value.as_encoded_bytes().to_vec(),
            )
        })
        .collect();
    let bytes = serde_json::to_vec(&(
        snapshot.provider_id,
        configs,
        modes,
        snapshot.debug_text_enabled,
        environment,
    ))
    .map_err(|_| error("aec_configuration_unavailable"))?;
    Ok(digest(&bytes))
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn positive_fixture_requires_exact_regular_bounded_non_silent_pcm() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("positive.pcm");
        let pcm = vec![1_u8; 640];
        std::fs::write(&path, &pcm).unwrap();
        let fixture = NativeAecPositiveFixture::read(&path, &digest(&pcm)).unwrap();
        assert_eq!(fixture.pcm, pcm);
        assert!(NativeAecPositiveFixture::read(&path, &"0".repeat(64)).is_err());
        let link = root.path().join("link.pcm");
        symlink(&path, &link).unwrap();
        assert!(NativeAecPositiveFixture::read(&link, &digest(&pcm)).is_err());
        let fifo = root.path().join("fifo.pcm");
        rustix::fs::mknodat(
            rustix::fs::CWD,
            &fifo,
            rustix::fs::FileType::Fifo,
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            0,
        )
        .unwrap();
        assert!(NativeAecPositiveFixture::read(&fifo, &digest(&pcm)).is_err());
        for invalid in [vec![], vec![0; 640], vec![1; 639], vec![1; 1_048_640]] {
            std::fs::write(&path, &invalid).unwrap();
            assert!(NativeAecPositiveFixture::read(&path, &digest(&invalid)).is_err());
        }
        assert!(NativeAecPositiveFixture::read(root.path(), &digest(&pcm)).is_err());
    }
}
