use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use libpulse_binding as pulse;
use pulse::{
    callbacks::ListResult,
    channelmap,
    context::{Context, FlagSet as ContextFlags, State as ContextState},
    def::{BufferAttr, INVALID_INDEX},
    mainloop::standard::{IterateResult, Mainloop},
    proplist::{Proplist, properties},
    sample,
    stream::{FlagSet as StreamFlags, PeekResult, SeekMode, State as StreamState, Stream},
    volume::{ChannelVolumes, Volume},
};
use thiserror::Error;
use uuid::Uuid;

use crate::MIC_OUT_SINK;

pub const MICROPHONE_ORIGINAL_PLAYBACK: &str = "translator-microphone-original";
pub const MICROPHONE_ORIGINAL_CAPTURE: &str = "translator-microphone-original-capture";
pub const SESSION_PROPERTY: &str = "translator.original_microphone_session";

const APPLICATION_NAME: &str = "translator-daemon";
const STARTUP_LIMIT: Duration = Duration::from_secs(2);
const CLEANUP_LIMIT: Duration = Duration::from_secs(1);
const POLL_INTERVAL: Duration = Duration::from_millis(2);
const READY_DISPATCH_LIMIT: usize = 64;
const CAPTURE_BYTES: u32 = 4_800;
const PLAYBACK_BYTES: u32 = 4_800;
const PENDING_BYTES: usize = 9_600;
const FRAME_BYTES: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum OriginalMicrophoneError {
    #[error("Original microphone deadline expired")]
    Deadline,
    #[error("Original microphone worker was cancelled")]
    Cancelled,
    #[error("Original microphone resource custody is unavailable")]
    Custody,
    #[error("Original microphone native stream identity is invalid")]
    Identity,
    #[error("Original microphone native transport failed")]
    Transport,
    #[error("Original microphone buffering exceeded its bound")]
    Buffer,
    #[error("Original microphone cleanup was not confirmed")]
    Cleanup,
}

#[derive(Debug, Default)]
struct Lease {
    alive: AtomicBool,
    cancelled: AtomicBool,
}

impl Lease {
    fn cancel(&self) {
        self.cancel_once();
    }

    fn cancel_once(&self) -> bool {
        let first = !self.cancelled.swap(true, Ordering::AcqRel);
        self.alive.store(false, Ordering::Release);
        first
    }

    fn fail(&self, code: OriginalMicrophoneError, component: &'static str) {
        if self.cancel_once() {
            tracing::warn!(
                event = "original_microphone_forwarding_failed",
                ?code,
                component,
                "Native original microphone forwarding stopped"
            );
        }
    }
}

/// An opaque identity issued only after the retained native worker is ready.
#[derive(Debug, Clone)]
pub struct OriginalMicrophoneRegistration {
    playback_index: u32,
    capture_index: u32,
    source_index: u32,
    sink_index: u32,
    source_name: String,
    session_id: Uuid,
    process_id: u32,
    client_id: u32,
    lease: Arc<Lease>,
}

impl OriginalMicrophoneRegistration {
    pub const fn playback_index(&self) -> u32 {
        self.playback_index
    }

    pub const fn capture_index(&self) -> u32 {
        self.capture_index
    }

    pub const fn source_index(&self) -> u32 {
        self.source_index
    }

    pub const fn sink_index(&self) -> u32 {
        self.sink_index
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    pub const fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub const fn process_id(&self) -> u32 {
        self.process_id
    }

    pub const fn client_id(&self) -> u32 {
        self.client_id
    }

    pub fn is_live(&self) -> bool {
        self.lease.alive.load(Ordering::Acquire) && !self.lease.cancelled.load(Ordering::Acquire)
    }

    pub fn same_session(&self, other: &Self) -> bool {
        self.session_id == other.session_id && Arc::ptr_eq(&self.lease, &other.lease)
    }

    pub(crate) fn cancel(&self) {
        self.lease.cancel();
    }

    #[cfg(test)]
    pub(crate) fn test_invalidate(&self) {
        self.cancel();
    }
}

#[derive(Default)]
struct RegistryState {
    custody: Option<Uuid>,
    registration: Option<OriginalMicrophoneRegistration>,
    retained_worker: Option<JoinHandle<Result<(), OriginalMicrophoneError>>>,
}

#[derive(Clone, Default)]
pub struct OriginalMicrophoneRegistry {
    state: Arc<Mutex<RegistryState>>,
}

impl OriginalMicrophoneRegistry {
    pub fn current(
        &self,
    ) -> Result<Option<OriginalMicrophoneRegistration>, OriginalMicrophoneError> {
        let state = self
            .state
            .lock()
            .map_err(|_| OriginalMicrophoneError::Custody)?;
        match (&state.custody, &state.registration) {
            (None, None) => Ok(None),
            (Some(session), Some(registration))
                if *session == registration.session_id && registration.is_live() =>
            {
                Ok(Some(registration.clone()))
            }
            _ => Err(OriginalMicrophoneError::Custody),
        }
    }

    pub fn cancel_current(&self) {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(registration) = &state.registration {
            registration.cancel();
        }
    }

    fn claim(&self, session: Uuid) -> Result<(), OriginalMicrophoneError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| OriginalMicrophoneError::Custody)?;
        if state.custody.is_some() || state.retained_worker.is_some() {
            return Err(OriginalMicrophoneError::Custody);
        }
        state.custody = Some(session);
        Ok(())
    }

    fn publish(
        &self,
        registration: OriginalMicrophoneRegistration,
    ) -> Result<(), OriginalMicrophoneError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| OriginalMicrophoneError::Custody)?;
        if state.custody != Some(registration.session_id) || state.registration.is_some() {
            return Err(OriginalMicrophoneError::Custody);
        }
        check_cancelled(&registration.lease)?;
        registration.lease.alive.store(true, Ordering::Release);
        state.registration = Some(registration);
        Ok(())
    }

    fn revoke(&self, session: Uuid) {
        if let Ok(state) = self.state.lock()
            && state.custody == Some(session)
            && let Some(registration) = &state.registration
        {
            registration.lease.cancel();
        }
    }

    fn release(&self, session: Uuid) -> Result<(), OriginalMicrophoneError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| OriginalMicrophoneError::Custody)?;
        if state.custody != Some(session) || state.retained_worker.is_some() {
            return Err(OriginalMicrophoneError::Custody);
        }
        if let Some(registration) = &state.registration {
            registration.lease.cancel();
        }
        state.registration = None;
        state.custody = None;
        Ok(())
    }

    fn retain(&self, session: Uuid, handle: JoinHandle<Result<(), OriginalMicrophoneError>>) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.custody == Some(session) {
            state.retained_worker = Some(handle);
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn test_register(
        &self,
        playback_index: u32,
        capture_index: u32,
        source_index: u32,
        sink_index: u32,
        source_name: &str,
        session_id: Uuid,
        process_id: u32,
        client_id: u32,
    ) -> OriginalMicrophoneRegistration {
        self.test_clear();
        self.claim(session_id).unwrap();
        let registration = OriginalMicrophoneRegistration {
            playback_index,
            capture_index,
            source_index,
            sink_index,
            source_name: source_name.to_owned(),
            session_id,
            process_id,
            client_id,
            lease: Arc::new(Lease::default()),
        };
        self.publish(registration.clone()).unwrap();
        registration
    }

    #[cfg(test)]
    pub(crate) fn test_clear(&self) {
        let mut state = self.state.lock().unwrap();
        assert!(state.retained_worker.is_none());
        if let Some(registration) = &state.registration {
            registration.lease.cancel();
        }
        state.registration = None;
        state.custody = None;
    }
}

enum WorkerCommand {
    Verify {
        deadline: Instant,
        reply: SyncSender<Result<(), OriginalMicrophoneError>>,
    },
}

struct Worker {
    session_id: Uuid,
    lease: Arc<Lease>,
    commands: SyncSender<WorkerCommand>,
    handle: Option<JoinHandle<Result<(), OriginalMicrophoneError>>>,
    failed_join: Option<OriginalMicrophoneError>,
}

pub struct PulseOriginalMicrophone {
    registry: OriginalMicrophoneRegistry,
    worker: Option<Worker>,
}

impl PulseOriginalMicrophone {
    pub fn new(registry: OriginalMicrophoneRegistry) -> Self {
        Self {
            registry,
            worker: None,
        }
    }

    pub fn prepare(
        &mut self,
        source: &str,
        source_index: u32,
        sink_index: u32,
        deadline: Instant,
    ) -> Result<(), OriginalMicrophoneError> {
        let deadline = deadline.min(Instant::now() + STARTUP_LIMIT);
        validate_request(source, source_index, sink_index)?;
        check_deadline(deadline)?;
        if self.worker.is_some() {
            return self.verify_until(source, source_index, sink_index, deadline);
        }
        let session_id = Uuid::new_v4();
        self.registry.claim(session_id)?;
        let lease = Arc::new(Lease::default());
        let (commands, command_receiver) = mpsc::sync_channel(1);
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let registry = self.registry.clone();
        let worker_lease = Arc::clone(&lease);
        let source = source.to_owned();
        let handle = thread::Builder::new()
            .name("original-microphone".to_owned())
            .spawn(move || {
                run_worker(
                    registry,
                    worker_lease,
                    session_id,
                    Route {
                        source,
                        source_index,
                        sink_index,
                    },
                    deadline,
                    command_receiver,
                    ready_sender,
                )
            })
            .map_err(|_| {
                let _ = self.registry.release(session_id);
                OriginalMicrophoneError::Transport
            })?;
        self.worker = Some(Worker {
            session_id,
            lease,
            commands,
            handle: Some(handle),
            failed_join: None,
        });
        let result = ready_receiver
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|_| OriginalMicrophoneError::Deadline)
            .and_then(|result| result);
        if result.is_err() {
            self.cancel();
        }
        result
    }

    pub fn verify(
        &self,
        source: &str,
        source_index: u32,
        sink_index: u32,
    ) -> Result<(), OriginalMicrophoneError> {
        self.verify_until(
            source,
            source_index,
            sink_index,
            Instant::now() + STARTUP_LIMIT,
        )
    }

    fn verify_until(
        &self,
        source: &str,
        source_index: u32,
        sink_index: u32,
        deadline: Instant,
    ) -> Result<(), OriginalMicrophoneError> {
        validate_request(source, source_index, sink_index)?;
        check_deadline(deadline)?;
        let registration = self
            .registry
            .current()?
            .ok_or(OriginalMicrophoneError::Identity)?;
        let worker = self
            .worker
            .as_ref()
            .ok_or(OriginalMicrophoneError::Custody)?;
        if registration.source_name != source
            || registration.source_index != source_index
            || registration.sink_index != sink_index
            || registration.session_id != worker.session_id
            || !Arc::ptr_eq(&registration.lease, &worker.lease)
        {
            self.cancel();
            return Err(OriginalMicrophoneError::Identity);
        }
        let (reply, receiver) = mpsc::sync_channel(1);
        let result = worker
            .commands
            .try_send(WorkerCommand::Verify { deadline, reply })
            .map_err(|_| OriginalMicrophoneError::Transport)
            .and_then(|()| {
                receiver
                    .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                    .map_err(|_| OriginalMicrophoneError::Deadline)?
            });
        if result.is_err() {
            self.cancel();
        }
        result
    }

    pub fn stop(&mut self, deadline: Instant) -> Result<(), OriginalMicrophoneError> {
        self.cancel();
        let deadline = deadline.min(Instant::now() + CLEANUP_LIMIT);
        let Some(worker) = self.worker.as_mut() else {
            return match self.registry.current()? {
                None => Ok(()),
                Some(_) => Err(OriginalMicrophoneError::Custody),
            };
        };
        if let Some(error) = worker.failed_join {
            return Err(error);
        }
        let handle = worker
            .handle
            .as_ref()
            .ok_or(OriginalMicrophoneError::Cleanup)?;
        while !handle.is_finished() {
            check_deadline(deadline)?;
            thread::sleep(POLL_INTERVAL.min(deadline.saturating_duration_since(Instant::now())));
        }
        let result = worker
            .handle
            .take()
            .unwrap()
            .join()
            .unwrap_or(Err(OriginalMicrophoneError::Cleanup));
        if let Err(error) = result {
            worker.failed_join = Some(error);
            return Err(error);
        }
        self.registry.release(worker.session_id)?;
        self.worker = None;
        Ok(())
    }

    fn cancel(&self) {
        if let Some(worker) = &self.worker {
            worker.lease.cancel();
            self.registry.revoke(worker.session_id);
        }
    }
}

impl Drop for PulseOriginalMicrophone {
    fn drop(&mut self) {
        if self.stop(Instant::now() + CLEANUP_LIMIT).is_err()
            && let Some(worker) = self.worker.as_mut()
            && let Some(handle) = worker.handle.take()
        {
            self.registry.retain(worker.session_id, handle);
        }
    }
}

struct WorkerLife {
    registry: OriginalMicrophoneRegistry,
    lease: Arc<Lease>,
    session_id: Uuid,
}

impl Drop for WorkerLife {
    fn drop(&mut self) {
        self.lease.cancel();
        self.registry.revoke(self.session_id);
    }
}

struct Route {
    source: String,
    source_index: u32,
    sink_index: u32,
}

fn run_worker(
    registry: OriginalMicrophoneRegistry,
    lease: Arc<Lease>,
    session_id: Uuid,
    route: Route,
    deadline: Instant,
    commands: Receiver<WorkerCommand>,
    ready: SyncSender<Result<(), OriginalMicrophoneError>>,
) -> Result<(), OriginalMicrophoneError> {
    let _life = WorkerLife {
        registry: registry.clone(),
        lease: Arc::clone(&lease),
        session_id,
    };
    if let Err(error) = check_work(&lease, deadline) {
        let _ = ready.send(Err(error));
        return Ok(());
    }
    let mut native = match NativeTransport::new(session_id) {
        Ok(native) => native,
        Err(error) => {
            let _ = ready.send(Err(error));
            return Ok(());
        }
    };
    let mut phase = "startup";
    let outcome = (|| {
        let registration = native.connect(route, session_id, Arc::clone(&lease), deadline)?;
        phase = "publish";
        registry.publish(registration.clone())?;
        #[cfg(test)]
        native.checkpoint(NativePhase::Published, &lease)?;
        ready
            .send(Ok(()))
            .map_err(|_| OriginalMicrophoneError::Cancelled)?;
        loop {
            check_cancelled(&lease)?;
            phase = "forward";
            native.iterate_streams(&registration, true)?;
            match commands.try_recv() {
                Ok(WorkerCommand::Verify { deadline, reply }) => {
                    phase = "verify";
                    let result = native.inspect(&registration, false, deadline);
                    let _ = reply.send(result);
                    result?;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            }
            thread::sleep(POLL_INTERVAL);
        }
    })();
    let cancelled = lease.cancelled.load(Ordering::Acquire);
    lease.cancel();
    registry.revoke(session_id);
    if let Err(error) = outcome {
        if !cancelled && error != OriginalMicrophoneError::Cancelled {
            tracing::warn!(
                event = "original_microphone_forwarding_failed",
                code = ?error,
                component = phase,
                "Native original microphone forwarding stopped"
            );
        }
        let _ = ready.try_send(Err(error));
    }
    native.disconnect(Instant::now() + CLEANUP_LIMIT)
}

struct NativeTransport {
    playback: Option<Stream>,
    capture: Option<Stream>,
    context: Context,
    mainloop: Mainloop,
    pending: VecDeque<u8>,
    streaming_started_at: Instant,
    captured_bytes: u64,
    forwarded_bytes: u64,
    #[cfg(test)]
    cancel_at: Option<NativePhase>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativePhase {
    Context,
    Playback,
    Capture,
    Activating,
    Verified,
    Published,
}

impl NativeTransport {
    fn new(session_id: Uuid) -> Result<Self, OriginalMicrophoneError> {
        let mainloop = Mainloop::new().ok_or(OriginalMicrophoneError::Transport)?;
        let proplist = native_properties(session_id, None)?;
        let mut context = Context::new_with_proplist(&mainloop, APPLICATION_NAME, &proplist)
            .ok_or(OriginalMicrophoneError::Transport)?;
        context
            .connect(None, ContextFlags::NOAUTOSPAWN, None)
            .map_err(|_| OriginalMicrophoneError::Transport)?;
        Ok(Self {
            playback: None,
            capture: None,
            context,
            mainloop,
            pending: VecDeque::with_capacity(PENDING_BYTES),
            streaming_started_at: Instant::now(),
            captured_bytes: 0,
            forwarded_bytes: 0,
            #[cfg(test)]
            cancel_at: None,
        })
    }

    fn connect(
        &mut self,
        route: Route,
        session_id: Uuid,
        lease: Arc<Lease>,
        deadline: Instant,
    ) -> Result<OriginalMicrophoneRegistration, OriginalMicrophoneError> {
        while self.context.get_state() != ContextState::Ready {
            check_work(&lease, deadline)?;
            self.iterate()?;
            thread::sleep(POLL_INTERVAL);
        }
        #[cfg(test)]
        self.checkpoint(NativePhase::Context, &lease)?;
        check_work(&lease, deadline)?;
        let mut playback_properties =
            native_properties(session_id, Some(MICROPHONE_ORIGINAL_PLAYBACK))?;
        self.playback = Some(
            Stream::new_with_proplist(
                &mut self.context,
                MICROPHONE_ORIGINAL_PLAYBACK,
                &sample_spec(),
                Some(&mono_map()),
                &mut playback_properties,
            )
            .ok_or(OriginalMicrophoneError::Transport)?,
        );
        let flags = native_stream_flags();
        let mut volume = ChannelVolumes::default();
        volume.set(1, Volume::MUTED);
        self.playback
            .as_mut()
            .unwrap()
            .connect_playback(
                Some(MIC_OUT_SINK),
                Some(&playback_buffer()),
                flags,
                Some(&volume),
                None,
            )
            .map_err(|_| OriginalMicrophoneError::Transport)?;
        #[cfg(test)]
        self.checkpoint(NativePhase::Playback, &lease)?;
        check_work(&lease, deadline)?;
        let mut capture_properties =
            native_properties(session_id, Some(MICROPHONE_ORIGINAL_CAPTURE))?;
        self.capture = Some(
            Stream::new_with_proplist(
                &mut self.context,
                MICROPHONE_ORIGINAL_CAPTURE,
                &sample_spec(),
                Some(&mono_map()),
                &mut capture_properties,
            )
            .ok_or(OriginalMicrophoneError::Transport)?,
        );
        self.capture
            .as_mut()
            .unwrap()
            .connect_record(Some(&route.source), Some(&capture_buffer()), flags)
            .map_err(|_| OriginalMicrophoneError::Transport)?;
        #[cfg(test)]
        self.checkpoint(NativePhase::Capture, &lease)?;
        while self.playback.as_ref().unwrap().get_state() != StreamState::Ready
            || self.capture.as_ref().unwrap().get_state() != StreamState::Ready
        {
            check_work(&lease, deadline)?;
            if [&self.playback, &self.capture].iter().any(|stream| {
                matches!(
                    stream.as_ref().unwrap().get_state(),
                    StreamState::Failed | StreamState::Terminated
                )
            }) {
                return Err(OriginalMicrophoneError::Transport);
            }
            self.iterate()?;
            thread::sleep(POLL_INTERVAL);
        }
        let playback = self.playback.as_ref().unwrap();
        let capture = self.capture.as_ref().unwrap();
        let registration = OriginalMicrophoneRegistration {
            playback_index: playback
                .get_index()
                .ok_or(OriginalMicrophoneError::Identity)?,
            capture_index: capture
                .get_index()
                .ok_or(OriginalMicrophoneError::Identity)?,
            source_index: capture
                .get_device_index()
                .ok_or(OriginalMicrophoneError::Identity)?,
            sink_index: playback
                .get_device_index()
                .ok_or(OriginalMicrophoneError::Identity)?,
            source_name: capture
                .get_device_name()
                .ok_or(OriginalMicrophoneError::Identity)?
                .into_owned(),
            session_id,
            process_id: std::process::id(),
            client_id: self
                .context
                .get_index()
                .ok_or(OriginalMicrophoneError::Identity)?,
            lease,
        };
        if registration.source_name != route.source
            || registration.source_index != route.source_index
            || registration.sink_index != route.sink_index
        {
            return Err(OriginalMicrophoneError::Identity);
        }
        watch_stream(self.capture.as_mut().unwrap(), &registration.lease);
        watch_stream(self.playback.as_mut().unwrap(), &registration.lease);
        let context_lease = Arc::clone(&registration.lease);
        self.context.set_state_callback(Some(Box::new(move || {
            context_lease.fail(OriginalMicrophoneError::Transport, "context_state")
        })));
        self.inspect(&registration, true, deadline)?;
        self.activate(&registration, deadline)?;
        self.inspect(&registration, true, deadline)?;
        check_work(&registration.lease, deadline)?;
        #[cfg(test)]
        self.checkpoint(NativePhase::Verified, &registration.lease)?;
        self.streaming_started_at = Instant::now();
        self.captured_bytes = 0;
        self.forwarded_bytes = 0;
        Ok(registration)
    }

    fn activate(
        &mut self,
        registration: &OriginalMicrophoneRegistration,
        deadline: Instant,
    ) -> Result<(), OriginalMicrophoneError> {
        check_work(&registration.lease, deadline)?;
        self.check_streams(registration)?;
        let playback = self.playback.as_mut().unwrap();
        let length = playback
            .writable_size()
            .ok_or(OriginalMicrophoneError::Transport)?;
        if length == 0 || length > PLAYBACK_BYTES as usize || length % FRAME_BYTES != 0 {
            return Err(OriginalMicrophoneError::Buffer);
        }
        self.pending.resize(length, 0);
        playback
            .write_copy(self.pending.make_contiguous(), 0, SeekMode::Relative)
            .map_err(|_| OriginalMicrophoneError::Transport)?;
        self.pending.clear();
        // Exhausting creation credit makes the next credit grant proof of consumer progress.
        if playback.writable_size() != Some(0) {
            return Err(OriginalMicrophoneError::Transport);
        }
        let acknowledgement = Rc::new(Cell::new(None));
        let observed = acknowledgement.clone();
        let mut trigger = playback.trigger(Some(Box::new(move |success| {
            observed.set(Some(success));
        })));
        #[cfg(test)]
        if self.cancel_at == Some(NativePhase::Activating) {
            registration.lease.cancel();
        }
        let result = (|| {
            loop {
                let consumer_progress = self
                    .playback
                    .as_ref()
                    .unwrap()
                    .writable_size()
                    .ok_or(OriginalMicrophoneError::Transport)?
                    > 0;
                if activation_ready(
                    &registration.lease,
                    deadline,
                    acknowledgement.get(),
                    consumer_progress,
                )? {
                    break;
                }
                self.iterate_streams(registration, false)?;
                thread::sleep(POLL_INTERVAL);
            }
            self.check_streams(registration)
        })();
        if result.is_err() {
            trigger.cancel();
        }
        result
    }

    #[cfg(test)]
    fn checkpoint(&self, phase: NativePhase, lease: &Lease) -> Result<(), OriginalMicrophoneError> {
        if self.cancel_at == Some(phase) {
            lease.cancel();
        }
        check_cancelled(lease)
    }

    fn iterate(&mut self) -> Result<(), OriginalMicrophoneError> {
        dispatch_ready(|| self.iterate_once())
    }

    fn iterate_once(&mut self) -> Result<u32, OriginalMicrophoneError> {
        let result = self.mainloop.iterate(false);
        if matches!(
            self.context.get_state(),
            ContextState::Failed | ContextState::Terminated
        ) {
            return Err(OriginalMicrophoneError::Transport);
        }
        match result {
            IterateResult::Success(events) => Ok(events),
            _ => Err(OriginalMicrophoneError::Transport),
        }
    }

    fn iterate_streams(
        &mut self,
        registration: &OriginalMicrophoneRegistration,
        ready: bool,
    ) -> Result<usize, OriginalMicrophoneError> {
        let mut captured = 0;
        dispatch_ready(|| {
            let events = self.iterate_once()?;
            self.check_streams(registration)?;
            captured += self.pump(ready, &registration.lease)?;
            Ok(events)
        })?;
        Ok(captured)
    }

    fn check_streams(
        &mut self,
        registration: &OriginalMicrophoneRegistration,
    ) -> Result<(), OriginalMicrophoneError> {
        check_cancelled(&registration.lease)?;
        let capture = self
            .capture
            .as_mut()
            .ok_or(OriginalMicrophoneError::Transport)?;
        let playback = self
            .playback
            .as_mut()
            .ok_or(OriginalMicrophoneError::Transport)?;
        if self.context.get_state() != ContextState::Ready
            || self.context.get_index() != Some(registration.client_id)
            || capture.get_state() != StreamState::Ready
            || playback.get_state() != StreamState::Ready
            || capture.get_index() != Some(registration.capture_index)
            || playback.get_index() != Some(registration.playback_index)
            || capture.get_device_index() != Some(registration.source_index)
            || playback.get_device_index() != Some(registration.sink_index)
            || capture.get_device_name().as_deref() != Some(registration.source_name.as_str())
            || playback.get_device_name().as_deref() != Some(MIC_OUT_SINK)
        {
            return Err(OriginalMicrophoneError::Identity);
        }
        validate_native_formats(
            registration,
            [
                capture.get_sample_spec().copied(),
                playback.get_sample_spec().copied(),
            ],
            [
                capture.get_channel_map().copied(),
                playback.get_channel_map().copied(),
            ],
        )?;
        let capture_attr = capture
            .get_buffer_attr()
            .ok_or(OriginalMicrophoneError::Buffer)?;
        let playback_attr = playback
            .get_buffer_attr()
            .ok_or(OriginalMicrophoneError::Buffer)?;
        validate_buffers(capture_attr, playback_attr)
    }

    fn pump(&mut self, ready: bool, lease: &Lease) -> Result<usize, OriginalMicrophoneError> {
        check_cancelled(lease)?;
        if !ready {
            self.pending.clear();
        }
        let playback = self
            .playback
            .as_mut()
            .ok_or(OriginalMicrophoneError::Transport)?;
        for _ in 0..2 {
            let writable = playback
                .writable_size()
                .ok_or(OriginalMicrophoneError::Transport)?;
            let contiguous = self.pending.as_slices().0;
            let length = writable.min(contiguous.len()).min(PLAYBACK_BYTES as usize) / FRAME_BYTES
                * FRAME_BYTES;
            if length == 0 {
                break;
            }
            playback
                .write_copy(&contiguous[..length], 0, SeekMode::Relative)
                .map_err(|_| OriginalMicrophoneError::Transport)?;
            self.forwarded_bytes = self.forwarded_bytes.saturating_add(length as u64);
            self.pending.drain(..length);
        }
        let capture = self
            .capture
            .as_mut()
            .ok_or(OriginalMicrophoneError::Transport)?;
        let mut captured = 0;
        loop {
            let readable = capture
                .readable_size()
                .ok_or(OriginalMicrophoneError::Transport)?;
            if readable == 0 {
                break;
            }
            if readable > CAPTURE_BYTES as usize {
                lease.fail(OriginalMicrophoneError::Buffer, "capture_readable_bound");
                return Err(OriginalMicrophoneError::Buffer);
            }
            match capture
                .peek()
                .map_err(|_| OriginalMicrophoneError::Transport)?
            {
                PeekResult::Empty => break,
                PeekResult::Hole(_) => {
                    lease.fail(OriginalMicrophoneError::Transport, "capture_hole");
                    return Err(OriginalMicrophoneError::Transport);
                }
                PeekResult::Data(data) => {
                    let fragment_bytes = data.len();
                    self.captured_bytes = self.captured_bytes.saturating_add(fragment_bytes as u64);
                    captured += data.len();
                    if captured > CAPTURE_BYTES as usize {
                        lease.fail(OriginalMicrophoneError::Buffer, "capture_turn_bound");
                        return Err(OriginalMicrophoneError::Buffer);
                    }
                    if ready {
                        if let Err(error) = append_capture(&mut self.pending, data) {
                            if lease.cancel_once() {
                                let playback_attr = playback.get_buffer_attr().copied();
                                let playback_timing = playback.get_timing_info().map(|timing| {
                                    (
                                        timing.read_index,
                                        timing.write_index,
                                        timing.read_index_corrupt,
                                        timing.write_index_corrupt,
                                        timing.playing,
                                        timing.configured_sink_usec,
                                    )
                                });
                                let capture_timing = capture.get_timing_info().map(|timing| {
                                    (
                                        timing.read_index,
                                        timing.write_index,
                                        timing.read_index_corrupt,
                                        timing.write_index_corrupt,
                                        timing.configured_source_usec,
                                    )
                                });
                                tracing::warn!(
                                    event = "original_microphone_forwarding_failed",
                                    code = ?error,
                                    component = "pending_capture_bound",
                                    elapsed_ms = self.streaming_started_at.elapsed().as_millis() as u64,
                                    captured_bytes = self.captured_bytes,
                                    forwarded_bytes = self.forwarded_bytes,
                                    pending_bytes = self.pending.len(),
                                    fragment_bytes,
                                    writable_bytes = ?playback.writable_size(),
                                    playback_target_bytes = ?playback_attr.map(|attr| attr.tlength),
                                    playback_prebuffer_bytes = ?playback_attr.map(|attr| attr.prebuf),
                                    playback_min_request_bytes = ?playback_attr.map(|attr| attr.minreq),
                                    playback_corked = ?playback.is_corked().ok(),
                                    playback_suspended = ?playback.is_suspended().ok(),
                                    playback_underflow_index = ?playback.get_underflow_index(),
                                    playback_latency = ?playback.get_latency().ok(),
                                    capture_latency = ?capture.get_latency().ok(),
                                    playback_timing = ?playback_timing,
                                    capture_timing = ?capture_timing,
                                    "Native original microphone forwarding stopped"
                                );
                            }
                            return Err(error);
                        }
                    }
                }
            }
            capture
                .discard()
                .map_err(|_| OriginalMicrophoneError::Transport)?;
        }
        Ok(captured)
    }

    fn inspect(
        &mut self,
        registration: &OriginalMicrophoneRegistration,
        initial_zero: bool,
        deadline: Instant,
    ) -> Result<(), OriginalMicrophoneError> {
        self.check_streams(registration)?;
        check_work(&registration.lease, deadline)?;
        let checks = Rc::new(RefCell::new([Inspection::default(); 5]));
        let introspector = self.context.introspect();
        let results = Rc::clone(&checks);
        let identity = registration.clone();
        let mut playback =
            introspector.get_sink_input_info(registration.playback_index, move |result| {
                record_inspection(&results, 0, result, |info| {
                    info.index == identity.playback_index
                        && info.sink == identity.sink_index
                        && info.client == Some(identity.client_id)
                        && info.name.as_deref() == Some(MICROPHONE_ORIGINAL_PLAYBACK)
                        && info.sample_spec == sample_spec()
                        && info.channel_map == mono_map()
                        && info.has_volume
                        && info.volume_writable
                        && info.volume.len() == 1
                        && (!initial_zero || info.volume == Volume::MUTED)
                        && !info.mute
                        && !info.corked
                        && properties_match(
                            &info.proplist,
                            &identity,
                            Some(MICROPHONE_ORIGINAL_PLAYBACK),
                        )
                });
            });
        let results = Rc::clone(&checks);
        let identity = registration.clone();
        let mut capture =
            introspector.get_source_output_info(registration.capture_index, move |result| {
                record_inspection(&results, 1, result, |info| {
                    info.index == identity.capture_index
                        && info.source == identity.source_index
                        && info.client == Some(identity.client_id)
                        && info.name.as_deref() == Some(MICROPHONE_ORIGINAL_CAPTURE)
                        && info.sample_spec == sample_spec()
                        && info.channel_map == mono_map()
                        && info.has_volume
                        && info.volume.len() == 1
                        && info.volume == Volume::NORMAL
                        && !info.mute
                        && !info.corked
                        && properties_match(
                            &info.proplist,
                            &identity,
                            Some(MICROPHONE_ORIGINAL_CAPTURE),
                        )
                });
            });
        let results = Rc::clone(&checks);
        let identity = registration.clone();
        let mut source =
            introspector.get_source_info_by_index(registration.source_index, move |result| {
                record_inspection(&results, 2, result, |info| {
                    info.index == identity.source_index
                        && info.name.as_deref() == Some(identity.source_name.as_str())
                });
            });
        let results = Rc::clone(&checks);
        let expected_sink = registration.sink_index;
        let mut sink = introspector.get_sink_info_by_index(expected_sink, move |result| {
            record_inspection(&results, 3, result, |info| {
                info.index == expected_sink && info.name.as_deref() == Some(MIC_OUT_SINK)
            });
        });
        let results = Rc::clone(&checks);
        let identity = registration.clone();
        let mut client = introspector.get_client_info(registration.client_id, move |result| {
            record_inspection(&results, 4, result, |info| {
                info.index == identity.client_id
                    && properties_match(&info.proplist, &identity, None)
            });
        });
        let outcome = (|| {
            loop {
                check_work(&registration.lease, deadline)?;
                self.iterate_streams(registration, !initial_zero)?;
                let observed = checks.borrow();
                if observed.iter().all(|check| check.complete) {
                    if observed
                        .iter()
                        .any(|check| !check.valid || check.items != 1)
                    {
                        return Err(OriginalMicrophoneError::Identity);
                    }
                    return Ok(());
                }
                if observed.iter().any(|check| !check.valid) {
                    return Err(OriginalMicrophoneError::Identity);
                }
                drop(observed);
                thread::sleep(POLL_INTERVAL);
            }
        })();
        if outcome.is_err() {
            playback.cancel();
            capture.cancel();
            source.cancel();
            sink.cancel();
            client.cancel();
        }
        outcome
    }

    fn disconnect(&mut self, deadline: Instant) -> Result<(), OriginalMicrophoneError> {
        self.pending.clear();
        for stream in [&mut self.capture, &mut self.playback]
            .into_iter()
            .flatten()
        {
            stream.set_state_callback(None);
            stream.set_moved_callback(None);
            stream.set_overflow_callback(None);
            stream.set_buffer_attr_callback(None);
        }
        while [&self.capture, &self.playback]
            .into_iter()
            .flatten()
            .any(|stream| stream.get_state() == StreamState::Creating)
        {
            if Instant::now() >= deadline || !self.mainloop.iterate(false).is_success() {
                return Err(OriginalMicrophoneError::Cleanup);
            }
            thread::sleep(POLL_INTERVAL);
        }
        for stream in [&mut self.capture, &mut self.playback]
            .into_iter()
            .flatten()
        {
            if !matches!(
                stream.get_state(),
                StreamState::Unconnected | StreamState::Failed | StreamState::Terminated
            ) {
                stream
                    .disconnect()
                    .map_err(|_| OriginalMicrophoneError::Cleanup)?;
            }
        }
        while [&self.capture, &self.playback]
            .into_iter()
            .flatten()
            .any(|stream| {
                !matches!(
                    stream.get_state(),
                    StreamState::Unconnected | StreamState::Failed | StreamState::Terminated
                )
            })
        {
            if Instant::now() >= deadline || !self.mainloop.iterate(false).is_success() {
                return Err(OriginalMicrophoneError::Cleanup);
            }
            thread::sleep(POLL_INTERVAL);
        }
        self.context.set_state_callback(None);
        self.context.disconnect();
        Ok(())
    }
}

impl Drop for NativeTransport {
    fn drop(&mut self) {
        self.context.set_state_callback(None);
        self.context.disconnect();
    }
}

fn watch_stream(stream: &mut Stream, lease: &Arc<Lease>) {
    let state = Arc::clone(lease);
    stream.set_state_callback(Some(Box::new(move || {
        state.fail(OriginalMicrophoneError::Transport, "stream_state")
    })));
    let moved = Arc::clone(lease);
    stream.set_moved_callback(Some(Box::new(move || {
        moved.fail(OriginalMicrophoneError::Identity, "stream_moved")
    })));
    let overflow = Arc::clone(lease);
    stream.set_overflow_callback(Some(Box::new(move || {
        overflow.fail(OriginalMicrophoneError::Buffer, "stream_overflow")
    })));
    let buffer = Arc::clone(lease);
    stream.set_buffer_attr_callback(Some(Box::new(move || {
        buffer.fail(OriginalMicrophoneError::Buffer, "stream_buffer_attributes")
    })));
}

fn native_properties(
    session_id: Uuid,
    name: Option<&str>,
) -> Result<Proplist, OriginalMicrophoneError> {
    let mut properties = Proplist::new().ok_or(OriginalMicrophoneError::Transport)?;
    for (key, value) in [
        (properties::APPLICATION_NAME, APPLICATION_NAME.to_owned()),
        (
            properties::APPLICATION_PROCESS_ID,
            std::process::id().to_string(),
        ),
        (SESSION_PROPERTY, session_id.to_string()),
    ] {
        properties
            .set_str(key, &value)
            .map_err(|_| OriginalMicrophoneError::Transport)?;
    }
    if let Some(name) = name {
        properties
            .set_str(properties::MEDIA_NAME, name)
            .map_err(|_| OriginalMicrophoneError::Transport)?;
        properties
            .set_str("module-stream-restore.id", &format!("{name}.{session_id}"))
            .map_err(|_| OriginalMicrophoneError::Transport)?;
    }
    Ok(properties)
}

fn properties_match(
    properties: &Proplist,
    registration: &OriginalMicrophoneRegistration,
    name: Option<&str>,
) -> bool {
    properties.get_str(properties::APPLICATION_NAME).as_deref() == Some(APPLICATION_NAME)
        && properties
            .get_str(properties::APPLICATION_PROCESS_ID)
            .as_deref()
            == Some(registration.process_id.to_string().as_str())
        && properties.get_str(SESSION_PROPERTY).as_deref()
            == Some(registration.session_id.to_string().as_str())
        && name
            .is_none_or(|name| properties.get_str(properties::MEDIA_NAME).as_deref() == Some(name))
}

fn sample_spec() -> sample::Spec {
    sample::Spec {
        format: sample::Format::S16le,
        channels: 1,
        rate: 48_000,
    }
}

fn mono_map() -> channelmap::Map {
    let mut map = channelmap::Map::default();
    map.init_mono();
    map
}

fn native_stream_flags() -> StreamFlags {
    StreamFlags::DONT_MOVE
        | StreamFlags::START_UNMUTED
        | StreamFlags::ADJUST_LATENCY
        | StreamFlags::AUTO_TIMING_UPDATE
}

fn validate_native_formats(
    registration: &OriginalMicrophoneRegistration,
    specs: [Option<sample::Spec>; 2],
    maps: [Option<channelmap::Map>; 2],
) -> Result<(), OriginalMicrophoneError> {
    if specs.iter().any(|actual| *actual != Some(sample_spec()))
        || maps.iter().any(|actual| *actual != Some(mono_map()))
    {
        registration
            .lease
            .fail(OriginalMicrophoneError::Identity, "native_format");
        return Err(OriginalMicrophoneError::Identity);
    }
    Ok(())
}

fn capture_buffer() -> BufferAttr {
    BufferAttr {
        maxlength: CAPTURE_BYTES,
        tlength: u32::MAX,
        prebuf: u32::MAX,
        minreq: u32::MAX,
        // PipeWire needs room for four fragments without increasing maxlength.
        fragsize: CAPTURE_BYTES / 4,
    }
}

fn playback_buffer() -> BufferAttr {
    BufferAttr {
        maxlength: PLAYBACK_BYTES,
        tlength: 4_800,
        prebuf: 1_920,
        minreq: 960,
        fragsize: u32::MAX,
    }
}

fn validate_buffers(
    capture: &BufferAttr,
    playback: &BufferAttr,
) -> Result<(), OriginalMicrophoneError> {
    if capture.maxlength == 0
        || capture.maxlength > CAPTURE_BYTES
        || capture.fragsize == 0
        || capture.fragsize > capture.maxlength
        || playback.maxlength == 0
        || playback.maxlength > PLAYBACK_BYTES
        || playback.tlength == 0
        || playback.tlength > playback.maxlength
        || playback.prebuf > playback.tlength
        || playback.minreq == 0
        || playback.minreq > playback.tlength
        || [
            capture.maxlength,
            capture.fragsize,
            playback.maxlength,
            playback.tlength,
            playback.prebuf,
            playback.minreq,
        ]
        .iter()
        .any(|length| *length as usize % FRAME_BYTES != 0)
    {
        Err(OriginalMicrophoneError::Buffer)
    } else {
        Ok(())
    }
}

fn dispatch_ready(
    mut iterate: impl FnMut() -> Result<u32, OriginalMicrophoneError>,
) -> Result<(), OriginalMicrophoneError> {
    for _ in 0..READY_DISPATCH_LIMIT {
        if iterate()? == 0 {
            return Ok(());
        }
    }
    Ok(())
}

fn activation_ready(
    lease: &Lease,
    deadline: Instant,
    acknowledgement: Option<bool>,
    consumer_progress: bool,
) -> Result<bool, OriginalMicrophoneError> {
    check_work(lease, deadline)?;
    match acknowledgement {
        Some(true) => Ok(consumer_progress),
        Some(false) => Err(OriginalMicrophoneError::Transport),
        None => Ok(false),
    }
}

fn append_capture(pending: &mut VecDeque<u8>, data: &[u8]) -> Result<(), OriginalMicrophoneError> {
    if data.len() % FRAME_BYTES != 0 || data.len() > PENDING_BYTES.saturating_sub(pending.len()) {
        return Err(OriginalMicrophoneError::Buffer);
    }
    pending.extend(data);
    Ok(())
}

#[derive(Clone, Copy)]
struct Inspection {
    items: usize,
    valid: bool,
    complete: bool,
}

impl Default for Inspection {
    fn default() -> Self {
        Self {
            items: 0,
            valid: true,
            complete: false,
        }
    }
}

fn record_inspection<T>(
    checks: &Rc<RefCell<[Inspection; 5]>>,
    index: usize,
    result: ListResult<&T>,
    matches: impl FnOnce(&T) -> bool,
) {
    let mut checks = checks.borrow_mut();
    let check = &mut checks[index];
    match result {
        ListResult::Item(info) => {
            check.items += 1;
            check.valid &= check.items == 1 && matches(info);
        }
        ListResult::End => check.complete = true,
        ListResult::Error => {
            check.valid = false;
            check.complete = true;
        }
    }
}

fn validate_request(
    source: &str,
    source_index: u32,
    sink_index: u32,
) -> Result<(), OriginalMicrophoneError> {
    if source.is_empty()
        || source.contains('\0')
        || source_index == INVALID_INDEX
        || sink_index == INVALID_INDEX
    {
        return Err(OriginalMicrophoneError::Identity);
    }
    Ok(())
}

fn check_cancelled(lease: &Lease) -> Result<(), OriginalMicrophoneError> {
    if lease.cancelled.load(Ordering::Acquire) {
        Err(OriginalMicrophoneError::Cancelled)
    } else {
        Ok(())
    }
}

fn check_deadline(deadline: Instant) -> Result<(), OriginalMicrophoneError> {
    if Instant::now() >= deadline {
        Err(OriginalMicrophoneError::Deadline)
    } else {
        Ok(())
    }
}

fn check_work(lease: &Lease, deadline: Instant) -> Result<(), OriginalMicrophoneError> {
    check_cancelled(lease)?;
    check_deadline(deadline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(registry: &OriginalMicrophoneRegistry) -> OriginalMicrophoneRegistration {
        registry.test_register(
            12,
            13,
            4,
            5,
            "fixture-microphone",
            Uuid::new_v4(),
            std::process::id(),
            8,
        )
    }

    #[test]
    fn registry_requires_a_live_retained_native_lease() {
        let registry = OriginalMicrophoneRegistry::default();
        assert!(registry.current().unwrap().is_none());
        let registration = fixture(&registry);
        assert!(registration.is_live());
        assert!(
            registry
                .clone()
                .current()
                .unwrap()
                .unwrap()
                .same_session(&registration)
        );
        registration.test_invalidate();
        assert!(!registration.is_live());
        assert_eq!(
            registry.current().unwrap_err(),
            OriginalMicrophoneError::Custody
        );
        assert_eq!(
            registry.claim(Uuid::new_v4()),
            Err(OriginalMicrophoneError::Custody)
        );
        registry.release(registration.session_id).unwrap();
        assert!(registry.current().unwrap().is_none());
    }

    #[test]
    fn registry_cancellation_revokes_clones_and_preserves_cleanup_custody() {
        let registry = OriginalMicrophoneRegistry::default();
        let registration = fixture(&registry);
        let clone = registration.clone();
        registry.clone().cancel_current();
        assert!(!registration.is_live());
        assert!(!clone.is_live());
        assert_eq!(
            registry.current().unwrap_err(),
            OriginalMicrophoneError::Custody
        );
        assert_eq!(
            registry.claim(Uuid::new_v4()),
            Err(OriginalMicrophoneError::Custody)
        );
        registry.cancel_current();
        let state = registry.state.lock().unwrap();
        assert_eq!(state.custody, Some(registration.session_id));
        assert!(
            state
                .registration
                .as_ref()
                .unwrap()
                .same_session(&registration)
        );
        drop(state);
        registry.release(registration.session_id).unwrap();
        assert!(registry.current().unwrap().is_none());
    }

    #[test]
    fn poisoned_registry_can_cancel_but_cannot_publish_or_release_custody() {
        let registry = OriginalMicrophoneRegistry::default();
        let registration = fixture(&registry);
        let state = registry.state.clone();
        let result = thread::spawn(move || {
            let _guard = state.lock().unwrap();
            panic!("fixture registry poison");
        })
        .join();
        assert!(result.is_err());
        assert!(registration.is_live());
        registry.cancel_current();
        assert!(!registration.is_live());
        assert_eq!(
            registry.current().unwrap_err(),
            OriginalMicrophoneError::Custody
        );
        assert_eq!(
            registry.claim(Uuid::new_v4()),
            Err(OriginalMicrophoneError::Custody)
        );
        assert_eq!(
            registry.release(registration.session_id),
            Err(OriginalMicrophoneError::Custody)
        );
        let state = registry
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert_eq!(state.custody, Some(registration.session_id));
        assert!(
            state
                .registration
                .as_ref()
                .unwrap()
                .same_session(&registration)
        );
    }

    #[test]
    fn native_format_change_revokes_live_lease_and_blocks_replacement() {
        for stream in 0..2 {
            for mutation in 0..5 {
                let registry = OriginalMicrophoneRegistry::default();
                let registration = fixture(&registry);
                let mut specs = [Some(sample_spec()); 2];
                let mut maps = [Some(mono_map()); 2];
                assert_eq!(validate_native_formats(&registration, specs, maps), Ok(()));
                assert!(registration.is_live());
                match mutation {
                    0 => specs[stream].as_mut().unwrap().format = sample::Format::S16be,
                    1 => specs[stream].as_mut().unwrap().rate = 44_100,
                    2 => specs[stream].as_mut().unwrap().channels = 2,
                    3 => {
                        maps[stream].as_mut().unwrap().init_stereo();
                    }
                    _ => specs[stream] = None,
                }
                assert_eq!(
                    validate_native_formats(&registration, specs, maps),
                    Err(OriginalMicrophoneError::Identity)
                );
                assert!(!registration.is_live());
                assert_eq!(
                    registry.current().unwrap_err(),
                    OriginalMicrophoneError::Custody
                );
                assert_eq!(
                    registry.claim(Uuid::new_v4()),
                    Err(OriginalMicrophoneError::Custody)
                );
            }
        }
    }

    #[test]
    fn identical_metadata_without_the_original_lease_is_not_the_same_session() {
        let registry = OriginalMicrophoneRegistry::default();
        let first = fixture(&registry);
        let replacement = registry.test_register(
            first.playback_index,
            first.capture_index,
            first.source_index,
            first.sink_index,
            first.source_name(),
            first.session_id,
            first.process_id,
            first.client_id,
        );
        assert!(!first.is_live());
        assert!(replacement.is_live());
        assert!(!first.same_session(&replacement));
        assert!(replacement.same_session(&replacement.clone()));
    }

    #[test]
    fn claimed_or_dead_worker_custody_never_looks_like_an_empty_registry() {
        let registry = OriginalMicrophoneRegistry::default();
        let session = Uuid::new_v4();
        registry.claim(session).unwrap();
        assert_eq!(
            registry.current().unwrap_err(),
            OriginalMicrophoneError::Custody
        );
        assert_eq!(
            registry.claim(Uuid::new_v4()),
            Err(OriginalMicrophoneError::Custody)
        );
        registry.release(session).unwrap();
        let registration = fixture(&registry);
        drop(WorkerLife {
            registry: registry.clone(),
            lease: registration.lease.clone(),
            session_id: registration.session_id,
        });
        assert!(!registration.is_live());
        assert_eq!(
            registry.current().unwrap_err(),
            OriginalMicrophoneError::Custody
        );
    }

    #[test]
    fn cancellation_prevents_publication_even_if_alive_was_previously_set() {
        let registry = OriginalMicrophoneRegistry::default();
        let registration = fixture(&registry);
        registry.test_clear();
        registry.claim(registration.session_id).unwrap();
        registration.lease.alive.store(true, Ordering::Release);
        assert!(!registration.is_live());
        assert_eq!(
            registry.publish(registration),
            Err(OriginalMicrophoneError::Cancelled)
        );
    }

    #[test]
    fn expired_prepare_and_invalid_source_never_claim_or_spawn() {
        let registry = OriginalMicrophoneRegistry::default();
        let mut native = PulseOriginalMicrophone::new(registry.clone());
        assert_eq!(
            native.prepare("fixture", 1, 2, Instant::now()),
            Err(OriginalMicrophoneError::Deadline)
        );
        for source in ["", "bad\0source"] {
            assert_eq!(
                native.prepare(source, 1, 2, Instant::now() + STARTUP_LIMIT),
                Err(OriginalMicrophoneError::Identity)
            );
        }
        assert!(native.worker.is_none());
        assert!(registry.current().unwrap().is_none());
    }

    #[test]
    fn stop_invalidates_immediately_and_retains_unjoined_worker() {
        let registry = OriginalMicrophoneRegistry::default();
        let registration = fixture(&registry);
        let (release, blocker) = mpsc::sync_channel(1);
        let handle = thread::spawn(move || {
            blocker.recv().unwrap();
            Ok(())
        });
        let (commands, _receiver) = mpsc::sync_channel(1);
        let mut native = PulseOriginalMicrophone::new(registry.clone());
        native.worker = Some(Worker {
            session_id: registration.session_id,
            lease: registration.lease.clone(),
            commands,
            handle: Some(handle),
            failed_join: None,
        });
        assert_eq!(
            native.stop(Instant::now()),
            Err(OriginalMicrophoneError::Deadline)
        );
        assert!(!registration.is_live());
        assert!(native.worker.as_ref().unwrap().handle.is_some());
        assert_eq!(
            registry.current().unwrap_err(),
            OriginalMicrophoneError::Custody
        );
        release.send(()).unwrap();
        native.stop(Instant::now() + CLEANUP_LIMIT).unwrap();
        assert!(native.worker.is_none());
        assert!(registry.current().unwrap().is_none());
    }

    #[test]
    fn unconfirmed_disconnect_retains_custody_after_worker_join() {
        let registry = OriginalMicrophoneRegistry::default();
        let registration = fixture(&registry);
        let (commands, _receiver) = mpsc::sync_channel(1);
        let handle = thread::spawn(|| Err(OriginalMicrophoneError::Cleanup));
        let mut native = PulseOriginalMicrophone::new(registry.clone());
        native.worker = Some(Worker {
            session_id: registration.session_id,
            lease: registration.lease.clone(),
            commands,
            handle: Some(handle),
            failed_join: None,
        });
        assert_eq!(
            native.stop(Instant::now() + CLEANUP_LIMIT),
            Err(OriginalMicrophoneError::Cleanup)
        );
        assert!(!registration.is_live());
        assert_eq!(
            native.stop(Instant::now() + CLEANUP_LIMIT),
            Err(OriginalMicrophoneError::Cleanup)
        );
        assert_eq!(
            registry.claim(Uuid::new_v4()),
            Err(OriginalMicrophoneError::Custody)
        );
    }

    #[test]
    fn worker_cancelled_before_connect_creates_no_native_context() {
        let registry = OriginalMicrophoneRegistry::default();
        let session_id = Uuid::new_v4();
        registry.claim(session_id).unwrap();
        let lease = Arc::new(Lease::default());
        lease.cancel();
        let (_commands, receiver) = mpsc::sync_channel(1);
        let (ready, result) = mpsc::sync_channel(1);
        assert_eq!(
            run_worker(
                registry.clone(),
                lease,
                session_id,
                Route {
                    source: "fixture".to_owned(),
                    source_index: 1,
                    sink_index: 2,
                },
                Instant::now() + STARTUP_LIMIT,
                receiver,
                ready,
            ),
            Ok(())
        );
        assert_eq!(
            result.recv().unwrap(),
            Err(OriginalMicrophoneError::Cancelled)
        );
        assert_eq!(
            registry.current().unwrap_err(),
            OriginalMicrophoneError::Custody
        );
        registry.release(session_id).unwrap();
    }

    #[test]
    fn capture_request_remains_bounded_after_server_fragment_negotiation() {
        let mut negotiated = capture_buffer();
        assert_eq!(negotiated.maxlength, 4_800);
        // PipeWire queues at least four capture fragments, even for a lower maxlength.
        negotiated.maxlength = negotiated.maxlength.max(4 * negotiated.fragsize);
        assert_eq!(validate_buffers(&negotiated, &playback_buffer()), Ok(()));
    }

    #[test]
    fn negotiated_buffers_and_bridge_total_at_most_two_hundred_ms() {
        let capture = capture_buffer();
        let playback = playback_buffer();
        assert_eq!(validate_buffers(&capture, &playback), Ok(()));
        let total = capture.maxlength as usize + playback.maxlength as usize + PENDING_BYTES;
        assert_eq!(total, 48_000 * FRAME_BYTES / 5);
        let mut oversized = capture;
        oversized.maxlength += 2;
        assert_eq!(
            validate_buffers(&oversized, &playback),
            Err(OriginalMicrophoneError::Buffer)
        );
        let mut unaligned = capture;
        unaligned.maxlength -= 1;
        assert_eq!(
            validate_buffers(&unaligned, &playback),
            Err(OriginalMicrophoneError::Buffer)
        );
        unaligned = capture;
        unaligned.fragsize -= 1;
        assert_eq!(
            validate_buffers(&unaligned, &playback),
            Err(OriginalMicrophoneError::Buffer)
        );
        let mut oversized = playback;
        oversized.maxlength += 2;
        assert_eq!(
            validate_buffers(&capture, &oversized),
            Err(OriginalMicrophoneError::Buffer)
        );
        oversized = playback;
        oversized.prebuf = oversized.tlength + 2;
        assert_eq!(
            validate_buffers(&capture, &oversized),
            Err(OriginalMicrophoneError::Buffer)
        );
        oversized = playback;
        oversized.minreq = 1;
        assert_eq!(
            validate_buffers(&capture, &oversized),
            Err(OriginalMicrophoneError::Buffer)
        );
    }

    #[test]
    fn pending_absorbs_two_capture_windows_within_the_fixed_total_budget() {
        let mut pending = VecDeque::with_capacity(PENDING_BYTES);
        let captured = vec![7; CAPTURE_BYTES as usize];
        append_capture(&mut pending, &captured).unwrap();
        append_capture(&mut pending, &captured).unwrap();
        assert_eq!(pending.len(), PENDING_BYTES);
        assert_eq!(PENDING_BYTES, CAPTURE_BYTES as usize * 2);
        assert_eq!(PLAYBACK_BYTES, CAPTURE_BYTES);
        let before = pending.clone();
        assert_eq!(
            append_capture(&mut pending, &[1, 2]),
            Err(OriginalMicrophoneError::Buffer)
        );
        assert_eq!(pending, before);
    }

    #[test]
    fn capture_overflow_or_partial_sample_does_not_mutate_pending_audio() {
        let mut pending = VecDeque::with_capacity(PENDING_BYTES);
        append_capture(&mut pending, &vec![5; PENDING_BYTES]).unwrap();
        let before = pending.clone();
        assert_eq!(
            append_capture(&mut pending, &[1, 2]),
            Err(OriginalMicrophoneError::Buffer)
        );
        assert_eq!(pending, before);
        pending.clear();
        assert_eq!(
            append_capture(&mut pending, &[1]),
            Err(OriginalMicrophoneError::Buffer)
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn inspection_rejects_empty_duplicate_mismatched_and_failed_results() {
        let checks = Rc::new(RefCell::new([Inspection::default(); 5]));
        record_inspection::<u32>(&checks, 0, ListResult::End, |_| true);
        record_inspection(&checks, 1, ListResult::Item(&1), |_| true);
        record_inspection(&checks, 1, ListResult::Item(&1), |_| true);
        record_inspection(&checks, 2, ListResult::Item(&1), |_| false);
        record_inspection::<u32>(&checks, 3, ListResult::Error, |_| true);
        record_inspection(&checks, 4, ListResult::Item(&1), |value| *value == 1);
        record_inspection::<u32>(&checks, 4, ListResult::End, |_| true);
        let checks = checks.borrow();
        assert_eq!(checks[0].items, 0);
        assert!(!checks[1].valid);
        assert!(!checks[2].valid);
        assert!(!checks[3].valid);
        assert!(checks[4].complete && checks[4].valid && checks[4].items == 1);
    }

    #[test]
    fn native_properties_bind_both_streams_and_context_to_one_session() {
        let registry = OriginalMicrophoneRegistry::default();
        let registration = fixture(&registry);
        for name in [
            None,
            Some(MICROPHONE_ORIGINAL_CAPTURE),
            Some(MICROPHONE_ORIGINAL_PLAYBACK),
        ] {
            let mut props = native_properties(registration.session_id, name).unwrap();
            assert!(properties_match(&props, &registration, name));
            props
                .set_str(SESSION_PROPERTY, &Uuid::new_v4().to_string())
                .unwrap();
            assert!(!properties_match(&props, &registration, name));
        }
        assert_ne!(MICROPHONE_ORIGINAL_CAPTURE, MICROPHONE_ORIGINAL_PLAYBACK);
        assert_eq!(sample_spec().channels, 1);
        assert_eq!(sample_spec().rate, 48_000);
        assert_eq!(sample_spec().format, sample::Format::S16le);
    }

    #[test]
    fn native_session_identity_survives_server_restoration_policy_projection() {
        let registry = OriginalMicrophoneRegistry::default();
        let registration = fixture(&registry);
        for (name, server_key) in [
            (
                MICROPHONE_ORIGINAL_CAPTURE,
                "source-output-by-application-name:translator-daemon",
            ),
            (
                MICROPHONE_ORIGINAL_PLAYBACK,
                "sink-input-by-application-name:translator-daemon",
            ),
        ] {
            let mut props = native_properties(registration.session_id, Some(name)).unwrap();
            props
                .set_str("module-stream-restore.id", server_key)
                .unwrap();
            assert!(properties_match(&props, &registration, Some(name)));
            props.unset("module-stream-restore.id").unwrap();
            assert!(properties_match(&props, &registration, Some(name)));
        }
    }

    #[test]
    fn native_session_identity_still_rejects_each_required_property_mismatch() {
        let registry = OriginalMicrophoneRegistry::default();
        let registration = fixture(&registry);
        for name in [
            None,
            Some(MICROPHONE_ORIGINAL_CAPTURE),
            Some(MICROPHONE_ORIGINAL_PLAYBACK),
        ] {
            let baseline = || {
                let mut props = native_properties(registration.session_id, name).unwrap();
                props
                    .set_str("module-stream-restore.id", "server-owned-restoration-group")
                    .unwrap();
                assert!(properties_match(&props, &registration, name));
                props
            };
            let mut required = vec![
                (
                    properties::APPLICATION_NAME,
                    "foreign-application".to_owned(),
                ),
                (
                    properties::APPLICATION_PROCESS_ID,
                    (registration.process_id + 1).to_string(),
                ),
                (SESSION_PROPERTY, Uuid::new_v4().to_string()),
            ];
            if name.is_some() {
                required.push((properties::MEDIA_NAME, "foreign-stream".to_owned()));
            }
            for (key, foreign) in required {
                let mut corrupted = baseline();
                corrupted.set_str(key, &foreign).unwrap();
                assert!(!properties_match(&corrupted, &registration, name));
                let mut missing = baseline();
                missing.unset(key).unwrap();
                assert!(!properties_match(&missing, &registration, name));
            }
        }
    }

    #[test]
    fn native_activation_requires_success_acknowledgement_and_uncancelled_budget() {
        let lease = Lease::default();
        let deadline = Instant::now() + STARTUP_LIMIT;
        assert_eq!(activation_ready(&lease, deadline, None, true), Ok(false));
        assert_eq!(
            activation_ready(&lease, deadline, Some(true), false),
            Ok(false)
        );
        assert_eq!(
            activation_ready(&lease, deadline, Some(true), true),
            Ok(true)
        );
        assert_eq!(
            activation_ready(&lease, deadline, Some(false), true),
            Err(OriginalMicrophoneError::Transport)
        );
        assert_eq!(
            activation_ready(&lease, Instant::now(), Some(true), true),
            Err(OriginalMicrophoneError::Deadline)
        );
        lease.cancel();
        assert_eq!(
            activation_ready(&lease, deadline, Some(true), true),
            Err(OriginalMicrophoneError::Cancelled)
        );
    }

    #[test]
    fn native_streams_pin_devices_and_request_bounded_device_latency() {
        let flags = native_stream_flags();
        assert!(flags.contains(StreamFlags::DONT_MOVE));
        assert!(flags.contains(StreamFlags::START_UNMUTED));
        assert!(flags.contains(StreamFlags::ADJUST_LATENCY));
        assert!(flags.contains(StreamFlags::AUTO_TIMING_UPDATE));
        assert!(!flags.contains(StreamFlags::START_CORKED));
        assert!(!flags.contains(StreamFlags::FIX_FORMAT));
        assert!(!flags.contains(StreamFlags::FIX_CHANNELS));
    }

    #[test]
    fn ready_dispatch_drains_queued_events_without_an_unbounded_turn() {
        let mut calls = 0;
        assert_eq!(
            dispatch_ready(|| {
                calls += 1;
                Ok(u32::from(calls < 4))
            }),
            Ok(())
        );
        assert_eq!(calls, 4);
        calls = 0;
        assert_eq!(
            dispatch_ready(|| {
                calls += 1;
                Ok(1)
            }),
            Ok(())
        );
        assert_eq!(calls, READY_DISPATCH_LIMIT);
        assert_eq!(
            dispatch_ready(|| Err(OriginalMicrophoneError::Transport)),
            Err(OriginalMicrophoneError::Transport)
        );
    }

    #[test]
    #[ignore = "requires a disposable private PulseAudio socket and virtual fixture sinks"]
    fn private_pulse_cancellation_at_native_phases_confirms_cleanup() {
        use std::{os::unix::fs::FileTypeExt, process::Command};

        let server = std::env::var("PULSE_SERVER").expect("private PULSE_SERVER required");
        assert!(
            server.starts_with("unix:/tmp/translator-loopback-") && server.ends_with("/native"),
            "refusing non-fixture Pulse server"
        );
        assert!(
            std::fs::symlink_metadata(server.strip_prefix("unix:").unwrap())
                .unwrap()
                .file_type()
                .is_socket(),
            "fixture socket must be an actual socket"
        );
        let inventory = |kind: &str| -> Vec<serde_json::Value> {
            let output = Command::new("pactl")
                .args(["--format=json", "list", kind])
                .output()
                .unwrap();
            assert!(output.status.success());
            serde_json::from_slice(&output.stdout).unwrap()
        };
        let microphone = "translator_test_mic.monitor";
        let endpoint = |kind: &str, name: &str| -> u32 {
            let values = inventory(kind);
            values.iter().find(|value| value["name"] == name).unwrap()["index"]
                .as_u64()
                .unwrap()
                .try_into()
                .unwrap()
        };
        let source_index = endpoint("sources", microphone);
        let sink_index = endpoint("sinks", MIC_OUT_SINK);
        for phase in [
            NativePhase::Context,
            NativePhase::Playback,
            NativePhase::Capture,
            NativePhase::Activating,
            NativePhase::Verified,
            NativePhase::Published,
        ] {
            let registry = OriginalMicrophoneRegistry::default();
            let session_id = Uuid::new_v4();
            registry.claim(session_id).unwrap();
            let lease = Arc::new(Lease::default());
            let mut native = NativeTransport::new(session_id).unwrap();
            native.cancel_at = Some(phase);
            let result = native.connect(
                Route {
                    source: microphone.to_owned(),
                    source_index,
                    sink_index,
                },
                session_id,
                lease.clone(),
                Instant::now() + STARTUP_LIMIT,
            );
            if phase == NativePhase::Published {
                let registration = result.unwrap();
                registry.publish(registration.clone()).unwrap();
                assert!(registration.is_live());
                assert_eq!(
                    native.checkpoint(phase, &lease),
                    Err(OriginalMicrophoneError::Cancelled)
                );
                assert!(!registration.is_live());
            } else {
                assert_eq!(
                    result.unwrap_err(),
                    OriginalMicrophoneError::Cancelled,
                    "{phase:?}"
                );
            }
            assert!(lease.cancelled.load(Ordering::Acquire));
            assert_eq!(
                registry.current().unwrap_err(),
                OriginalMicrophoneError::Custody
            );
            assert_eq!(
                registry.claim(Uuid::new_v4()),
                Err(OriginalMicrophoneError::Custody)
            );
            native.disconnect(Instant::now() + CLEANUP_LIMIT).unwrap();
            assert_eq!(native.context.get_state(), ContextState::Terminated);
            assert!(native.pending.is_empty());
            for stream in [&native.capture, &native.playback].into_iter().flatten() {
                assert!(matches!(
                    stream.get_state(),
                    StreamState::Failed | StreamState::Terminated
                ));
            }
            for kind in ["sink-inputs", "source-outputs"] {
                assert!(
                    inventory(kind).iter().all(|value| {
                        value["properties"][SESSION_PROPERTY] != session_id.to_string()
                    }),
                    "native resources must be absent after {phase:?}"
                );
            }
            registry.release(session_id).unwrap();
            assert!(registry.current().unwrap().is_none());
        }
    }
}
