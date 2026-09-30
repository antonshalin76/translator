//! Bounded decoder for the isolated AEC helper's non-admissible sample stream.

use crate::AecMeasurementError;

const MAX_BODY: usize = 4 * 1024 * 1024;
const FRAME_SAMPLES: usize = 480;
const WINDOW_SAMPLES: usize = 48_000;
const FRAME_BODY: usize = 72 + FRAME_SAMPLES * 3 * 4;
const INVALID_ID: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Hello,
    Links,
    Armed,
    Frames,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AecBackendIdentity {
    pub session: u64,
    pub generation: u64,
    pub node_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AecBackendLink {
    pub local_port: u32,
    pub link_id: u32,
    pub peer_node: u32,
    pub peer_port: u32,
}

/// A transport-verified native window, not a calibration acquisition or AEC proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AecBackendWindow {
    pub identity: AecBackendIdentity,
    pub links: [AecBackendLink; 3],
    pub clock_id: u32,
    pub xrun: u64,
    pub first_sequence: u64,
    pub last_sequence: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub raw: Vec<i16>,
    pub clean: Vec<i16>,
}

/// Transport facts are not physical clock provenance or an AEC proof.
pub struct AecBackendWire {
    stage: Stage,
    pending: Vec<u8>,
    hello: Option<AecBackendIdentity>,
    links: Option<[AecBackendLink; 3]>,
    clock: Option<u32>,
    window_start_position: Option<u64>,
    window_first_sequence: Option<u64>,
    next_position: Option<u64>,
    next_sequence: Option<u64>,
    xrun: Option<u64>,
    rate: Option<(u32, u32)>,
    raw: Vec<i16>,
    clean: Vec<i16>,
    accepted_frames: u64,
}

impl Default for AecBackendWire {
    fn default() -> Self {
        Self::new()
    }
}

impl AecBackendWire {
    pub fn new() -> Self {
        Self {
            stage: Stage::Hello,
            pending: Vec::new(),
            hello: None,
            links: None,
            clock: None,
            window_start_position: None,
            window_first_sequence: None,
            next_position: None,
            next_sequence: None,
            xrun: None,
            rate: None,
            raw: Vec::new(),
            clean: Vec::new(),
            accepted_frames: 0,
        }
    }

    pub fn push_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<Vec<AecBackendWindow>, AecMeasurementError> {
        if self.stage == Stage::Invalid {
            return Err(AecMeasurementError::Invalidated);
        }
        let result = self.feed(bytes);
        if result.is_err() {
            self.stage = Stage::Invalid;
            self.pending.clear();
            self.raw.clear();
            self.clean.clear();
        }
        result
    }

    /// Increases only after a complete FRAME passes every metadata and PCM check.
    pub fn accepted_frames(&self) -> u64 {
        self.accepted_frames
    }

    /// Claimed transport identity; caller must compare with its launched helper.
    pub fn identity(&self) -> Option<AecBackendIdentity> {
        self.hello
    }

    /// Claimed graph links; caller must compare with the private server inventory.
    pub fn links(&self) -> Option<[AecBackendLink; 3]> {
        self.links
    }

    /// Checks framing only; the session owner must verify the intended interval.
    pub fn finish(&self) -> Result<(), AecMeasurementError> {
        if self.stage == Stage::Invalid {
            Err(AecMeasurementError::Invalidated)
        } else if self.stage != Stage::Frames || !self.pending.is_empty() || !self.raw.is_empty() {
            Err(AecMeasurementError::Incomplete)
        } else {
            Ok(())
        }
    }

    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<AecBackendWindow>, AecMeasurementError> {
        if self
            .pending
            .len()
            .checked_add(bytes.len())
            .is_none_or(|len| len > MAX_BODY + 4)
        {
            return Err(AecMeasurementError::InvalidFormat);
        }
        self.pending.extend_from_slice(bytes);
        let mut windows = Vec::new();
        loop {
            if self.pending.len() < 4 {
                break;
            }
            let body_len = u32::from_le_bytes(self.pending[..4].try_into().unwrap()) as usize;
            if !(4..=MAX_BODY).contains(&body_len) {
                return Err(AecMeasurementError::InvalidFormat);
            }
            if self.pending.len() < body_len + 4 {
                break;
            }
            let body = self.pending[4..body_len + 4].to_vec();
            self.pending.drain(..body_len + 4);
            if body[1] != 1 || body[2..4] != [0, 0] {
                return Err(AecMeasurementError::InvalidFormat);
            }
            match (self.stage, body[0]) {
                (Stage::Hello, 1) => self.accept_hello(&body)?,
                (Stage::Links, 6) => self.accept_links(&body)?,
                (Stage::Armed, 7) => self.accept_armed(&body)?,
                (Stage::Frames, 2) => {
                    if let Some(pair) = self.accept_frame(&body)? {
                        windows.push(pair);
                    }
                }
                (_, 3) => return Err(self.accept_fatal(&body)?),
                _ => return Err(AecMeasurementError::InvalidProvenance),
            }
        }
        Ok(windows)
    }

    fn accept_hello(&mut self, body: &[u8]) -> Result<(), AecMeasurementError> {
        if body.len() != 36
            || read_u32(body, 24) != 48_000
            || read_u32(body, 28) != 480
            || read_u32(body, 32) != 480
        {
            return Err(AecMeasurementError::InvalidFormat);
        }
        let hello = AecBackendIdentity {
            session: read_u64(body, 4),
            generation: read_u64(body, 12),
            node_id: read_u32(body, 20),
        };
        if hello.session == 0 || hello.generation == 0 || hello.node_id == INVALID_ID {
            return Err(AecMeasurementError::InvalidProvenance);
        }
        self.hello = Some(hello);
        self.stage = Stage::Links;
        Ok(())
    }

    fn accept_links(&mut self, body: &[u8]) -> Result<(), AecMeasurementError> {
        if body.len() != 72 {
            return Err(AecMeasurementError::InvalidFormat);
        }
        let hello = self.hello.ok_or(AecMeasurementError::InvalidProvenance)?;
        if read_u64(body, 4) != hello.session
            || read_u64(body, 12) != hello.generation
            || read_u32(body, 20) != hello.node_id
        {
            return Err(AecMeasurementError::InvalidProvenance);
        }
        let links = std::array::from_fn(|i| {
            let at = 24 + i * 16;
            AecBackendLink {
                local_port: read_u32(body, at),
                link_id: read_u32(body, at + 4),
                peer_node: read_u32(body, at + 8),
                peer_port: read_u32(body, at + 12),
            }
        });
        if links.iter().any(|link| {
            [
                link.local_port,
                link.link_id,
                link.peer_node,
                link.peer_port,
            ]
            .contains(&INVALID_ID)
                || link.peer_node == hello.node_id
        }) || links[0].local_port == links[1].local_port
            || links[0].local_port == links[2].local_port
            || links[1].local_port == links[2].local_port
            || links[0].link_id == links[1].link_id
            || links[0].link_id == links[2].link_id
            || links[1].link_id == links[2].link_id
        {
            return Err(AecMeasurementError::InvalidProvenance);
        }
        self.links = Some(links);
        self.stage = Stage::Armed;
        Ok(())
    }

    fn accept_armed(&mut self, body: &[u8]) -> Result<(), AecMeasurementError> {
        if body.len() != 24 {
            return Err(AecMeasurementError::InvalidFormat);
        }
        let hello = self.hello.ok_or(AecMeasurementError::InvalidProvenance)?;
        if read_u64(body, 4) != hello.session
            || read_u64(body, 12) != hello.generation
            || read_u32(body, 20) != hello.node_id
        {
            return Err(AecMeasurementError::InvalidProvenance);
        }
        self.stage = Stage::Frames;
        Ok(())
    }

    fn accept_fatal(&self, body: &[u8]) -> Result<AecMeasurementError, AecMeasurementError> {
        if body.len() != 32 || read_u32(body, 20) == 0 {
            return Err(AecMeasurementError::InvalidFormat);
        }
        let hello = self.hello.ok_or(AecMeasurementError::InvalidProvenance)?;
        if read_u64(body, 4) != hello.session || read_u64(body, 12) != hello.generation {
            return Err(AecMeasurementError::InvalidProvenance);
        }
        Ok(AecMeasurementError::SourceFailed)
    }

    fn accept_frame(
        &mut self,
        body: &[u8],
    ) -> Result<Option<AecBackendWindow>, AecMeasurementError> {
        if body.len() != FRAME_BODY
            || read_u32(body, 32) != FRAME_SAMPLES as u32
            || read_u64(body, 44) != FRAME_SAMPLES as u64
        {
            return Err(AecMeasurementError::InvalidFormat);
        }
        let hello = self.hello.ok_or(AecMeasurementError::InvalidProvenance)?;
        let links = self.links.ok_or(AecMeasurementError::InvalidProvenance)?;
        if read_u64(body, 4) != hello.session
            || read_u64(body, 12) != hello.generation
            || read_u32(body, 68) != hello.node_id
        {
            return Err(AecMeasurementError::InvalidProvenance);
        }
        let sequence = read_u64(body, 20);
        let clock = read_u32(body, 28);
        let position = read_u64(body, 36);
        let xrun = read_u64(body, 52);
        let rate_num = read_u32(body, 60);
        let rate_denom = read_u32(body, 64);
        if clock == INVALID_ID || rate_num == 0 || rate_num.checked_mul(48_000) != Some(rate_denom)
        {
            return Err(AecMeasurementError::InvalidFormat);
        }
        if self.clock.is_some_and(|old| old != clock)
            || self.xrun.is_some_and(|old| old != xrun)
            || self.rate.is_some_and(|old| old != (rate_num, rate_denom))
            || self.next_sequence.is_some_and(|next| next != sequence)
            || self.next_position.is_some_and(|next| next != position)
        {
            return Err(AecMeasurementError::Discontinuous);
        }
        let next_sequence = sequence
            .checked_add(1)
            .ok_or(AecMeasurementError::Discontinuous)?;
        let next_position = position
            .checked_add(FRAME_SAMPLES as u64)
            .ok_or(AecMeasurementError::Discontinuous)?;
        if self.raw.is_empty() {
            self.window_start_position = Some(position);
            self.window_first_sequence = Some(sequence);
        }
        for i in 0..FRAME_SAMPLES {
            let at = 72 + i * 12;
            self.raw.push(decode_sample(body, at)?);
            let _reference = decode_sample(body, at + 4)?;
            self.clean.push(decode_sample(body, at + 8)?);
        }
        self.clock.get_or_insert(clock);
        self.xrun.get_or_insert(xrun);
        self.rate.get_or_insert((rate_num, rate_denom));
        self.next_sequence = Some(next_sequence);
        self.next_position = Some(next_position);
        self.accepted_frames = self
            .accepted_frames
            .checked_add(1)
            .ok_or(AecMeasurementError::Discontinuous)?;
        if self.raw.len() != WINDOW_SAMPLES {
            return Ok(None);
        }
        let window = AecBackendWindow {
            identity: hello,
            links,
            clock_id: clock,
            xrun,
            first_sequence: self
                .window_first_sequence
                .take()
                .ok_or(AecMeasurementError::Discontinuous)?,
            last_sequence: sequence,
            start_sample: self
                .window_start_position
                .take()
                .ok_or(AecMeasurementError::Discontinuous)?,
            end_sample: next_position,
            raw: std::mem::take(&mut self.raw),
            clean: std::mem::take(&mut self.clean),
        };
        Ok(Some(window))
    }
}

fn read_u32(body: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(body[at..at + 4].try_into().unwrap())
}

fn read_u64(body: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(body[at..at + 8].try_into().unwrap())
}

fn decode_sample(body: &[u8], at: usize) -> Result<i16, AecMeasurementError> {
    let value = f32::from_le_bytes(body[at..at + 4].try_into().unwrap());
    if !value.is_finite() {
        return Err(AecMeasurementError::InvalidFormat);
    }
    if !(-1.0..1.0).contains(&value) {
        return Err(AecMeasurementError::Clipped);
    }
    let pcm = (value * 32_768.0).round();
    if pcm <= f32::from(i16::MIN) || pcm >= f32::from(i16::MAX) {
        return Err(AecMeasurementError::Clipped);
    }
    Ok(pcm as i16)
}
