use std::collections::VecDeque;

use translator_audio::{
    BoundedPcmQueue, CaptureEvent, PcmFrame, PulsePcmCommand, SpeechSegmenter, StreamPcmFormat,
    VoiceDetector,
};
use uuid::Uuid;

#[derive(Default)]
struct ScriptedDetector {
    decisions: VecDeque<bool>,
}

impl ScriptedDetector {
    fn new(decisions: impl IntoIterator<Item = bool>) -> Self {
        Self {
            decisions: decisions.into_iter().collect(),
        }
    }
}

impl VoiceDetector for ScriptedDetector {
    fn is_voice(&mut self, _samples: &[i16]) -> Result<bool, translator_audio::VadError> {
        Ok(self.decisions.pop_front().unwrap_or(false))
    }
}

fn frame(sequence: u64) -> PcmFrame {
    PcmFrame::try_new(
        sequence,
        sequence * 20_000_000,
        StreamPcmFormat::provider_default(),
        vec![0; 640],
    )
    .unwrap()
}

fn low_latency_segmenter(detector: ScriptedDetector) -> SpeechSegmenter<ScriptedDetector> {
    SpeechSegmenter::with_confirmation_and_max_frames(Uuid::new_v4(), detector, 3, 400)
}

fn process_events(
    segmenter: &mut SpeechSegmenter<ScriptedDetector>,
    sequences: impl IntoIterator<Item = u64>,
) -> Vec<CaptureEvent> {
    sequences
        .into_iter()
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect()
}

fn speech_start_ids(events: &[CaptureEvent]) -> Vec<Uuid> {
    events
        .iter()
        .filter_map(|event| match event {
            CaptureEvent::SpeechStarted { utterance_id, .. } => Some(*utterance_id),
            CaptureEvent::Frame { .. } => None,
        })
        .collect()
}

fn frame_sequences(events: &[CaptureEvent]) -> Vec<(u64, bool)> {
    events
        .iter()
        .filter_map(|event| match event {
            CaptureEvent::Frame {
                frame,
                end_of_utterance,
                ..
            } => Some((frame.sequence(), *end_of_utterance)),
            CaptureEvent::SpeechStarted { .. } => None,
        })
        .collect()
}

fn utterance_frames(events: &[CaptureEvent]) -> Vec<(Uuid, u64, bool)> {
    events
        .iter()
        .filter_map(|event| match event {
            CaptureEvent::Frame {
                utterance_id,
                frame,
                end_of_utterance,
                ..
            } => Some((*utterance_id, frame.sequence(), *end_of_utterance)),
            CaptureEvent::SpeechStarted { .. } => None,
        })
        .collect()
}

#[test]
fn pulse_commands_request_pipewire_resampling_and_mark_owned_streams() {
    let capture = PulsePcmCommand::capture(
        "alsa_input.usb-Jieli_Technology_UACDemoV1.0-00.mono-fallback",
        "translator-outgoing-capture",
    );
    assert_eq!(capture.program(), "parec");
    assert!(capture.arguments().contains(&"--rate=16000".to_owned()));
    assert!(capture.arguments().contains(&"--channels=1".to_owned()));
    assert!(
        capture
            .arguments()
            .contains(&"--process-time-msec=20".to_owned())
    );
    assert!(
        capture
            .arguments()
            .contains(&"--property=translator.owner=true".to_owned())
    );

    let playback = PulsePcmCommand::playback("translator_mic_out", "translator-outgoing-playback");
    assert_eq!(playback.program(), "pacat");
    assert!(
        playback
            .arguments()
            .contains(&"--device=translator_mic_out".to_owned())
    );

    let session_id = Uuid::new_v4();
    let virtual_peer =
        PulsePcmCommand::virtual_peer_playback("alsa_output.usb-headset", session_id);
    assert_eq!(virtual_peer.program(), "pacat");
    assert!(
        virtual_peer
            .arguments()
            .contains(&"--client-name=translator-virtual-peer".to_owned())
    );
    assert!(
        virtual_peer
            .arguments()
            .contains(&"--property=translator.test_profile=human_round_trip".to_owned())
    );
    assert!(virtual_peer.arguments().contains(&format!(
        "--property=translator.self_test_session={session_id}"
    )));
}

#[test]
fn pcm_queue_is_bounded_to_four_hundred_milliseconds() {
    let mut queue = BoundedPcmQueue::default();
    for sequence in 0..20 {
        queue.push(frame(sequence)).unwrap();
    }
    assert_eq!(queue.buffered_ms(), 400);
    assert!(queue.push(frame(20)).is_err());
    assert_eq!(queue.dropped_frames(), 1);

    for sequence in 0..20 {
        assert_eq!(queue.pop().unwrap().sequence(), sequence);
    }
    assert_eq!(queue.buffered_ms(), 0);
}

#[test]
fn segmenter_keeps_stream_stable_and_rotates_utterance_after_eou() {
    let detector = ScriptedDetector::new(
        [true, true, true]
            .into_iter()
            .chain(std::iter::repeat_n(false, 15))
            .chain([true, true, true]),
    );
    let mut segmenter = low_latency_segmenter(detector);
    let stream_id = segmenter.stream_id();

    let first = process_events(&mut segmenter, 0..18);
    let first_id = first
        .iter()
        .find_map(|event| match event {
            CaptureEvent::SpeechStarted { utterance_id, .. } => Some(*utterance_id),
            _ => None,
        })
        .unwrap();
    assert!(first.iter().any(|event| matches!(
        event,
        CaptureEvent::Frame {
            stream_id: observed_stream,
            utterance_id,
            end_of_utterance: true,
            ..
        } if *observed_stream == stream_id && *utterance_id == first_id
    )));

    let second = process_events(&mut segmenter, 18..21);
    let second_id = second
        .iter()
        .find_map(|event| match event {
            CaptureEvent::SpeechStarted { utterance_id, .. } => Some(*utterance_id),
            _ => None,
        })
        .unwrap();
    assert_ne!(first_id, second_id);
    assert_eq!(segmenter.stream_id(), stream_id);
}

#[test]
fn segmenter_bounds_continuous_voice_without_dropping_frames() {
    let detector = ScriptedDetector::new(std::iter::repeat_n(true, 500));
    let mut segmenter = low_latency_segmenter(detector);

    let events = process_events(&mut segmenter, 0..500);
    let starts = speech_start_ids(&events);
    assert_eq!(starts.len(), 2);
    assert_ne!(starts[0], starts[1]);

    let frames = utterance_frames(&events);
    assert_eq!(frames.len(), 500);
    assert_eq!(frames[399], (starts[0], 399, true));
    assert_eq!(frames[400], (starts[1], 400, false));
    assert_eq!(frames[499], (starts[1], 499, false));
    assert_eq!(
        frames
            .iter()
            .map(|(_utterance_id, sequence, _eou)| *sequence)
            .collect::<Vec<_>>(),
        (0..500).collect::<Vec<_>>()
    );
}

#[test]
fn segmenter_allows_podcast_length_continuous_voice_before_forcing_eou() {
    let detector = ScriptedDetector::new(std::iter::repeat_n(true, 500));
    let mut segmenter =
        SpeechSegmenter::with_confirmation_and_max_frames(Uuid::new_v4(), detector, 3, 1_200);

    let events = process_events(&mut segmenter, 0..500);
    let starts = speech_start_ids(&events);
    let frames = frame_sequences(&events);

    assert_eq!(starts.len(), 1);
    assert_eq!(frames.len(), 500);
    assert_eq!(frames[0], (0, false));
    assert_eq!(frames[499], (499, false));
    assert!(
        frames
            .iter()
            .all(|(_sequence, end_of_utterance)| !end_of_utterance)
    );
}

#[test]
fn default_segmenter_forces_continuous_voice_at_six_seconds() {
    let detector = ScriptedDetector::new(std::iter::repeat_n(true, 320));
    let mut segmenter = SpeechSegmenter::new(Uuid::new_v4(), detector);

    let events = process_events(&mut segmenter, 0..320);
    let starts = speech_start_ids(&events);
    let frames = utterance_frames(&events);

    assert_eq!(starts.len(), 2);
    assert_ne!(starts[0], starts[1]);
    assert_eq!(frames.len(), 320);
    assert_eq!(frames[299], (starts[0], 299, true));
    assert_eq!(
        frames
            .iter()
            .map(|(_utterance_id, sequence, _eou)| *sequence)
            .collect::<Vec<_>>(),
        (0..320).collect::<Vec<_>>()
    );
}

#[test]
fn default_segmenter_closes_on_short_pause_after_minimum_live_window() {
    let detector =
        ScriptedDetector::new(std::iter::repeat_n(true, 125).chain(std::iter::repeat_n(false, 6)));
    let mut segmenter = SpeechSegmenter::new(Uuid::new_v4(), detector);

    let events = process_events(&mut segmenter, 0..131);
    let starts = speech_start_ids(&events);
    let frames = frame_sequences(&events);

    assert_eq!(starts.len(), 1);
    assert_eq!(frames.len(), 131);
    assert_eq!(frames[130], (130, true));
}

#[test]
fn segmenter_rearms_after_forced_continuous_voice_without_dropping_speech() {
    let detector = ScriptedDetector::new(std::iter::repeat_n(true, 406));
    let mut segmenter = low_latency_segmenter(detector);

    let events = process_events(&mut segmenter, 0..406);
    let starts = speech_start_ids(&events);
    assert_eq!(starts.len(), 2);
    assert_ne!(starts[0], starts[1]);

    let frames = utterance_frames(&events);
    assert_eq!(frames.len(), 406);
    assert_eq!(frames[399], (starts[0], 399, true));
    assert_eq!(
        &frames[400..],
        &[
            (starts[1], 400, false),
            (starts[1], 401, false),
            (starts[1], 402, false),
            (starts[1], 403, false),
            (starts[1], 404, false),
            (starts[1], 405, false),
        ]
    );
}

#[test]
fn segmenter_preserves_trailing_silence_at_the_forced_eou_boundary() {
    let detector = ScriptedDetector::new(std::iter::repeat_n(true, 398).chain([false, false]));
    let mut segmenter = low_latency_segmenter(detector);

    let events = process_events(&mut segmenter, 0..400);
    let frames = frame_sequences(&events);
    assert_eq!(
        frames,
        (0..400)
            .map(|sequence| (sequence, sequence == 399))
            .collect::<Vec<_>>()
    );
}

#[test]
fn segmenter_counts_an_internal_silence_pause_only_once() {
    let detector = ScriptedDetector::new(
        std::iter::repeat_n(true, 200)
            .chain(std::iter::repeat_n(false, 10))
            .chain(std::iter::repeat_n(true, 190)),
    );
    let mut segmenter = low_latency_segmenter(detector);

    let events = process_events(&mut segmenter, 0..400);
    let frames = frame_sequences(&events);
    assert_eq!(
        frames,
        (0..400)
            .map(|sequence| (sequence, sequence == 399))
            .collect::<Vec<_>>()
    );
}

#[test]
fn default_segmenter_rejects_short_noise_bursts_before_asr() {
    let detector = ScriptedDetector::new(
        std::iter::repeat_n(true, 9)
            .chain(std::iter::repeat_n(false, 15))
            .chain(std::iter::repeat_n(true, 10)),
    );
    let mut segmenter = SpeechSegmenter::new(Uuid::new_v4(), detector);

    let short_noise = process_events(&mut segmenter, 0..24);
    assert!(short_noise.is_empty());

    let confirmed_speech = process_events(&mut segmenter, 24..34);
    assert!(
        confirmed_speech
            .iter()
            .any(|event| matches!(event, CaptureEvent::SpeechStarted { .. }))
    );
}

#[test]
fn pcm_frame_rejects_wrong_byte_length() {
    let error =
        PcmFrame::try_new(0, 0, StreamPcmFormat::provider_default(), vec![0; 639]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "PCM frame byte length does not match its format"
    );
}
