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
    SpeechSegmenter::with_confirmation_frames(Uuid::new_v4(), detector, 3)
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

    let first = (0..18)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
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

    let second = (18..21)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
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
fn segmenter_bounds_continuous_voice_and_requires_silence_before_rearming() {
    let detector = ScriptedDetector::new(
        std::iter::repeat_n(true, 500)
            .chain(std::iter::repeat_n(false, 14))
            .chain(std::iter::repeat_n(true, 3))
            .chain(std::iter::repeat_n(false, 15))
            .chain([true, true, true]),
    );
    let mut segmenter = low_latency_segmenter(detector);

    let continuous_voice = (0..500)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
    let first_id = continuous_voice
        .iter()
        .find_map(|event| match event {
            CaptureEvent::SpeechStarted { utterance_id, .. } => Some(*utterance_id),
            _ => None,
        })
        .unwrap();
    let first_frames = continuous_voice
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
        .collect::<Vec<_>>();
    assert_eq!(first_frames.len(), 400);
    assert_eq!(
        first_frames,
        (0..400)
            .map(|sequence| (first_id, sequence, sequence == 399))
            .collect::<Vec<_>>()
    );

    let insufficient_silence = (500..514)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
    assert!(insufficient_silence.is_empty());

    let interrupted_silence = (514..517)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
    assert!(interrupted_silence.is_empty());

    let rearm_silence = (517..532)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
    assert!(rearm_silence.is_empty());

    let rearmed = (532..535)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
    let second_id = rearmed
        .iter()
        .find_map(|event| match event {
            CaptureEvent::SpeechStarted { utterance_id, .. } => Some(*utterance_id),
            _ => None,
        })
        .unwrap();
    assert_ne!(first_id, second_id);
}

#[test]
fn segmenter_preserves_trailing_silence_at_the_forced_eou_boundary() {
    let detector = ScriptedDetector::new(std::iter::repeat_n(true, 398).chain([false, false]));
    let mut segmenter = low_latency_segmenter(detector);

    let events = (0..400)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
    let frames = events
        .iter()
        .filter_map(|event| match event {
            CaptureEvent::Frame {
                frame,
                end_of_utterance,
                ..
            } => Some((frame.sequence(), *end_of_utterance)),
            CaptureEvent::SpeechStarted { .. } => None,
        })
        .collect::<Vec<_>>();
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

    let events = (0..400)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
    let frames = events
        .iter()
        .filter_map(|event| match event {
            CaptureEvent::Frame {
                frame,
                end_of_utterance,
                ..
            } => Some((frame.sequence(), *end_of_utterance)),
            CaptureEvent::SpeechStarted { .. } => None,
        })
        .collect::<Vec<_>>();
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

    let short_noise = (0..24)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
    assert!(short_noise.is_empty());

    let confirmed_speech = (24..34)
        .flat_map(|sequence| segmenter.process(frame(sequence)).unwrap())
        .collect::<Vec<_>>();
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
