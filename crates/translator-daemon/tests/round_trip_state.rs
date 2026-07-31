use translator_audio::{PcmFrame, StreamPcmFormat};
use translator_daemon::{
    ExactPcmEvidence, RoundTripCheckpoint, RoundTripErrorCode, RoundTripLatency,
    RoundTripPreconditions, RoundTripSelfTest,
};

fn ready() -> RoundTripPreconditions {
    RoundTripPreconditions {
        headphones: true,
        outgoing_provider_ready: true,
        incoming_provider_ready: true,
        virtual_graph_ready: true,
        incoming_route_idle: true,
    }
}

fn pcm_frame(sequence: u64, value: u8) -> PcmFrame {
    PcmFrame::try_new(
        sequence,
        1_000_000_000 + sequence * 20_000_000,
        StreamPcmFormat::provider_default(),
        vec![value; StreamPcmFormat::provider_default().frame_bytes()],
    )
    .unwrap()
}

fn matching_proof() -> translator_daemon::ExactPcmProof {
    let mut evidence = ExactPcmEvidence::new(StreamPcmFormat::provider_default());
    for frame in [pcm_frame(0, 7), pcm_frame(1, 9)] {
        evidence.capture(&frame).unwrap();
        evidence.reinject(&frame).unwrap();
    }
    evidence.proof()
}

#[test]
fn exact_pcm_evidence_requires_identical_format_sequence_count_and_hash() {
    let mut evidence = ExactPcmEvidence::new(StreamPcmFormat::provider_default());
    let frames = [pcm_frame(0, 7), pcm_frame(1, 9)];

    for frame in &frames {
        evidence.capture(frame).unwrap();
    }
    for frame in &frames {
        evidence.reinject(frame).unwrap();
    }

    let proof = evidence.proof();
    assert_eq!(proof.frame_count, 2);
    assert_eq!(proof.captured_frame_count, 2);
    assert_eq!(proof.reinjected_frame_count, 2);
    assert_eq!(proof.first_sequence, Some(0));
    assert_eq!(proof.last_sequence, Some(1));
    assert_eq!(proof.captured_rolling_hash, proof.reinjected_rolling_hash);
    assert!(proof.format_matches);
    assert!(proof.sequence_monotonic);
    assert!(proof.exact_match);
}

#[test]
fn exact_pcm_evidence_fails_closed_on_sequence_or_payload_mismatch() {
    let mut evidence = ExactPcmEvidence::new(StreamPcmFormat::provider_default());
    evidence.capture(&pcm_frame(0, 7)).unwrap();
    evidence.capture(&pcm_frame(1, 9)).unwrap();
    evidence.reinject(&pcm_frame(0, 7)).unwrap();

    assert!(evidence.reinject(&pcm_frame(2, 9)).is_err());
    let proof = evidence.proof();
    assert!(!proof.sequence_monotonic);
    assert!(!proof.exact_match);

    let mut payload_mismatch = ExactPcmEvidence::new(StreamPcmFormat::provider_default());
    payload_mismatch.capture(&pcm_frame(0, 7)).unwrap();
    payload_mismatch.reinject(&pcm_frame(0, 8)).unwrap();
    assert!(!payload_mismatch.proof().exact_match);
}

fn advance_to_english_audio(self_test: &mut RoundTripSelfTest, session_id: uuid::Uuid) {
    for checkpoint in [
        RoundTripCheckpoint::OutgoingVad,
        RoundTripCheckpoint::OutgoingAsrFinal,
        RoundTripCheckpoint::OutgoingTranslationFinal,
        RoundTripCheckpoint::EnglishFirstAudio,
    ] {
        self_test
            .advance(session_id, checkpoint, RoundTripLatency::default())
            .unwrap();
    }
}

#[test]
fn start_rejects_open_speaker_route_conflict_and_unavailable_provider() {
    let mut self_test = RoundTripSelfTest::default();

    let mut preconditions = ready();
    preconditions.headphones = false;
    assert_eq!(
        self_test.start(preconditions, 0).unwrap_err(),
        RoundTripErrorCode::HeadphonesRequired
    );
    let mut preconditions = ready();
    preconditions.incoming_route_idle = false;
    assert_eq!(
        self_test.start(preconditions, 0).unwrap_err(),
        RoundTripErrorCode::IncomingRouteConflict
    );
    let mut preconditions = ready();
    preconditions.incoming_provider_ready = false;
    assert_eq!(
        self_test.start(preconditions, 0).unwrap_err(),
        RoundTripErrorCode::ProviderUnavailable
    );
    let mut preconditions = ready();
    preconditions.outgoing_provider_ready = false;
    assert_eq!(
        self_test.start(preconditions, 0).unwrap_err(),
        RoundTripErrorCode::ProviderUnavailable
    );
    let mut preconditions = ready();
    preconditions.virtual_graph_ready = false;
    assert_eq!(
        self_test.start(preconditions, 0).unwrap_err(),
        RoundTripErrorCode::VirtualGraphUnavailable
    );
}

#[test]
fn only_one_session_runs_and_timeout_stops_it_at_five_minutes() {
    let mut self_test = RoundTripSelfTest::default();
    let session_id = self_test.start(ready(), 10).unwrap();
    assert_eq!(
        self_test.status(false).checkpoint,
        Some(RoundTripCheckpoint::WaitingForSpeech)
    );
    assert_eq!(
        self_test.start(ready(), 11).unwrap_err(),
        RoundTripErrorCode::AlreadyRunning
    );
    assert!(!self_test.expire(300_009));
    assert!(self_test.expire(300_010));
    let status = self_test.status(false);
    assert_eq!(status.session_id, Some(session_id));
    assert_eq!(status.checkpoint, Some(RoundTripCheckpoint::Stopped));
    assert_eq!(status.safe_error, Some(RoundTripErrorCode::Timeout));
}

#[test]
fn recursion_count_is_retained_and_resets_for_the_next_self_test() {
    let mut self_test = RoundTripSelfTest::default();
    let first_session_id = self_test.start(ready(), 0).unwrap();

    self_test
        .record_recursion_trigger(first_session_id)
        .unwrap();
    self_test
        .record_recursion_trigger(first_session_id)
        .unwrap();
    assert_eq!(self_test.status(false).recursion_count, 2);
    assert!(self_test.stop(first_session_id));
    assert_eq!(self_test.status(false).recursion_count, 2);

    let second_session_id = self_test.start(ready(), 1).unwrap();
    assert_ne!(second_session_id, first_session_id);
    assert_eq!(self_test.status(false).recursion_count, 0);

    let serialized = serde_json::to_value(self_test.status(false)).unwrap();
    assert_eq!(serialized["recursion_count"], 0);
}

#[test]
fn checkpoints_and_latency_are_typed_but_capability_and_text_are_hidden() {
    let marker = "private-round-trip-text";
    let mut self_test = RoundTripSelfTest::default();
    let session_id = self_test.start(ready(), 0).unwrap();
    for checkpoint in [
        RoundTripCheckpoint::OutgoingVad,
        RoundTripCheckpoint::OutgoingAsrFinal,
        RoundTripCheckpoint::OutgoingTranslationFinal,
    ] {
        self_test
            .advance(session_id, checkpoint, RoundTripLatency::default())
            .unwrap();
    }
    self_test
        .advance(
            session_id,
            RoundTripCheckpoint::EnglishFirstAudio,
            RoundTripLatency {
                outgoing_first_audio_ms: Some(740),
                english_monitor_complete_ms: Some(1_200),
                incoming_first_audio_ms: None,
                physical_mic_onset_to_returned_ru_first_audible_ms: None,
            },
        )
        .unwrap();
    self_test
        .set_debug_text(session_id, marker, "translation")
        .unwrap();

    let safe = serde_json::to_value(self_test.status(false)).unwrap();
    assert_eq!(safe["checkpoint"], "english_first_audio");
    assert_eq!(safe["latency"]["outgoing_first_audio_ms"], 740);
    assert!(safe.get("debug_text").is_none());
    let serialized = safe.to_string();
    assert!(!serialized.contains(marker));
    assert!(!serialized.contains("stream_serial"));
    assert!(!serialized.contains("process_identity"));

    let debug = serde_json::to_value(self_test.status(true)).unwrap();
    assert_eq!(debug["debug_text"]["transcript"], marker);
}

#[test]
fn virtual_peer_reinjection_requires_completed_english_monitor_tap() {
    let mut self_test = RoundTripSelfTest::default();
    let session_id = self_test.start(ready(), 0).unwrap();
    advance_to_english_audio(&mut self_test, session_id);
    assert_eq!(
        self_test
            .advance(
                session_id,
                RoundTripCheckpoint::VirtualPeerReinjecting,
                RoundTripLatency {
                    outgoing_first_audio_ms: Some(700),
                    english_monitor_complete_ms: None,
                    incoming_first_audio_ms: None,
                    physical_mic_onset_to_returned_ru_first_audible_ms: None,
                }
            )
            .unwrap_err(),
        RoundTripErrorCode::InvalidCheckpoint
    );
    self_test
        .set_exact_pcm_proof(session_id, matching_proof())
        .unwrap();
    self_test
        .advance(
            session_id,
            RoundTripCheckpoint::VirtualPeerReinjecting,
            RoundTripLatency {
                outgoing_first_audio_ms: Some(700),
                english_monitor_complete_ms: Some(1_100),
                incoming_first_audio_ms: None,
                physical_mic_onset_to_returned_ru_first_audible_ms: None,
            },
        )
        .unwrap();
}

#[test]
fn checkpoints_are_linear_and_terminal_state_is_immutable() {
    let mut self_test = RoundTripSelfTest::default();
    let session_id = self_test.start(ready(), 0).unwrap();
    assert_eq!(
        self_test
            .advance(
                session_id,
                RoundTripCheckpoint::OutgoingAsrFinal,
                RoundTripLatency::default()
            )
            .unwrap_err(),
        RoundTripErrorCode::InvalidCheckpoint
    );
    advance_to_english_audio(&mut self_test, session_id);
    self_test
        .set_exact_pcm_proof(session_id, matching_proof())
        .unwrap();
    self_test
        .advance(
            session_id,
            RoundTripCheckpoint::VirtualPeerReinjecting,
            RoundTripLatency {
                english_monitor_complete_ms: Some(1_000),
                ..RoundTripLatency::default()
            },
        )
        .unwrap();
    for checkpoint in [
        RoundTripCheckpoint::IncomingAsrFinal,
        RoundTripCheckpoint::IncomingTranslationFinal,
        RoundTripCheckpoint::RussianFirstAudio,
        RoundTripCheckpoint::Completed,
    ] {
        self_test
            .advance(session_id, checkpoint, RoundTripLatency::default())
            .unwrap();
    }
    assert_eq!(
        self_test
            .advance(
                session_id,
                RoundTripCheckpoint::IncomingAsrFinal,
                RoundTripLatency::default()
            )
            .unwrap_err(),
        RoundTripErrorCode::NotRunning
    );
}

#[test]
fn stale_session_cannot_mutate_new_session_and_timeout_clears_debug_text() {
    let marker = "private-timeout-marker";
    let mut self_test = RoundTripSelfTest::default();
    let old_session = self_test.start(ready(), 0).unwrap();
    self_test
        .set_debug_text(old_session, marker, marker)
        .unwrap();
    assert!(self_test.expire(300_000));
    assert!(
        !serde_json::to_string(&self_test.status(true))
            .unwrap()
            .contains(marker)
    );
    let new_session = self_test.start(ready(), 300_001).unwrap();
    assert_ne!(old_session, new_session);
    assert!(!self_test.stop(old_session));
    assert_eq!(
        self_test.status(false).checkpoint,
        Some(RoundTripCheckpoint::WaitingForSpeech)
    );
    assert_eq!(
        self_test
            .advance(
                old_session,
                RoundTripCheckpoint::OutgoingVad,
                RoundTripLatency::default()
            )
            .unwrap_err(),
        RoundTripErrorCode::SessionMismatch
    );
}

#[test]
fn stop_is_idempotent_and_stale_session_cannot_advance() {
    let mut self_test = RoundTripSelfTest::default();
    let session_id = self_test.start(ready(), 0).unwrap();
    assert!(self_test.stop(session_id));
    assert!(!self_test.stop(session_id));
    assert_eq!(
        self_test
            .advance(
                session_id,
                RoundTripCheckpoint::RussianFirstAudio,
                RoundTripLatency::default()
            )
            .unwrap_err(),
        RoundTripErrorCode::NotRunning
    );
}
