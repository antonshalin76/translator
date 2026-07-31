use translator_core::AudioDirection;
use translator_daemon::{DaemonQueues, QueueConsumeResult, QueueKind, QueuePushResult};

#[test]
fn capture_and_playback_queues_are_bounded_by_400_buffered_ms() {
    let mut queues = DaemonQueues::default();
    for _ in 0..10 {
        assert_eq!(
            queues.push(AudioDirection::Microphone, QueueKind::Capture, 40),
            QueuePushResult::Accepted
        );
    }
    assert_eq!(
        queues.push(AudioDirection::Microphone, QueueKind::Capture, 20),
        QueuePushResult::RejectedOverflow
    );
    let state = queues.state(AudioDirection::Microphone, QueueKind::Capture);
    assert_eq!(state.buffered_ms, 400);
    assert_eq!(state.dropped_frames, 1);
}

#[test]
fn queue_capacity_is_independent_of_frame_count_and_other_direction() {
    let mut queues = DaemonQueues::default();
    for _ in 0..4 {
        assert_eq!(
            queues.push(AudioDirection::Microphone, QueueKind::Playback, 100),
            QueuePushResult::Accepted
        );
    }
    for _ in 0..20 {
        assert_eq!(
            queues.push(AudioDirection::Speaker, QueueKind::Playback, 20),
            QueuePushResult::Accepted
        );
    }
    assert_eq!(
        queues
            .state(AudioDirection::Microphone, QueueKind::Playback)
            .buffered_ms,
        400
    );
    assert_eq!(
        queues
            .state(AudioDirection::Speaker, QueueKind::Playback)
            .buffered_ms,
        400
    );
}

#[test]
fn consuming_frames_releases_duration_and_rejects_underflow_without_mutation() {
    let mut queues = DaemonQueues::default();
    queues.push(AudioDirection::Speaker, QueueKind::Capture, 100);
    assert_eq!(
        queues.consume(AudioDirection::Speaker, QueueKind::Capture, 40),
        QueueConsumeResult::Consumed
    );
    assert_eq!(
        queues
            .state(AudioDirection::Speaker, QueueKind::Capture)
            .buffered_ms,
        60
    );
    assert_eq!(
        queues.consume(AudioDirection::Speaker, QueueKind::Capture, 100),
        QueueConsumeResult::RejectedUnderflow
    );
    assert_eq!(
        queues
            .state(AudioDirection::Speaker, QueueKind::Capture)
            .buffered_ms,
        60
    );
}

#[test]
fn capture_and_playback_ledgers_are_independent_within_one_direction() {
    let mut queues = DaemonQueues::default();
    queues.push(AudioDirection::Speaker, QueueKind::Capture, 100);
    queues.push(AudioDirection::Speaker, QueueKind::Playback, 20);
    assert_eq!(
        queues
            .state(AudioDirection::Speaker, QueueKind::Capture)
            .buffered_ms,
        100
    );
    assert_eq!(
        queues
            .state(AudioDirection::Speaker, QueueKind::Playback)
            .buffered_ms,
        20
    );
}
