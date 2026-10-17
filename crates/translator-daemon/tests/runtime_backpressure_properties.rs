use std::collections::VecDeque;

use proptest::{prelude::*, test_runner::RngSeed};
use translator_audio::{BoundedPcmQueue, PcmFrame, StreamPcmFormat};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        max_shrink_iters: 512,
        rng_seed: RngSeed::Fixed(0x2026_1017),
        ..ProptestConfig::default()
    })]

    #[test]
    fn runtime_backpressure_preserves_bounded_fifo_under_arbitrary_produce_consume(
        actions in proptest::collection::vec(prop_oneof![4 => Just(true), 1 => Just(false)], 0..=64)
    ) {
        let format = StreamPcmFormat::provider_default();
        let mut queue = BoundedPcmQueue::default();
        let mut accepted = VecDeque::new();
        for (sequence, produce) in actions.into_iter().enumerate() {
            prop_assert_eq!(queue.has_capacity_for(format), accepted.len() < 20);
            if produce && queue.has_capacity_for(format) {
                queue.push(PcmFrame::try_new(sequence as u64, 0, format,
                    vec![0; format.frame_bytes()]).unwrap()).unwrap();
                accepted.push_back(sequence as u64);
            } else if !produce {
                prop_assert_eq!(queue.pop().map(|frame| frame.sequence()), accepted.pop_front());
            }
            prop_assert!(queue.buffered_ms() <= 400);
            prop_assert_eq!(queue.dropped_frames(), 0);
            prop_assert_eq!(queue.len(), accepted.len());
        }
        while let Some(expected) = accepted.pop_front() {
            prop_assert_eq!(queue.pop().unwrap().sequence(), expected);
        }
        prop_assert!(queue.is_empty());
        prop_assert_eq!(queue.buffered_ms(), 0);
    }
}
