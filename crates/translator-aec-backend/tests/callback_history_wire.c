#define _POSIX_C_SOURCE 200809L
#include "../src/callback_history.h"

int main(void) {
    struct aec_callback_history history = { 0 };
    struct spa_io_position position = { 0 };
    position.clock.id = 3;
    position.clock.duration = 480;
    position.clock.rate.num = 1;
    position.clock.rate.denom = 48000;
    for (uint64_t index = 1; index <= 18; index++) {
        position.clock.position = index * 480;
        struct aec_callback_event event;
        bool active = aec_history_begin(&history, &event, index, &position);
        if (!active) return 2;
        aec_history_admission(&event, &position.clock);
        event.admission_position = position.clock.position;
        event.seq_position = position.clock.position;
        event.raw_seq = index;
        event.reference_seq = index;
        event.output_seq = index;
        event.expected_seq = index;
        event.expected_position = position.clock.position;
        event.dsp_begin_ns = aec_history_time(CLOCK_MONOTONIC);
        event.dsp_end_ns = aec_history_time(CLOCK_MONOTONIC);
        aec_history_publication(&event, &position.clock);
        event.publication_position = position.clock.position;
        if (index == 18) {
            position.clock.position += 480;
            position.clock.duration = 481;
            position.clock.xrun = 19;
            position.clock.id = 23;
            position.clock.rate.num = 2;
            position.clock.rate.denom = 96000;
        }
        aec_history_next_source(&event, &position.clock);
        event.next_source_position = position.clock.position;
        event.next_position = position.clock.position + 480;
        event.valid_mask = 7;
        event.stage = 4;
        if (index == 18) {
            event.reason = 4;
            aec_history_fail(&history);
        }
        aec_history_end(&history, &event, true);
    }
    aec_history_report(&history, 12, "backend");
    return 0;
}
