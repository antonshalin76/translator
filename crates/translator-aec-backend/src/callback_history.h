#ifndef TRANSLATOR_AEC_CALLBACK_HISTORY_H
#define TRANSLATOR_AEC_CALLBACK_HISTORY_H

#include <spa/node/io.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

#define AEC_HISTORY_CAPACITY 16u

struct aec_callback_event {
    uint64_t callback, enter_ns, exit_ns, cpu_ns;
    uint64_t entry_position, admission_position, seq_position;
    uint64_t publication_position, next_source_position, next_position;
    uint64_t xrun, admission_xrun, publication_xrun, next_source_xrun;
    uint64_t raw_seq, reference_seq, output_seq, expected_seq, expected_position;
    uint64_t dsp_begin_ns, dsp_end_ns;
    uint32_t state, duration, clock_id, rate_num, rate_denom;
    uint32_t admission_duration, admission_clock, admission_rate_num, admission_rate_denom;
    uint32_t publication_duration, publication_clock, publication_rate_num, publication_rate_denom;
    uint32_t next_source_duration, next_source_clock, next_source_rate_num, next_source_rate_denom;
    uint32_t stage, reason, valid_mask, head, tail;
};

struct aec_callback_history {
    _Atomic bool freeze_requested;
    uint64_t total;
    struct aec_callback_event events[AEC_HISTORY_CAPACITY];
};

_Static_assert(3u * sizeof(struct aec_callback_history) <= 16384u,
               "three callback histories must fit within 16 KiB");

static uint64_t aec_history_time(clockid_t clock_id) {
    struct timespec now;
    if (clock_gettime(clock_id, &now) != 0 || now.tv_sec < 0) return 0;
    return (uint64_t)now.tv_sec * 1000000000u + (uint64_t)now.tv_nsec;
}

static bool aec_history_begin(struct aec_callback_history *history,
                              struct aec_callback_event *event,
                              uint64_t callback,
                              const struct spa_io_position *position) {
    if (atomic_load_explicit(&history->freeze_requested, memory_order_acquire))
        return false;
    *event = (struct aec_callback_event) { .callback = callback };
    event->enter_ns = aec_history_time(CLOCK_MONOTONIC);
    event->cpu_ns = aec_history_time(CLOCK_THREAD_CPUTIME_ID);
    if (position != NULL) {
        event->entry_position = position->clock.position;
        event->state = (uint32_t)position->state;
        event->duration = (uint32_t)position->clock.duration;
        event->clock_id = position->clock.id;
        event->rate_num = position->clock.rate.num;
        event->rate_denom = position->clock.rate.denom;
        event->xrun = position->clock.xrun;
    }
    return true;
}

static void aec_history_admission(struct aec_callback_event *event,
                                  const struct spa_io_clock *clock) {
    event->admission_xrun = clock->xrun;
    event->admission_duration = (uint32_t)clock->duration;
    event->admission_clock = clock->id;
    event->admission_rate_num = clock->rate.num;
    event->admission_rate_denom = clock->rate.denom;
}

static inline void aec_history_publication(struct aec_callback_event *event,
                                    const struct spa_io_clock *clock) {
    event->publication_xrun = clock->xrun;
    event->publication_duration = (uint32_t)clock->duration;
    event->publication_clock = clock->id;
    event->publication_rate_num = clock->rate.num;
    event->publication_rate_denom = clock->rate.denom;
}

static inline void aec_history_next_source(struct aec_callback_event *event,
                                    const struct spa_io_clock *clock) {
    event->next_source_xrun = clock->xrun;
    event->next_source_duration = (uint32_t)clock->duration;
    event->next_source_clock = clock->id;
    event->next_source_rate_num = clock->rate.num;
    event->next_source_rate_denom = clock->rate.denom;
}

static inline uint64_t aec_history_used_position(uint64_t value,
                                          struct aec_callback_event *event,
                                          bool active) {
    if (active) event->admission_position = value;
    return value;
}

static void aec_history_fail(struct aec_callback_history *history) {
    atomic_store_explicit(&history->freeze_requested, true, memory_order_release);
}

static void aec_history_end(struct aec_callback_history *history,
                            struct aec_callback_event *event, bool active) {
    if (!active) return;
    uint64_t cpu_end = aec_history_time(CLOCK_THREAD_CPUTIME_ID);
    event->exit_ns = aec_history_time(CLOCK_MONOTONIC);
    event->cpu_ns = event->cpu_ns > 0 && cpu_end >= event->cpu_ns ?
        cpu_end - event->cpu_ns : 0;
    history->events[history->total % AEC_HISTORY_CAPACITY] = *event;
    history->total++;
}

/* Called only after the owning filter and writer have stopped. */
static void aec_history_report(const struct aec_callback_history *history,
                               uint64_t session, const char *component) {
    struct stat time_namespace;
    if (stat("/proc/self/ns/time", &time_namespace) != 0) return;
    uint64_t count = history->total < AEC_HISTORY_CAPACITY ?
        history->total : AEC_HISTORY_CAPACITY;
    fprintf(stderr,
            "AEC_CALLBACK_HISTORY session=%llu component=%s clock=CLOCK_MONOTONIC time_ns=%llu total=%llu count=%llu frozen=%u\n",
            (unsigned long long)session, component,
            (unsigned long long)time_namespace.st_ino,
            (unsigned long long)history->total, (unsigned long long)count,
            atomic_load_explicit(&history->freeze_requested, memory_order_acquire) ? 1u : 0u);
    for (uint64_t i = 0; i < count; i++) {
        const struct aec_callback_event *e =
            &history->events[(history->total - count + i) % AEC_HISTORY_CAPACITY];
        fprintf(stderr,
                "AEC_CALLBACK_TRACE session=%llu component=%s index=%llu callback=%llu enter=%llu exit=%llu cpu=%llu entry=%llu admission=%llu seq_position=%llu publication=%llu next_source=%llu next=%llu xrun=%llu admission_xrun=%llu publication_xrun=%llu next_source_xrun=%llu raw=%llu ref=%llu output=%llu expected_seq=%llu expected_pos=%llu dsp_begin=%llu dsp_end=%llu state=%u duration=%u clock=%u rate_num=%u rate_denom=%u admission_duration=%u admission_clock=%u admission_rate_num=%u admission_rate_denom=%u publication_duration=%u publication_clock=%u publication_rate_num=%u publication_rate_denom=%u next_source_duration=%u next_source_clock=%u next_source_rate_num=%u next_source_rate_denom=%u stage=%u reason=%u valid=%u head=%u tail=%u\n",
                (unsigned long long)session, component, (unsigned long long)i,
                (unsigned long long)e->callback, (unsigned long long)e->enter_ns,
                (unsigned long long)e->exit_ns, (unsigned long long)e->cpu_ns,
                (unsigned long long)e->entry_position,
                (unsigned long long)e->admission_position,
                (unsigned long long)e->seq_position,
                (unsigned long long)e->publication_position,
                (unsigned long long)e->next_source_position,
                (unsigned long long)e->next_position, (unsigned long long)e->xrun,
                (unsigned long long)e->admission_xrun,
                (unsigned long long)e->publication_xrun,
                (unsigned long long)e->next_source_xrun,
                (unsigned long long)e->raw_seq, (unsigned long long)e->reference_seq,
                (unsigned long long)e->output_seq, (unsigned long long)e->expected_seq,
                (unsigned long long)e->expected_position,
                (unsigned long long)e->dsp_begin_ns, (unsigned long long)e->dsp_end_ns,
                e->state, e->duration, e->clock_id, e->rate_num, e->rate_denom,
                e->admission_duration, e->admission_clock,
                e->admission_rate_num, e->admission_rate_denom,
                e->publication_duration, e->publication_clock,
                e->publication_rate_num, e->publication_rate_denom,
                e->next_source_duration, e->next_source_clock,
                e->next_source_rate_num, e->next_source_rate_denom,
                e->stage, e->reason, e->valid_mask, e->head, e->tail);
    }
}

#endif
