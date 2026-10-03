#define pw_filter_dequeue_buffer test_filter_dequeue_buffer
#define pw_filter_queue_buffer test_filter_queue_buffer
#include "../src/native.c"
#include <assert.h>

static uint64_t ticks(const struct timespec *value) {
    return (uint64_t)value->tv_sec * 1000000000u + (uint64_t)value->tv_nsec;
}

static void capture_frame_report(struct backend *b, char *log, size_t capacity) {
    int errors[2];
    assert(pipe(errors) == 0);
    int saved_stderr = dup(STDERR_FILENO);
    assert(saved_stderr >= 0);
    assert(dup2(errors[1], STDERR_FILENO) >= 0);
    close(errors[1]);
    report_frame_state(b);
    fflush(stderr);
    assert(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);
    ssize_t got = read(errors[0], log, capacity - 1);
    assert(got > 0);
    log[got] = '\0';
    close(errors[0]);
}

static void test_callback_history(void) {
    struct aec_callback_history history = { 0 };
    struct spa_io_position position = { 0 };
    position.clock.duration = QUANTUM;
    position.clock.id = 3;
    for (uint64_t callback = 1; callback <= 34; callback++) {
        position.clock.position = callback * QUANTUM;
        struct aec_callback_event event;
        assert(aec_history_begin(&history, &event, callback, &position));
        event.admission_position = position.clock.position;
        if (callback == 34) {
            aec_history_fail(&history);
            event.reason = FAIL_GAP;
        }
        aec_history_end(&history, &event, true);
    }
    assert(history.total == 34);
    assert(history.events[(34 - 16) % AEC_HISTORY_CAPACITY].callback == 19);
    assert(history.events[(34 - 1) % AEC_HISTORY_CAPACITY].callback == 34);
    struct aec_callback_event rejected;
    assert(!aec_history_begin(&history, &rejected, 35, &position));
    aec_history_end(&history, &rejected, false);
    assert(history.total == 34);
    assert(history.events[(34 - 1) % AEC_HISTORY_CAPACITY].reason == FAIL_GAP);
    FILE *diagnostic = tmpfile();
    assert(diagnostic != NULL);
    int saved = dup(STDERR_FILENO);
    assert(saved >= 0 && dup2(fileno(diagnostic), STDERR_FILENO) >= 0);
    aec_history_report(&history, 12, "backend");
    fflush(stderr);
    assert(dup2(saved, STDERR_FILENO) >= 0);
    close(saved);
    assert(fseek(diagnostic, 0, SEEK_SET) == 0);
    char output[16384] = { 0 };
    size_t got = fread(output, 1, sizeof(output) - 1, diagnostic);
    assert(got > 0 && feof(diagnostic));
    assert(strstr(output, "total=34 count=16 frozen=1") != NULL);
    assert(strstr(output, "index=0 callback=19") != NULL);
    assert(strstr(output, "index=15 callback=34") != NULL);
    assert(strstr(output, "callback=18") == NULL);
    fclose(diagnostic);
}


struct synthetic_buffer {
    float samples[QUANTUM];
    struct spa_chunk chunk;
    struct spa_data data;
    struct spa_meta_header header;
    struct spa_meta meta;
    struct spa_buffer buffer;
    struct pw_buffer pw;
};

static struct synthetic_buffer synthetic_buffers[3];
static struct spa_io_position *synthetic_live_position;
static bool synthetic_change_clock;
static void *synthetic_fail_return_port;

struct pw_buffer *test_filter_dequeue_buffer(void *port) {
    uintptr_t index = (uintptr_t)port;
    assert(index >= 1 && index <= 3);
    return &synthetic_buffers[index - 1].pw;
}

int test_filter_queue_buffer(void *port, struct pw_buffer *buffer) {
    uintptr_t index = (uintptr_t)port;
    assert(index >= 1 && index <= 3);
    assert(buffer == &synthetic_buffers[index - 1].pw);
    return port == synthetic_fail_return_port ? -1 : 0;
}

static int synthetic_run(void *object, const float *rec[], const float *play[],
                         float *out[], uint32_t samples) {
    (void)object;
    (void)play;
    assert(samples == QUANTUM);
    memcpy(out[0], rec[0], samples * sizeof(float));
    if (synthetic_change_clock) {
        synthetic_live_position->clock.position += QUANTUM;
        synthetic_live_position->clock.duration = QUANTUM + 1;
        synthetic_live_position->clock.xrun = 19;
        synthetic_live_position->clock.id = 23;
        synthetic_live_position->clock.rate.num = 2;
        synthetic_live_position->clock.rate.denom = RATE * 2;
    }
    return 0;
}

static void initialize_synthetic_buffer(struct synthetic_buffer *item,
                                        uint64_t sequence, bool output) {
    *item = (struct synthetic_buffer) { 0 };
    item->chunk = (struct spa_chunk) {
        .size = QUANTUM * sizeof(float), .stride = sizeof(float),
    };
    item->data = (struct spa_data) {
        .flags = output ? SPA_DATA_FLAG_WRITABLE : SPA_DATA_FLAG_READABLE,
        .maxsize = sizeof(item->samples), .data = item->samples, .chunk = &item->chunk,
    };
    item->header.seq = sequence;
    item->meta = (struct spa_meta) {
        .type = SPA_META_Header, .size = sizeof(item->header), .data = &item->header,
    };
    item->buffer = (struct spa_buffer) {
        .n_metas = 1, .n_datas = 1, .metas = &item->meta, .datas = &item->data,
    };
    item->pw.buffer = &item->buffer;
}

static void test_callback_fidelity(bool mutate_clock, bool fail_recycle) {
    struct backend b = { 0 };
    struct spa_io_position position = { 0 };
    static const struct spa_audio_aec_methods methods = {
        .version = SPA_VERSION_AUDIO_AEC_METHODS, .run = synthetic_run,
    };
    struct spa_audio_aec aec = {
        .iface = SPA_INTERFACE_INIT(SPA_TYPE_INTERFACE_AUDIO_AEC,
                                    SPA_VERSION_AUDIO_AEC, &methods, NULL),
    };
    for (unsigned i = 0; i < 3; i++)
        initialize_synthetic_buffer(&synthetic_buffers[i], 0, i == 2);
    b.raw_port = (void *)(uintptr_t)1;
    b.reference_port = (void *)(uintptr_t)2;
    b.clean_port = (void *)(uintptr_t)3;
    b.aec = &aec;
    atomic_store(&b.started, true);
    position.clock.duration = QUANTUM;
    position.clock.id = 15;
    position.clock.rate.num = 1;
    position.clock.rate.denom = RATE;
    synthetic_live_position = &position;
    synthetic_change_clock = mutate_clock;
    synthetic_fail_return_port = fail_recycle ? b.raw_port : NULL;

    on_process(&b, &position);
    assert(b.history.total == 1);
    const struct aec_callback_event *event = &b.history.events[0];
    const struct frame *frame = &b.frames[0];
    assert(event->stage == 4);
    assert(event->publication_position == frame->position);
    assert(event->publication_duration == frame->duration);
    assert(event->publication_xrun == frame->xrun);
    assert(event->publication_clock == frame->clock_id);
    assert(event->publication_rate_num == frame->rate_num);
    assert(event->publication_rate_denom == frame->rate_denom);
    assert(event->reason == (fail_recycle ? FAIL_BUFFER : FAIL_NONE));
    assert(atomic_load(&b.failure) == (fail_recycle ? FAIL_BUFFER : FAIL_NONE));
    assert(atomic_load(&b.history.freeze_requested) == fail_recycle);
    assert(event->next_source_duration == position.clock.duration);
    assert(event->next_source_xrun == position.clock.xrun);
    assert(event->next_source_clock == position.clock.id);
    assert(event->next_source_rate_num == position.clock.rate.num);
    assert(event->next_source_rate_denom == position.clock.rate.denom);
    if (mutate_clock) {
        assert(event->next_source_position == QUANTUM);
        assert(event->publication_position != event->next_source_position);
        assert(event->publication_clock != event->next_source_clock);
    }
    if (fail_recycle) {
        assert(b.snapshot.reason == FAIL_BUFFER);
        assert(b.snapshot.callback == 1);
        on_process(&b, &position);
        assert(b.history.total == 1);
        assert(b.history.events[0].reason == FAIL_BUFFER);
        assert(b.history.events[0].stage == 4);
    }
    synthetic_change_clock = false;
    synthetic_fail_return_port = NULL;
}

int main(void) {
    test_callback_fidelity(false, true);
    test_callback_fidelity(true, false);
    test_callback_fidelity(false, false);
    test_callback_history();
    struct backend b = { 0 };
    b.session = 0x0123456789abcdefull;
    struct spa_chunk chunk = { .size = 123, .stride = 4, .flags = 0 };
    struct spa_data data = { .chunk = &chunk };
    struct spa_buffer spa = { .n_datas = 1, .datas = &data };
    struct pw_buffer pw = { .buffer = &spa };

    atomic_store(&b.callback_count, 17);
    atomic_store(&b.last_position_tick, 123456);
    struct timespec outside_before, outside_after;
    assert(clock_gettime(CLOCK_MONOTONIC, &outside_before) == 0);
    poison_at(&b, FAIL_BUFFER, 101);
    assert(clock_gettime(CLOCK_MONOTONIC, &outside_after) == 0);
    uint64_t first_before = b.first_failure_before_ns;
    uint64_t first_after = b.first_failure_after_ns;
    assert(ticks(&outside_before) <= first_before);
    assert(first_before > 0 && first_before <= first_after);
    assert(first_after <= ticks(&outside_after));
    assert(atomic_load_explicit(&b.snapshot_ready, memory_order_acquire));
    assert(b.snapshot.reason == FAIL_BUFFER);
    assert(b.snapshot.callback == 17);
    capture_input_metadata(&b, 0, &pw);
    assert(atomic_load(&b.failure) == FAIL_BUFFER);
    assert(atomic_load(&b.bad_buffer_site) == 101);
    assert(atomic_load(&b.bad_buffer_callback) == 17);
    assert(atomic_load(&b.bad_buffer_position) == 123456);
    assert(atomic_load(&b.bad_input_chunk_size[0]) == 123);

    chunk.size = 456;
    atomic_store(&b.callback_count, 18);
    atomic_store(&b.last_position_tick, 123936);
    poison_at(&b, FAIL_BUFFER, 202);
    capture_input_metadata(&b, 0, &pw);
    assert(b.first_failure_before_ns == first_before);
    assert(b.first_failure_after_ns == first_after);
    assert(b.snapshot.reason == FAIL_BUFFER);
    assert(b.snapshot.callback == 17);
    assert(atomic_load(&b.bad_buffer_site) == 101);
    assert(atomic_load(&b.bad_buffer_callback) == 17);
    assert(atomic_load(&b.bad_buffer_position) == 123456);
    assert(atomic_load(&b.bad_input_chunk_size[0]) == 123);

    int ipc[2], errors[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, ipc) == 0);
    assert(pipe(errors) == 0);
    int saved_stderr = dup(STDERR_FILENO);
    assert(saved_stderr >= 0);
    assert(dup2(errors[1], STDERR_FILENO) >= 0);
    close(errors[1]);
    b.ipc_fd = ipc[0];
    writer_main(&b);
    fflush(stderr);
    assert(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);
    char log[1024] = { 0 };
    ssize_t got = read(errors[0], log, sizeof(log) - 1);
    assert(got > 0);
    assert(strstr(log, "AEC_BUFFER_PRECONDITION site=101") != NULL);
    assert(strstr(log, "callback=17 position=123456") != NULL);
    close(errors[0]);
    close(ipc[0]);
    close(ipc[1]);

    struct stat time_namespace;
    assert(stat("/proc/self/ns/time", &time_namespace) == 0);
    memset(log, 0, sizeof(log));
    capture_frame_report(&b, log, sizeof(log));
    assert(strstr(log, "AEC_FRAME_STATE session=81985529216486895 failed=1 reason=4") != NULL);
    char expected[256];
    int length = snprintf(expected, sizeof(expected),
                          "AEC_FIRST_FAILURE session=%llu clock=CLOCK_MONOTONIC time_ns=%llu before_ns=%llu after_ns=%llu\n",
                          (unsigned long long)b.session,
                          (unsigned long long)time_namespace.st_ino,
                          (unsigned long long)first_before,
                          (unsigned long long)first_after);
    assert(length > 0 && (size_t)length < sizeof(expected));
    char *first_marker = strstr(log, "AEC_FIRST_FAILURE ");
    assert(first_marker != NULL && strstr(first_marker + 1, "AEC_FIRST_FAILURE ") == NULL);
    assert(strstr(log, expected) == first_marker);

    b.first_failure_before_ns = 0;
    memset(log, 0, sizeof(log));
    capture_frame_report(&b, log, sizeof(log));
    assert(strstr(log, "AEC_FRAME_STATE session=81985529216486895 failed=1 reason=4") != NULL);
    assert(strstr(log, "AEC_FIRST_FAILURE ") == NULL);

    b.first_failure_before_ns = 2;
    b.first_failure_after_ns = 1;
    memset(log, 0, sizeof(log));
    capture_frame_report(&b, log, sizeof(log));
    assert(strstr(log, "AEC_FRAME_STATE session=81985529216486895 failed=1 reason=4") != NULL);
    assert(strstr(log, "AEC_FIRST_FAILURE ") == NULL);
    return 0;
}
