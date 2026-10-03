#define _POSIX_C_SOURCE 200809L
#include <pipewire/pipewire.h>
#include <pipewire/filter.h>
#include <spa/buffer/meta.h>
#include <spa/param/buffers.h>
#include <spa/pod/builder.h>
#include <spa/pod/vararg.h>
#include <spa/node/io.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "callback_history.h"

#define RATE 48000u
#define QUANTUM 480u
#define QUEUE_SLOTS 20u
#define RECORD_BYTES (25u + QUANTUM * sizeof(float))

enum witness_failure {
    WITNESS_OK = 0, WITNESS_CLOCK = 1, WITNESS_BUFFER = 2,
    WITNESS_GAP = 3, WITNESS_PCM = 4, WITNESS_QUEUE = 5,
    WITNESS_IO = 6, WITNESS_GRAPH = 7
};
struct frame_state {
    uint32_t reason, callback, duration, clock, valid_mask, queue_head, queue_tail;
    uint64_t expected_position, position, xrun, sequence;
};

struct witness_frame {
    uint64_t seq;
    uint32_t clock_id;
    uint64_t position;
    float samples[QUANTUM];
};
struct witness {
    uint64_t session;
    struct pw_main_loop *loop;
    struct pw_context *context;
    struct pw_core *core;
    struct pw_filter *filter;
    struct spa_hook listener;
    struct spa_source *control_source;
    struct spa_source *term_source;
    void *clean_port;
    int output_fd;
    uint32_t clock_id;
    uint32_t rate_num;
    uint32_t rate_denom;
    uint64_t next_position;
    uint64_t previous_xrun;
    uint64_t previous_seq;
    _Atomic bool have_frame;
    _Atomic bool started;
    _Atomic bool stopping;
    _Atomic bool ack_sent;
    _Atomic bool stop_writer;
    _Atomic uint32_t failure;
    _Atomic uint32_t callback_count;
    _Atomic uint32_t last_duration;
    _Atomic uint32_t last_clock;
    _Atomic uint64_t last_position;
    _Atomic uint64_t last_xrun;
    _Atomic uint64_t observed_sequence;
    _Atomic uint64_t accepted_next_position;
    _Atomic uint32_t valid_mask;
    _Atomic bool snapshot_ready;
    struct frame_state snapshot;
    struct aec_callback_history history;
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
    struct witness_frame queue[QUEUE_SLOTS];
    pthread_t writer;
    bool writer_started;
};

static void put32(uint8_t *dst, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) dst[i] = (uint8_t)(value >> (8 * i));
}
static void put64(uint8_t *dst, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) dst[i] = (uint8_t)(value >> (8 * i));
}
static void fail(struct witness *w, enum witness_failure reason) {
    uint32_t none = WITNESS_OK;
    if (atomic_compare_exchange_strong(&w->failure, &none, (uint32_t)reason)) {
        aec_history_fail(&w->history);
        w->snapshot = (struct frame_state) {
            .reason = (uint32_t)reason,
            .callback = atomic_load(&w->callback_count),
            .duration = atomic_load(&w->last_duration),
            .clock = atomic_load(&w->last_clock),
            .valid_mask = atomic_load(&w->valid_mask),
            .queue_head = atomic_load(&w->head),
            .queue_tail = atomic_load(&w->tail),
            .expected_position = atomic_load(&w->accepted_next_position),
            .position = atomic_load(&w->last_position),
            .xrun = atomic_load(&w->last_xrun),
            .sequence = atomic_load(&w->observed_sequence),
        };
        atomic_store_explicit(&w->snapshot_ready, true, memory_order_release);
    }
}
static bool write_all_bounded(int fd, const uint8_t *bytes, size_t length) {
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    size_t sent = 0;
    while (sent < length) {
        ssize_t count = send(fd, bytes + sent, length - sent, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (count > 0) { sent += (size_t)count; continue; }
        if (count < 0 && errno != EAGAIN && errno != EINTR) return false;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if ((now.tv_sec - start.tv_sec) * 1000000000LL + now.tv_nsec - start.tv_nsec > 100000000LL)
            return false;
        struct timespec pause = { .tv_sec = 0, .tv_nsec = 1000000 };
        nanosleep(&pause, NULL);
    }
    return true;
}
static void *writer_main(void *userdata) {
    struct witness *w = userdata;
    while (!atomic_load(&w->stop_writer)) {
        uint32_t reason = atomic_load(&w->failure);
        if (reason != WITNESS_OK) {
            uint8_t fatal[13] = { 'F' };
            put32(fatal + 1, reason);
            put64(fatal + 5, atomic_load(&w->head));
            (void)write_all_bounded(w->output_fd, fatal, sizeof(fatal));
            break;
        }
        uint32_t tail = atomic_load_explicit(&w->tail, memory_order_relaxed);
        uint32_t head = atomic_load_explicit(&w->head, memory_order_acquire);
        if (tail != head && atomic_load(&w->ack_sent)) {
            const struct witness_frame *frame = &w->queue[tail % QUEUE_SLOTS];
            uint8_t record[RECORD_BYTES] = { 'W' };
            put64(record + 1, frame->seq);
            put32(record + 9, frame->clock_id);
            put64(record + 13, frame->position);
            put32(record + 21, QUANTUM);
            memcpy(record + 25, frame->samples, QUANTUM * sizeof(float));
            if (!write_all_bounded(w->output_fd, record, sizeof(record))) {
                if (!atomic_load(&w->stopping)) fail(w, WITNESS_IO);
                break;
            }
            atomic_store_explicit(&w->tail, tail + 1, memory_order_release);
            continue;
        }
        struct timespec pause = { .tv_sec = 0, .tv_nsec = 1000000 };
        nanosleep(&pause, NULL);
    }
    return NULL;
}
static bool checked_input(struct pw_buffer *pw_buffer, const float **samples,
                          uint64_t *seq) {
    if (pw_buffer == NULL || pw_buffer->buffer == NULL) return false;
    struct spa_buffer *buffer = pw_buffer->buffer;
    if (buffer->n_datas != 1 || buffer->datas == NULL) return false;
    const struct spa_data *data = &buffer->datas[0];
    if (data->data == NULL || data->chunk == NULL ||
        data->maxsize < QUANTUM * sizeof(float)) return false;
    const struct spa_chunk *chunk = data->chunk;
    if (chunk->offset > data->maxsize ||
        data->maxsize - chunk->offset < QUANTUM * sizeof(float) ||
        chunk->size != QUANTUM * sizeof(float) ||
        chunk->stride != (int32_t)sizeof(float) ||
        chunk->flags != SPA_CHUNK_FLAG_NONE) return false;
    const struct spa_meta_header *header =
        spa_buffer_find_meta_data(buffer, SPA_META_Header, sizeof(*header));
    if (header == NULL ||
        (header->flags & (SPA_META_HEADER_FLAG_DISCONT |
                          SPA_META_HEADER_FLAG_CORRUPTED |
                          SPA_META_HEADER_FLAG_GAP)) != 0) return false;
    *seq = header->seq;
    *samples = (const float *)((const uint8_t *)data->data + chunk->offset);
    for (uint32_t i = 0; i < QUANTUM; i++)
        if (!isfinite((*samples)[i]) || fabsf((*samples)[i]) > 1.0f) return false;
    return true;
}
static void on_process(void *userdata, struct spa_io_position *position) {
    struct witness *w = userdata;
    uint32_t callbacks = atomic_fetch_add(&w->callback_count, 1);
    struct aec_callback_event trace;
    bool trace_active = aec_history_begin(&w->history, &trace, (uint64_t)callbacks + 1, position);
    if (trace_active) {
        trace.expected_position = w->next_position;
        trace.admission_position = UINT64_MAX;
        trace.expected_seq = w->previous_seq + 1;
        trace.head = atomic_load(&w->head);
        trace.tail = atomic_load(&w->tail);
        trace.stage = 1;
    }
    atomic_store(&w->observed_sequence, 0);
    atomic_store(&w->valid_mask, 0);
    if (position != NULL) {
        atomic_store(&w->last_duration, (uint32_t)position->clock.duration);
        atomic_store(&w->last_clock, position->clock.id);
        atomic_store(&w->last_position, position->clock.position);
        atomic_store(&w->last_xrun, position->clock.xrun);
    }
    struct pw_buffer *buffer = pw_filter_dequeue_buffer(w->clean_port);
    const float *samples = NULL;
    uint64_t seq = 0;
    bool input_valid = checked_input(buffer, &samples, &seq);
    atomic_store(&w->observed_sequence, seq);
    atomic_store(&w->valid_mask, input_valid ? 4u : 0u);
    if (trace_active) { trace.output_seq = seq; trace.valid_mask = input_valid ? 4u : 0u; trace.stage = 2; }
    if (atomic_load(&w->stopping) ||
        atomic_load(&w->failure) != WITNESS_OK) goto done;
    if (position == NULL || position->clock.duration != QUANTUM ||
        position->clock.rate.num == 0 ||
        position->clock.rate.denom != RATE * position->clock.rate.num ||
        position->clock.id == SPA_ID_INVALID) {
        fail(w, WITNESS_CLOCK);
        goto done;
    }
    if (!atomic_load(&w->started)) goto done;
    if (!input_valid) {
        if (atomic_load(&w->have_frame)) fail(w, WITNESS_BUFFER);
        goto done;
    }
    const struct spa_io_clock *clock = &position->clock;
    if (trace_active) aec_history_admission(&trace, clock);
    if (atomic_load(&w->have_frame) &&
        (seq != w->previous_seq + 1 || clock->id != w->clock_id ||
         clock->rate.num != w->rate_num || clock->rate.denom != w->rate_denom ||
         aec_history_used_position(clock->position, &trace, trace_active) != w->next_position ||
         clock->xrun != w->previous_xrun)) {
        fail(w, WITNESS_GAP);
        goto done;
    }
    uint32_t head = atomic_load_explicit(&w->head, memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(&w->tail, memory_order_acquire);
    if (head - tail >= QUEUE_SLOTS) {
        fail(w, WITNESS_QUEUE);
        goto done;
    }
    struct witness_frame *frame = &w->queue[head % QUEUE_SLOTS];
    frame->seq = seq;
    frame->clock_id = clock->id;
    frame->position = clock->position;
    if (trace_active) { aec_history_publication(&trace, clock); trace.publication_position = frame->position; }
    memcpy(frame->samples, samples, sizeof(frame->samples));
    w->clock_id = clock->id;
    w->rate_num = clock->rate.num;
    w->rate_denom = clock->rate.denom;
    uint64_t next_source_position = clock->position;
    w->next_position = next_source_position + QUANTUM;
    if (trace_active) {
        trace.next_source_position = next_source_position;
        if (trace.admission_position == UINT64_MAX) trace.admission_position = next_source_position;
        trace.next_position = w->next_position; trace.stage = 4; }
    atomic_store(&w->accepted_next_position, w->next_position);
    w->previous_xrun = clock->xrun;
    w->previous_seq = seq;
    atomic_store(&w->have_frame, true);
    atomic_store_explicit(&w->head, head + 1, memory_order_release);
done:
    if (buffer != NULL && pw_filter_queue_buffer(w->clean_port, buffer) < 0 &&
        !atomic_load(&w->stopping))
        fail(w, WITNESS_BUFFER);
    if (trace_active) {
        trace.reason = atomic_load(&w->failure);
        trace.head = atomic_load(&w->head);
        trace.tail = atomic_load(&w->tail);
        if (trace.reason) trace.stage = 5;
    }
    aec_history_end(&w->history, &trace, trace_active);
}
static void on_state(void *userdata, enum pw_filter_state old_state,
                     enum pw_filter_state state, const char *error) {
    struct witness *w = userdata;
    (void)error;
    if (!atomic_load(&w->stopping) &&
        (state == PW_FILTER_STATE_ERROR ||
         (old_state == PW_FILTER_STATE_STREAMING &&
          state != PW_FILTER_STATE_STREAMING && atomic_load(&w->have_frame))))
        fail(w, WITNESS_GRAPH);
}
static const struct pw_filter_events events = {
    .version = PW_VERSION_FILTER_EVENTS,
    .state_changed = on_state,
    .process = on_process,
};
static void on_control(void *userdata, int fd, uint32_t mask) {
    struct witness *w = userdata;
    (void)mask;
    uint8_t command;
    ssize_t n = recv(fd, &command, 1, MSG_DONTWAIT);
    if (n == 1 && command == 'S' && !atomic_load(&w->started)) {
        atomic_store(&w->started, true);
        if (pw_filter_set_active(w->filter, true) >= 0) {
            const uint8_t active = 'A';
            if (send(fd, &active, 1, MSG_DONTWAIT | MSG_NOSIGNAL) == 1) {
                atomic_store(&w->ack_sent, true);
                return;
            }
        }
    } else if ((n == 1 && command == 'X') || n == 0) {
        atomic_store(&w->stopping, true);
        pw_main_loop_quit(w->loop);
        return;
    } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return;
    }
    fail(w, WITNESS_IO);
    pw_main_loop_quit(w->loop);
}
static void on_term(void *userdata, int signum) {
    struct witness *w = userdata;
    (void)signum;
    atomic_store(&w->stopping, true);
    pw_main_loop_quit(w->loop);
}
static void report_frame_state(const struct witness *w) {
    bool failed = atomic_load_explicit(&w->snapshot_ready, memory_order_acquire);
    if (atomic_load(&w->failure) != WITNESS_OK && !failed) return;
    struct frame_state state = failed ? w->snapshot : (struct frame_state) {
        .callback = atomic_load(&w->callback_count),
        .duration = atomic_load(&w->last_duration),
        .clock = atomic_load(&w->last_clock),
        .valid_mask = atomic_load(&w->valid_mask),
        .queue_head = atomic_load(&w->head),
        .queue_tail = atomic_load(&w->tail),
        .expected_position = atomic_load(&w->accepted_next_position),
        .position = atomic_load(&w->last_position),
        .xrun = atomic_load(&w->last_xrun),
        .sequence = atomic_load(&w->observed_sequence),
    };
    fprintf(stderr,
            "AEC_FRAME_STATE session=%llu failed=%d reason=%u callback=%u expected_pos=%llu position=%llu duration=%u clock=%u xrun=%llu seq=%llu valid=%u queue_head=%u queue_tail=%u\n",
            (unsigned long long)w->session, failed, state.reason, state.callback,
            (unsigned long long)state.expected_position,
            (unsigned long long)state.position, state.duration, state.clock,
            (unsigned long long)state.xrun, (unsigned long long)state.sequence,
            state.valid_mask, state.queue_head, state.queue_tail);
}

static bool parse_fd(const char *text, int *fd) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 3 || value > 1024 ||
        fcntl((int)value, F_GETFD) < 0) return false;
    int type;
    socklen_t size = sizeof(type);
    if (getsockopt((int)value, SOL_SOCKET, SO_TYPE, &type, &size) < 0 || type != SOCK_STREAM)
        return false;
    struct sockaddr_storage peer;
    socklen_t peer_size = sizeof(peer);
    if (getpeername((int)value, (struct sockaddr *)&peer, &peer_size) < 0 ||
        peer.ss_family != AF_UNIX) return false;
    *fd = (int)value;
    return true;
}
static bool parse_session(const char *text) {
    if (strlen(text) != 16) return false;
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 16);
    return errno == 0 && *end == '\0' && value != 0;
}
int translator_aec_witness_run(int argc, const char **argv) {
    int pw_fd = -1, output_fd = -1;
    if (argc != 7 || strcmp(argv[1], "--pipewire-fd") != 0 ||
        strcmp(argv[3], "--output-fd") != 0 ||
        strcmp(argv[5], "--session-id") != 0 ||
        !parse_fd(argv[2], &pw_fd) || !parse_fd(argv[4], &output_fd) ||
        pw_fd == output_fd || !parse_session(argv[6])) {
        fputs("translator-aec-witness: connected private PipeWire/output FDs and session-id required\n", stderr);
        return 2;
    }
    struct witness *w = calloc(1, sizeof(*w));
    if (w == NULL) return 2;
    w->session = (uint64_t)strtoull(argv[6], NULL, 16);
    w->output_fd = output_fd;
    pw_init(NULL, NULL);
    int result = 2;
    bool ran_loop = false;
    w->loop = pw_main_loop_new(NULL);
    if (w->loop == NULL) goto out;
    struct pw_loop *loop = pw_main_loop_get_loop(w->loop);
    w->context = pw_context_new(loop, NULL, 0);
    if (w->context == NULL) goto out;
    int owned_pw_fd = pw_fd;
    pw_fd = -1;
    w->core = pw_context_connect_fd(w->context, owned_pw_fd, NULL, 0);
    if (w->core == NULL) goto out;
    char name[64];
    snprintf(name, sizeof(name), "translator-aec-witness-%s", argv[6]);
    w->filter = pw_filter_new(w->core, name,
        pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio",
                          PW_KEY_MEDIA_CATEGORY, "Filter",
                          PW_KEY_MEDIA_ROLE, "DSP",
                          PW_KEY_MEDIA_CLASS, "Audio/Sink",
                          PW_KEY_NODE_NAME, name,
                          PW_KEY_NODE_VIRTUAL, "true",
                          PW_KEY_NODE_AUTOCONNECT, "false",
                          PW_KEY_NODE_ALWAYS_PROCESS, "true", NULL));
    if (w->filter == NULL) goto out;
    pw_filter_add_listener(w->filter, &w->listener, &events, w);
    uint8_t pod_buffer[256];
    struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(pod_buffer, sizeof(pod_buffer));
    const struct spa_pod *meta = spa_pod_builder_add_object(&builder,
        SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
        SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
        SPA_PARAM_META_size, SPA_POD_Int(sizeof(struct spa_meta_header)));
    if (meta == NULL) goto out;
    const struct spa_pod *port_params[] = { meta };
    w->clean_port = pw_filter_add_port(w->filter, PW_DIRECTION_INPUT,
        PW_FILTER_PORT_FLAG_MAP_BUFFERS, 0,
        pw_properties_new(PW_KEY_FORMAT_DSP, "32 bit float mono audio",
                          PW_KEY_PORT_NAME, "clean", NULL), port_params, 1);
    if (w->clean_port == NULL) goto out;
    int flags = fcntl(output_fd, F_GETFL, 0);
    if (flags < 0 || fcntl(output_fd, F_SETFL, flags | O_NONBLOCK) < 0) goto out;
    w->control_source = pw_loop_add_io(loop, output_fd,
        SPA_IO_IN | SPA_IO_ERR | SPA_IO_HUP, false, on_control, w);
    w->term_source = pw_loop_add_signal(loop, SIGTERM, on_term, w);
    if (w->control_source == NULL || w->term_source == NULL) goto out;
    if (pthread_create(&w->writer, NULL, writer_main, w) != 0) goto out;
    w->writer_started = true;
    if (pw_filter_connect(w->filter, PW_FILTER_FLAG_INACTIVE | PW_FILTER_FLAG_RT_PROCESS,
                          NULL, 0) < 0) goto out;
    const uint8_t ready = 'R';
    if (send(output_fd, &ready, 1, MSG_DONTWAIT | MSG_NOSIGNAL) != 1) goto out;
    ran_loop = true;
    pw_main_loop_run(w->loop);
out:
    atomic_store(&w->stopping, true);
    if (pw_fd >= 0) close(pw_fd);
    if (w->loop != NULL) {
        struct pw_loop *loop = pw_main_loop_get_loop(w->loop);
        if (w->control_source != NULL) pw_loop_destroy_source(loop, w->control_source);
        if (w->term_source != NULL) pw_loop_destroy_source(loop, w->term_source);
    }
    if (w->filter != NULL) pw_filter_destroy(w->filter);
    atomic_store(&w->stop_writer, true);
    if (w->writer_started) pthread_join(w->writer, NULL);
    if (w->core != NULL) pw_core_disconnect(w->core);
    if (w->context != NULL) pw_context_destroy(w->context);
    if (w->loop != NULL) pw_main_loop_destroy(w->loop);
    if (w->output_fd >= 0) close(w->output_fd);
    pw_deinit();
    if (ran_loop) {
        report_frame_state(w);
        aec_history_report(&w->history, w->session, "witness");
        result = atomic_load(&w->failure) == WITNESS_OK ? 0 : 1;
    }
    free(w);
    return result;
}
