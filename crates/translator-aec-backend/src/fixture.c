#define _GNU_SOURCE
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
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#define RATE 48000u
#define QUANTUM 480u

struct fixture {
    struct pw_main_loop *loop;
    struct pw_context *context;
    struct pw_core *core;
    struct pw_filter *filter;
    struct spa_hook listener;
    struct spa_source *control_source;
    struct spa_source *term_source;
    int control_fd;
    void *raw_port;
    void *reference_port;
    uint64_t sample_index;
    uint64_t next_position;
    uint64_t previous_xrun;
    uint32_t first_clock_id;
    uint32_t first_rate_num;
    uint32_t first_rate_denom;
    bool have_clock;
    uint32_t noise_state;
    uint32_t wrong_noise_state;
    float echo_history[QUANTUM];
    float far_current[QUANTUM];
    bool near_only;
    bool wrong_reference;
    bool seq_skew;
    bool meta_gap;
    bool default_header;
    bool prevalid_once;
    bool prevalid_sent;
    uint64_t producer_seq_base;
    bool speech_far;
    bool speech_near;
    bool speech_wrong_reference;
    float *far_pcm;
    float *near_pcm;
    size_t far_samples;
    size_t near_samples;
    size_t far_index;
    size_t near_index;
    _Atomic bool activated;
    _Atomic bool failed;
    _Atomic uint32_t first_failure;
    _Atomic uint64_t failure_expected_position;
    _Atomic uint64_t failure_observed_position;
    _Atomic uint32_t failure_expected_clock;
    _Atomic uint32_t failure_observed_clock;
    _Atomic uint32_t failure_raw_ready;
    _Atomic uint32_t failure_reference_ready;
    _Atomic uint32_t failure_raw_buffer_reason;
    _Atomic uint32_t failure_reference_buffer_reason;
    _Atomic bool have_valid_frame;
    _Atomic uint32_t callback_count;
    _Atomic uint32_t last_position_state;
    _Atomic uint32_t last_position_duration;
    _Atomic uint32_t last_position_clock_id;
    _Atomic uint64_t first_position_tick;
    _Atomic uint64_t last_position_tick;
    _Atomic uint64_t last_position_xrun;
};

enum fixture_failure {
    FIXTURE_POSITION_NULL = 1, FIXTURE_CLOCK = 2, FIXTURE_INACTIVE = 3,
    FIXTURE_BUFFER = 4, FIXTURE_RAW_QUEUE = 5, FIXTURE_REFERENCE_QUEUE = 6,
    FIXTURE_FILTER_STATE = 7, FIXTURE_CONTROL = 8
};
static bool fixture_fail(struct fixture *f, enum fixture_failure reason) {
    uint32_t none = 0;
    bool first = atomic_compare_exchange_strong(&f->first_failure, &none, (uint32_t)reason);
    atomic_store(&f->failed, true);
    return first;
}
static float sample_noise(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return ((float)(x >> 8) / 16777215.0f - 0.5f) * 0.4f;
}
static float *prepare_output(struct pw_buffer *pw_buffer, uint32_t *reason) {
    *reason = 0;
    if (pw_buffer == NULL || pw_buffer->buffer == NULL) {
        *reason = 1; return NULL;
    }
    struct spa_buffer *buffer = pw_buffer->buffer;
    if (buffer->n_datas != 1 || buffer->datas == NULL) {
        *reason = 2; return NULL;
    }
    struct spa_data *data = &buffer->datas[0];
    if (data->data == NULL || data->chunk == NULL ||
        data->maxsize < QUANTUM * sizeof(float)) {
        *reason = 3; return NULL;
    }
    if ((data->flags & SPA_DATA_FLAG_WRITABLE) == 0) {
        *reason = 4; return NULL;
    }
    struct spa_meta_header *header =
        spa_buffer_find_meta_data(buffer, SPA_META_Header, sizeof(*header));
    if (header == NULL) { *reason = 5; return NULL; }
    data->chunk->offset = 0;
    data->chunk->size = QUANTUM * sizeof(float);
    data->chunk->stride = sizeof(float);
    data->chunk->flags = SPA_CHUNK_FLAG_NONE;
    header->flags = SPA_META_HEADER_FLAG_GAP | SPA_META_HEADER_FLAG_CORRUPTED;
    header->offset = 0;
    header->pts = -1;
    header->dts_offset = 0;
    header->seq = 0;
    return data->data;
}
static void mark_valid(struct pw_buffer *pw_buffer, uint64_t seq) {
    struct spa_meta_header *header = spa_buffer_find_meta_data(
        pw_buffer->buffer, SPA_META_Header, sizeof(*header));
    header->flags = 0;
    header->seq = seq;
}
static void on_process(void *userdata, struct spa_io_position *position) {
    struct fixture *f = userdata;
    uint32_t callbacks = atomic_fetch_add(&f->callback_count, 1);
    if (position != NULL) {
        if (callbacks == 0) atomic_store(&f->first_position_tick, position->clock.position);
        atomic_store(&f->last_position_state, (uint32_t)position->state);
        atomic_store(&f->last_position_duration, (uint32_t)position->clock.duration);
        atomic_store(&f->last_position_clock_id, position->clock.id);
        atomic_store(&f->last_position_tick, position->clock.position);
        atomic_store(&f->last_position_xrun, position->clock.xrun);
    }
    if (position == NULL) {
        fixture_fail(f, FIXTURE_POSITION_NULL);
        return;
    }
    struct pw_buffer *raw_buffer = pw_filter_dequeue_buffer(f->raw_port);
    struct pw_buffer *ref_buffer = pw_filter_dequeue_buffer(f->reference_port);
    uint32_t raw_reason, reference_reason;
    float *raw = prepare_output(raw_buffer, &raw_reason);
    float *reference = prepare_output(ref_buffer, &reference_reason);
    const struct spa_io_clock *clock = &position->clock;
    bool clock_valid = clock->duration == QUANTUM &&
        clock->rate.num != 0 && clock->rate.denom == RATE * clock->rate.num &&
        clock->id != SPA_ID_INVALID;
    if (clock_valid && f->have_clock)
        clock_valid = clock->id == f->first_clock_id &&
            clock->rate.num == f->first_rate_num &&
            clock->rate.denom == f->first_rate_denom &&
            clock->position == f->next_position &&
            clock->xrun == f->previous_xrun;
    if (atomic_load(&f->failed) || !atomic_load(&f->activated) || !clock_valid) {
        bool first = false;
        if (!atomic_load(&f->activated))
            first = fixture_fail(f, FIXTURE_INACTIVE);
        else if (!clock_valid)
            first = fixture_fail(f, FIXTURE_CLOCK);
        if (first) {
            atomic_store(&f->failure_expected_position, f->next_position);
            atomic_store(&f->failure_observed_position, clock->position);
            atomic_store(&f->failure_expected_clock, f->first_clock_id);
            atomic_store(&f->failure_observed_clock, clock->id);
            atomic_store(&f->failure_raw_ready, raw != NULL);
            atomic_store(&f->failure_reference_ready, reference != NULL);
            atomic_store(&f->failure_raw_buffer_reason, raw_reason);
            atomic_store(&f->failure_reference_buffer_reason, reference_reason);
        }
        if (raw != NULL) memset(raw, 0, QUANTUM * sizeof(float));
        if (reference != NULL) memset(reference, 0, QUANTUM * sizeof(float));
        if (raw_buffer != NULL) (void)pw_filter_queue_buffer(f->raw_port, raw_buffer);
        if (ref_buffer != NULL) (void)pw_filter_queue_buffer(f->reference_port, ref_buffer);
        return;
    }
    if (!f->have_clock) {
        f->first_clock_id = clock->id;
        f->first_rate_num = clock->rate.num;
        f->first_rate_denom = clock->rate.denom;
        f->have_clock = true;
    }
    f->next_position = clock->position + QUANTUM;
    f->previous_xrun = clock->xrun;
    if (raw == NULL || reference == NULL) {
        if (fixture_fail(f, FIXTURE_BUFFER)) {
            atomic_store(&f->failure_expected_position, f->next_position);
            atomic_store(&f->failure_observed_position, clock->position);
            atomic_store(&f->failure_expected_clock, f->first_clock_id);
            atomic_store(&f->failure_observed_clock, clock->id);
            atomic_store(&f->failure_raw_ready, raw != NULL);
            atomic_store(&f->failure_reference_ready, reference != NULL);
            atomic_store(&f->failure_raw_buffer_reason, raw_reason);
            atomic_store(&f->failure_reference_buffer_reason, reference_reason);
        }
        if (raw != NULL) memset(raw, 0, QUANTUM * sizeof(float));
        if (reference != NULL) memset(reference, 0, QUANTUM * sizeof(float));
    } else if (f->prevalid_once && !f->prevalid_sent) {
        memset(raw, 0, QUANTUM * sizeof(float));
        memset(reference, 0, QUANTUM * sizeof(float));
        f->prevalid_sent = true;
    } else {
        for (uint32_t i = 0; i < QUANTUM; i++) {
            if (f->speech_near) {
                raw[i] = f->near_pcm[f->near_index++ % f->near_samples];
                reference[i] = 0.0f;
            } else if (f->near_only) {
                uint64_t phase = f->sample_index % 109u;
                float triangle = phase < 54u ? (float)phase / 54.0f :
                                 (float)(109u - phase) / 55.0f;
                raw[i] = (triangle * 2.0f - 1.0f) * 0.20f;
                reference[i] = 0.0f;
            } else {
                float far = (f->speech_far || f->speech_wrong_reference) ?
                    f->far_pcm[f->far_index++ % f->far_samples] :
                    sample_noise(&f->noise_state);
                float tap_240 = i >= 240u ? f->far_current[i - 240u] :
                                f->echo_history[i + 240u];
                float tap_480 = f->echo_history[i];
                raw[i] = 0.35f * tap_240 + 0.15f * tap_480;
                f->far_current[i] = far;
                reference[i] = f->speech_wrong_reference ?
                    f->near_pcm[f->near_index++ % f->near_samples] :
                    (f->wrong_reference ?
                     sample_noise(&f->wrong_noise_state) : far);
            }
            f->sample_index++;
        }
        if (!f->near_only && !f->speech_near)
            memcpy(f->echo_history, f->far_current, sizeof(f->echo_history));
        if (!f->meta_gap) {
            uint64_t seq = f->default_header ? 0 :
                f->producer_seq_base + clock->position / QUANTUM;
            mark_valid(raw_buffer, seq);
            mark_valid(ref_buffer, seq + (f->seq_skew ? 1u : 0u));
        }
        atomic_store(&f->have_valid_frame, true);
    }
    if (raw_buffer != NULL && pw_filter_queue_buffer(f->raw_port, raw_buffer) < 0)
        fixture_fail(f, FIXTURE_RAW_QUEUE);
    if (ref_buffer != NULL && pw_filter_queue_buffer(f->reference_port, ref_buffer) < 0)
        fixture_fail(f, FIXTURE_REFERENCE_QUEUE);
}
static void on_state(void *userdata, enum pw_filter_state old_state,
                     enum pw_filter_state state, const char *error) {
    struct fixture *f = userdata;
    (void)old_state; (void)error;
    if (state == PW_FILTER_STATE_ERROR) fixture_fail(f, FIXTURE_FILTER_STATE);
}
static const struct pw_filter_events events = {
    .version = PW_VERSION_FILTER_EVENTS,
    .state_changed = on_state,
    .process = on_process,
};
static void report_fixture(const struct fixture *f) {
    fprintf(stderr,
            "AEC_FIXTURE callbacks=%u state=%u duration=%u clock=%u first=%llu last=%llu xrun=%llu activated=%d valid=%d failed=%d reason=%u expected_pos=%llu observed_pos=%llu expected_clock=%u observed_clock=%u raw_ready=%u ref_ready=%u raw_buffer_reason=%u ref_buffer_reason=%u\n",
            atomic_load(&f->callback_count),
            atomic_load(&f->last_position_state),
            atomic_load(&f->last_position_duration),
            atomic_load(&f->last_position_clock_id),
            (unsigned long long)atomic_load(&f->first_position_tick),
            (unsigned long long)atomic_load(&f->last_position_tick),
            (unsigned long long)atomic_load(&f->last_position_xrun),
            atomic_load(&f->activated),
            atomic_load(&f->have_valid_frame),
            atomic_load(&f->failed),
            atomic_load(&f->first_failure),
            (unsigned long long)atomic_load(&f->failure_expected_position),
            (unsigned long long)atomic_load(&f->failure_observed_position),
            atomic_load(&f->failure_expected_clock),
            atomic_load(&f->failure_observed_clock),
            atomic_load(&f->failure_raw_ready),
            atomic_load(&f->failure_reference_ready),
            atomic_load(&f->failure_raw_buffer_reason),
            atomic_load(&f->failure_reference_buffer_reason));
}
static void on_control(void *userdata, int fd, uint32_t mask) {
    struct fixture *f = userdata;
    (void)mask;
    uint8_t command;
    ssize_t n = recv(fd, &command, 1, MSG_DONTWAIT);
    if (n == 1 && command == 'S' && !atomic_load(&f->activated)) {
        atomic_store(&f->activated, true);
        if (pw_filter_set_active(f->filter, true) >= 0) {
            const uint8_t active = 'A';
            if (send(fd, &active, 1, MSG_DONTWAIT | MSG_NOSIGNAL) == 1) return;
        }
    } else if ((n == 1 && command == 'X') || n == 0) {
        report_fixture(f);
        pw_main_loop_quit(f->loop);
        return;
    } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return;
    }
    fixture_fail(f, FIXTURE_CONTROL);
    report_fixture(f);
    pw_main_loop_quit(f->loop);
}
static void on_term(void *userdata, int signum) {
    struct fixture *f = userdata;
    (void)signum;
    report_fixture(f);
    pw_main_loop_quit(f->loop);
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
static bool parse_sealed_pcm_fd(const char *text, int *fd) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 3 || value > 1024)
        return false;
    struct stat state;
    if (fstat((int)value, &state) < 0 || !S_ISREG(state.st_mode) ||
        state.st_size < (off_t)(RATE * 45u * sizeof(float)) ||
        state.st_size > (off_t)(RATE * 60u * sizeof(float)) ||
        state.st_size % (QUANTUM * sizeof(float)) != 0) return false;
    int seals = fcntl((int)value, F_GET_SEALS);
    int required = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
    if (seals < 0 || (seals & required) != required) return false;
    *fd = (int)value;
    return true;
}
static bool load_sealed_pcm(int fd, float **samples, size_t *count) {
    struct stat state;
    if (fstat(fd, &state) < 0) return false;
    size_t bytes = (size_t)state.st_size;
    float *data = malloc(bytes);
    if (data == NULL) return false;
    size_t offset = 0;
    while (offset < bytes) {
        ssize_t got = pread(fd, (uint8_t *)data + offset, bytes - offset, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { free(data); return false; }
        offset += (size_t)got;
    }
    for (size_t i = 0; i < bytes / sizeof(float); i++) {
        if (!isfinite(data[i]) || fabsf(data[i]) > 1.0f) {
            free(data);
            return false;
        }
    }
    *samples = data;
    *count = bytes / sizeof(float);
    return true;
}
static bool parse_session(const char *text) {
    if (strlen(text) != 16) return false;
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 16);
    return errno == 0 && *end == '\0' && value != 0;
}
int translator_aec_fixture_run(int argc, const char **argv) {
    int fd = -1, control_fd = -1, far_fd = -1, near_fd = -1;
    bool legacy_mode = argc == 7 ||
        (argc == 9 && strcmp(argv[7], "--mode") == 0 &&
         (strcmp(argv[8], "far-only") == 0 ||
          strcmp(argv[8], "near-only") == 0 ||
          strcmp(argv[8], "wrong-reference") == 0 ||
          strcmp(argv[8], "seq-skew") == 0 ||
          strcmp(argv[8], "meta-gap") == 0 ||
          strcmp(argv[8], "default-header") == 0 ||
          strcmp(argv[8], "prevalid-once") == 0));
    bool speech_mode = argc == 13 && strcmp(argv[7], "--mode") == 0 &&
        (strcmp(argv[8], "speech-far") == 0 ||
         strcmp(argv[8], "speech-near") == 0 ||
         strcmp(argv[8], "speech-wrong-reference") == 0) &&
        strcmp(argv[9], "--far-pcm-fd") == 0 &&
        strcmp(argv[11], "--near-pcm-fd") == 0 &&
        parse_sealed_pcm_fd(argv[10], &far_fd) &&
        parse_sealed_pcm_fd(argv[12], &near_fd) &&
        far_fd != near_fd;
    if (!((legacy_mode || speech_mode) &&
          strcmp(argv[1], "--pipewire-fd") == 0 &&
          strcmp(argv[3], "--control-fd") == 0 &&
          strcmp(argv[5], "--session-id") == 0 &&
          parse_fd(argv[2], &fd) && parse_fd(argv[4], &control_fd) &&
          fd != control_fd && fd != far_fd && fd != near_fd &&
          control_fd != far_fd && control_fd != near_fd &&
          parse_session(argv[6]))) {
        fputs("translator-aec-fixture: connected private sockets, sealed speech FDs when selected, and session-id required\n", stderr);
        return 2;
    }
    bool near_only = argc == 9 && strcmp(argv[8], "near-only") == 0;
    struct fixture *f = calloc(1, sizeof(*f));
    if (f == NULL) return 2;
    f->control_fd = control_fd;
    f->near_only = near_only;
    f->wrong_reference = argc == 9 && strcmp(argv[8], "wrong-reference") == 0;
    f->seq_skew = argc == 9 && strcmp(argv[8], "seq-skew") == 0;
    f->meta_gap = argc == 9 && strcmp(argv[8], "meta-gap") == 0;
    f->default_header = argc == 9 && strcmp(argv[8], "default-header") == 0;
    f->prevalid_once = argc == 9 && strcmp(argv[8], "prevalid-once") == 0;
    uint64_t session = (uint64_t)strtoull(argv[6], NULL, 16);
    f->producer_seq_base = (session & UINT64_C(0x3fffffffffffffff)) |
        UINT64_C(0x4000000000000000);
    f->speech_far = speech_mode && strcmp(argv[8], "speech-far") == 0;
    f->speech_near = speech_mode && strcmp(argv[8], "speech-near") == 0;
    f->speech_wrong_reference =
        speech_mode && strcmp(argv[8], "speech-wrong-reference") == 0;
    f->noise_state = 0x9e3779b9u;
    f->wrong_noise_state = 0x6a09e667u;
    pw_init(NULL, NULL);
    int result = 2;
    if (speech_mode &&
        (!load_sealed_pcm(far_fd, &f->far_pcm, &f->far_samples) ||
         !load_sealed_pcm(near_fd, &f->near_pcm, &f->near_samples))) goto out;
    if (far_fd >= 0) { close(far_fd); far_fd = -1; }
    if (near_fd >= 0) { close(near_fd); near_fd = -1; }
    f->loop = pw_main_loop_new(NULL);
    if (f->loop == NULL) goto out;
    struct pw_loop *loop = pw_main_loop_get_loop(f->loop);
    f->context = pw_context_new(loop, NULL, 0);
    if (f->context == NULL) goto out;
    int owned_fd = fd;
    fd = -1;
    f->core = pw_context_connect_fd(f->context, owned_fd, NULL, 0);
    if (f->core == NULL) goto out;
    char name[64];
    snprintf(name, sizeof(name), "translator-aec-fixture-%s", argv[6]);
    f->filter = pw_filter_new(f->core, name,
        pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio",
                          PW_KEY_MEDIA_CATEGORY, "Filter",
                          PW_KEY_MEDIA_ROLE, "DSP",
                          PW_KEY_NODE_NAME, name,
                          PW_KEY_NODE_VIRTUAL, "true",
                          PW_KEY_NODE_AUTOCONNECT, "false", NULL));
    if (f->filter == NULL) goto out;
    pw_filter_add_listener(f->filter, &f->listener, &events, f);
    uint8_t pod_buffer[256];
    struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(pod_buffer, sizeof(pod_buffer));
    const struct spa_pod *meta = spa_pod_builder_add_object(&builder,
        SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
        SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
        SPA_PARAM_META_size, SPA_POD_Int(sizeof(struct spa_meta_header)));
    if (meta == NULL) goto out;
    const struct spa_pod *params[] = { meta };
    f->raw_port = pw_filter_add_port(f->filter, PW_DIRECTION_OUTPUT,
        PW_FILTER_PORT_FLAG_MAP_BUFFERS, 0,
        pw_properties_new(PW_KEY_FORMAT_DSP, "32 bit float mono audio",
                          PW_KEY_PORT_NAME, "raw", NULL), params, 1);
    f->reference_port = pw_filter_add_port(f->filter, PW_DIRECTION_OUTPUT,
        PW_FILTER_PORT_FLAG_MAP_BUFFERS, 0,
        pw_properties_new(PW_KEY_FORMAT_DSP, "32 bit float mono audio",
                          PW_KEY_PORT_NAME, "reference", NULL), params, 1);
    if (f->raw_port == NULL || f->reference_port == NULL) goto out;
    int flags = fcntl(control_fd, F_GETFL, 0);
    if (flags < 0 || fcntl(control_fd, F_SETFL, flags | O_NONBLOCK) < 0) goto out;
    f->control_source = pw_loop_add_io(loop, control_fd,
        SPA_IO_IN | SPA_IO_ERR | SPA_IO_HUP, false, on_control, f);
    f->term_source = pw_loop_add_signal(loop, SIGTERM, on_term, f);
    if (f->control_source == NULL || f->term_source == NULL) goto out;
    if (pw_filter_connect(f->filter, PW_FILTER_FLAG_INACTIVE | PW_FILTER_FLAG_RT_PROCESS,
                          NULL, 0) < 0) goto out;
    const uint8_t ready = 'R';
    if (send(control_fd, &ready, 1, MSG_DONTWAIT | MSG_NOSIGNAL) != 1) goto out;
    pw_main_loop_run(f->loop);
    result = f->failed ? 1 : 0;
out:
    if (fd >= 0) close(fd);
    if (far_fd >= 0) close(far_fd);
    if (near_fd >= 0) close(near_fd);
    if (f->loop != NULL) {
        struct pw_loop *loop = pw_main_loop_get_loop(f->loop);
        if (f->control_source != NULL) pw_loop_destroy_source(loop, f->control_source);
        if (f->term_source != NULL) pw_loop_destroy_source(loop, f->term_source);
    }
    if (f->control_fd >= 0) close(f->control_fd);
    if (f->filter != NULL) pw_filter_destroy(f->filter);
    if (f->core != NULL) pw_core_disconnect(f->core);
    if (f->context != NULL) pw_context_destroy(f->context);
    if (f->loop != NULL) pw_main_loop_destroy(f->loop);
    pw_deinit();
    free(f->far_pcm);
    free(f->near_pcm);
    free(f);
    return result;
}
