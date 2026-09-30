#define _POSIX_C_SOURCE 200809L
#include <pipewire/pipewire.h>
#include <pipewire/filter.h>
#include <pipewire/node.h>
#include <pipewire/port.h>
#include <pipewire/link.h>
#include <spa/interfaces/audio/aec.h>
#include <spa/support/plugin.h>
#include <spa/utils/names.h>
#include <spa/node/io.h>
#include <spa/param/buffers.h>
#include <spa/pod/builder.h>
#include <spa/pod/vararg.h>
#include <spa/utils/string.h>
#include <stdatomic.h>
#include <errno.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define RATE 48000u
#define QUANTUM 480u
#define QUEUE_SLOTS 20u
#define IPC_VERSION 1u
#define IPC_MAX_BODY (72u + QUANTUM * 3u * 4u)
#define MAX_NODES 64u
#define MAX_PORTS 96u
#define MAX_LINKS 128u

enum ipc_type { IPC_HELLO = 1, IPC_FRAME = 2, IPC_FATAL = 3, IPC_STOP = 4, IPC_START = 5, IPC_LINKS = 6, IPC_ARMED = 7, IPC_INJECT_AEC_ERROR = 8 };
enum failure {
    FAIL_NONE = 0, FAIL_GRAPH_STATE = 1, FAIL_CLOCK = 2, FAIL_GAP = 3,
    FAIL_BUFFER = 4, FAIL_PCM = 5, FAIL_AEC = 6, FAIL_QUEUE = 7,
    FAIL_IPC = 8, FAIL_ABI = 9, FAIL_LINK = 10
};

struct frame {
    uint64_t sequence;
    uint64_t position;
    uint64_t duration;
    uint64_t xrun;
    uint32_t clock_id;
    uint32_t rate_num;
    uint32_t rate_denom;
    float raw[QUANTUM];
    float reference[QUANTUM];
    float clean[QUANTUM];
};

struct backend;
struct node_fact {
    struct backend *owner;
    struct pw_node *proxy;
    struct spa_hook listener;
    uint32_t id;
    bool virtual_node;
    bool seen_info;
    bool present;
};
struct port_fact { uint32_t id, node_id; char name[24]; bool present; };
struct link_fact {
    uint32_t id, input_node, input_port, output_node, output_port;
    bool present;
};
struct binding { uint32_t local_port, link_id, peer_node, peer_port; };

struct backend {
    struct pw_main_loop *loop;
    struct pw_context *context;
    struct pw_core *core;
    struct pw_filter *filter;
    struct pw_registry *registry;
    struct spa_hook filter_listener;
    struct spa_hook registry_listener;
    struct spa_source *ipc_source;
    struct spa_source *term_source;
    void *raw_port;
    void *reference_port;
    void *clean_port;
    void *plugin_library;
    struct spa_handle *plugin_handle;
    struct spa_audio_aec *aec;
    int ipc_fd;
    uint64_t session;
    uint64_t generation;
    uint64_t producer_seq_base;
    uint32_t node_id;
    uint32_t first_clock_id;
    uint32_t first_rate_num;
    uint32_t first_rate_denom;
    uint64_t next_position;
    uint64_t last_xrun;
    uint64_t last_raw_seq;
    uint64_t last_reference_seq;
    bool have_clock;
    bool have_port_seq;
    bool start_requested;
    unsigned diagnostic_count;
    struct node_fact nodes[MAX_NODES];
    struct port_fact ports[MAX_PORTS];
    struct link_fact links[MAX_LINKS];
    struct binding bindings[3];
    _Atomic bool links_ready;
    _Atomic bool links_sent;
    _Atomic bool armed_ready;
    _Atomic bool armed_sent;
    _Atomic uint32_t prevalid_quarantine_count;
    _Atomic uint32_t prevalid_empty_count;
    _Atomic uint32_t prevalid_sentinel_count;
    _Atomic bool ready;
    _Atomic bool hello_sent;
    _Atomic bool started;
    _Atomic bool stopping;
    _Atomic bool writer_stop;
    _Atomic bool inject_aec_error;
    _Atomic uint32_t failure;
    _Atomic bool bad_position_present;
    _Atomic uint32_t bad_position_duration;
    _Atomic uint32_t bad_position_state;
    _Atomic uint32_t bad_position_rate_num;
    _Atomic uint32_t bad_position_rate_denom;
    _Atomic uint32_t bad_buffer_site;
    _Atomic uint32_t bad_buffer_callback;
    _Atomic uint64_t bad_buffer_position;
    _Atomic bool bad_input_metadata_captured[2];
    _Atomic uint32_t bad_buffer_reason[3];
    _Atomic uint32_t bad_input_chunk_size[2];
    _Atomic uint32_t bad_input_chunk_stride[2];
    _Atomic uint32_t bad_input_chunk_flags[2];
    _Atomic uint32_t bad_input_header_present[2];
    _Atomic uint32_t bad_input_header_flags[2];
    _Atomic uint64_t bad_input_header_seq[2];
    _Atomic uint32_t bad_buffer_state;
    _Atomic bool bad_buffer_started;
    _Atomic uint32_t bad_gap_kind;
    _Atomic uint64_t bad_gap_expected_xrun;
    _Atomic uint64_t bad_gap_observed_xrun;
    _Atomic uint64_t bad_gap_expected_position;
    _Atomic uint64_t bad_gap_observed_position;
    _Atomic uint32_t bad_gap_expected_clock;
    _Atomic uint32_t bad_gap_observed_clock;
    _Atomic uint64_t bad_gap_expected_raw_seq;
    _Atomic uint64_t bad_gap_observed_raw_seq;
    _Atomic uint64_t bad_gap_expected_reference_seq;
    _Atomic uint64_t bad_gap_observed_reference_seq;
    _Atomic bool have_valid_frame;
    _Atomic uint32_t callback_count;
    _Atomic uint32_t last_position_state;
    _Atomic uint32_t last_position_duration;
    _Atomic uint32_t last_position_clock_id;
    _Atomic uint64_t first_position_tick;
    _Atomic uint64_t last_position_tick;
    _Atomic uint64_t last_position_xrun;
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
    struct frame frames[QUEUE_SLOTS];
    pthread_t writer;
    bool writer_started;
    uint8_t inbound[8];
    size_t inbound_used;
};

static void put32(uint8_t *dst, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) dst[i] = (uint8_t)(value >> (8 * i));
}
static void put64(uint8_t *dst, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) dst[i] = (uint8_t)(value >> (8 * i));
}
static uint32_t get32(const uint8_t *src) {
    return (uint32_t)src[0] | (uint32_t)src[1] << 8 | (uint32_t)src[2] << 16 | (uint32_t)src[3] << 24;
}
static void poison_at(struct backend *b, enum failure reason, unsigned site) {
    if (reason == FAIL_BUFFER) {
        uint32_t unknown = 0;
        if (atomic_compare_exchange_strong(&b->bad_buffer_site, &unknown, site)) {
            atomic_store(&b->bad_buffer_callback, atomic_load(&b->callback_count));
            atomic_store(&b->bad_buffer_position, atomic_load(&b->last_position_tick));
        }
    }
    uint32_t none = FAIL_NONE;
    if (atomic_compare_exchange_strong(&b->failure, &none, (uint32_t)reason) &&
        reason == FAIL_LINK)
        fprintf(stderr, "AEC_FAILURE reason=%u site=%u\n", (unsigned)reason, site);
}
#define poison(b, reason) poison_at((b), (reason), __LINE__)
static bool finite_pcm(const float *samples) {
    for (uint32_t i = 0; i < QUANTUM; i++)
        if (!isfinite(samples[i]) || fabsf(samples[i]) > 1.0f) return false;
    return true;
}
static float *checked_pcm(struct pw_buffer *pw_buffer, bool input, uint64_t *seq,
                          uint32_t *reason) {
    *reason = 0;
    if (pw_buffer == NULL || pw_buffer->buffer == NULL) { *reason = 1; return NULL; }
    struct spa_buffer *buffer = pw_buffer->buffer;
    if (buffer->n_datas != 1 || buffer->datas == NULL) { *reason = 2; return NULL; }
    struct spa_data *data = &buffer->datas[0];
    if (data->data == NULL || data->chunk == NULL ||
        data->maxsize < QUANTUM * sizeof(float)) { *reason = 3; return NULL; }
    struct spa_chunk *chunk = data->chunk;
    if (chunk->offset > data->maxsize ||
        data->maxsize - chunk->offset < QUANTUM * sizeof(float)) {
        *reason = 4; return NULL;
    }
    if (input) {
        if (chunk->size != QUANTUM * sizeof(float) ||
            chunk->stride != (int32_t)sizeof(float) ||
            chunk->flags != SPA_CHUNK_FLAG_NONE) { *reason = 5; return NULL; }
        struct spa_meta_header *header =
            spa_buffer_find_meta_data(buffer, SPA_META_Header, sizeof(*header));
        if (header == NULL) { *reason = 6; return NULL; }
        if ((header->flags & (SPA_META_HEADER_FLAG_DISCONT |
                              SPA_META_HEADER_FLAG_CORRUPTED |
                              SPA_META_HEADER_FLAG_GAP)) != 0) {
            *reason = 7; return NULL;
        }
        *seq = header->seq;
    } else {
        if ((data->flags & SPA_DATA_FLAG_WRITABLE) == 0) { *reason = 8; return NULL; }
        chunk->size = QUANTUM * sizeof(float);
        chunk->stride = (int32_t)sizeof(float);
        chunk->flags = SPA_CHUNK_FLAG_NONE;
        struct spa_meta_header *header =
            spa_buffer_find_meta_data(buffer, SPA_META_Header, sizeof(*header));
        if (header != NULL) {
            header->flags = 0;
            header->seq = *seq;
        }
    }
    return (float *)((uint8_t *)data->data + chunk->offset);
}
static void capture_input_metadata(struct backend *b, unsigned index,
                                   const struct pw_buffer *pw_buffer) {
    bool uncaptured = false;
    if (!atomic_compare_exchange_strong(&b->bad_input_metadata_captured[index],
                                        &uncaptured, true)) return;
    if (pw_buffer == NULL || pw_buffer->buffer == NULL) return;
    const struct spa_buffer *buffer = pw_buffer->buffer;
    if (buffer->n_datas != 1 || buffer->datas == NULL) return;
    const struct spa_data *data = &buffer->datas[0];
    if (data->chunk != NULL) {
        atomic_store(&b->bad_input_chunk_size[index], data->chunk->size);
        atomic_store(&b->bad_input_chunk_stride[index], (uint32_t)data->chunk->stride);
        atomic_store(&b->bad_input_chunk_flags[index], data->chunk->flags);
    }
    const struct spa_meta_header *header =
        spa_buffer_find_meta_data(buffer, SPA_META_Header, sizeof(*header));
    if (header != NULL) {
        atomic_store(&b->bad_input_header_present[index], 1);
        atomic_store(&b->bad_input_header_flags[index], header->flags);
        atomic_store(&b->bad_input_header_seq[index], header->seq);
    }
}
static bool prevalid_sentinel(const struct pw_buffer *pw_buffer) {
    const struct spa_buffer *buffer = pw_buffer->buffer;
    const struct spa_meta_header *header =
        spa_buffer_find_meta_data(buffer, SPA_META_Header, sizeof(*header));
    if (header == NULL ||
        header->flags != (SPA_META_HEADER_FLAG_GAP | SPA_META_HEADER_FLAG_CORRUPTED) ||
        header->offset != 0 || header->pts != -1 || header->dts_offset != 0 ||
        header->seq != 0) return false;
    const struct spa_data *data = &buffer->datas[0];
    const float *pcm = (const float *)((const uint8_t *)data->data + data->chunk->offset);
    for (uint32_t i = 0; i < QUANTUM; i++)
        if (pcm[i] != 0.0f) return false;
    return true;
}
static void invalidate_output(struct pw_buffer *pw_buffer) {
    if (pw_buffer == NULL || pw_buffer->buffer == NULL) return;
    struct spa_meta_header *header = spa_buffer_find_meta_data(
        pw_buffer->buffer, SPA_META_Header, sizeof(*header));
    if (header != NULL) {
        header->flags = SPA_META_HEADER_FLAG_GAP | SPA_META_HEADER_FLAG_CORRUPTED;
        header->seq = 0;
    }
}
static void recycle(struct backend *b, struct pw_buffer *raw, struct pw_buffer *reference,
                    struct pw_buffer *clean) {
    if (raw != NULL && pw_filter_queue_buffer(b->raw_port, raw) < 0) poison(b, FAIL_BUFFER);
    if (reference != NULL && pw_filter_queue_buffer(b->reference_port, reference) < 0)
        poison(b, FAIL_BUFFER);
    if (clean != NULL && pw_filter_queue_buffer(b->clean_port, clean) < 0) poison(b, FAIL_BUFFER);
}
static void on_process(void *userdata, struct spa_io_position *position) {
    struct backend *b = userdata;
    uint32_t callbacks = atomic_fetch_add(&b->callback_count, 1);
    if (position != NULL) {
        if (callbacks == 0) atomic_store(&b->first_position_tick, position->clock.position);
        atomic_store(&b->last_position_state, (uint32_t)position->state);
        atomic_store(&b->last_position_duration, (uint32_t)position->clock.duration);
        atomic_store(&b->last_position_clock_id, position->clock.id);
        atomic_store(&b->last_position_tick, position->clock.position);
        atomic_store(&b->last_position_xrun, position->clock.xrun);
    }
    if (position == NULL || position->clock.duration != QUANTUM) {
        atomic_store(&b->bad_position_present, position != NULL);
        if (position != NULL) {
            atomic_store(&b->bad_position_duration, (uint32_t)position->clock.duration);
            atomic_store(&b->bad_position_state, (uint32_t)position->state);
            atomic_store(&b->bad_position_rate_num, position->clock.rate.num);
            atomic_store(&b->bad_position_rate_denom, position->clock.rate.denom);
        }
        poison(b, FAIL_GRAPH_STATE);
        return;
    }
    struct pw_buffer *raw_buffer = pw_filter_dequeue_buffer(b->raw_port);
    struct pw_buffer *reference_buffer = pw_filter_dequeue_buffer(b->reference_port);
    struct pw_buffer *clean_buffer = pw_filter_dequeue_buffer(b->clean_port);
    uint32_t head = atomic_load_explicit(&b->head, memory_order_relaxed);
    uint64_t raw_seq = 0, reference_seq = 0, output_seq = head;
    uint32_t buffer_reason[3];
    float *raw = checked_pcm(raw_buffer, true, &raw_seq, &buffer_reason[0]);
    float *reference = checked_pcm(reference_buffer, true, &reference_seq, &buffer_reason[1]);
    float *out = checked_pcm(clean_buffer, false, &output_seq, &buffer_reason[2]);
    if (atomic_load_explicit(&b->failure, memory_order_relaxed) != FAIL_NONE ||
        atomic_load_explicit(&b->stopping, memory_order_relaxed)) goto muted;
    if (atomic_load_explicit(&b->started, memory_order_relaxed) &&
        !atomic_load_explicit(&b->have_valid_frame, memory_order_relaxed) &&
        out != NULL && position->clock.rate.num != 0 &&
        position->clock.rate.denom == RATE * position->clock.rate.num &&
        position->clock.id != SPA_ID_INVALID) {
        if (raw_buffer == NULL && reference_buffer == NULL) {
            atomic_fetch_add(&b->prevalid_quarantine_count, 1);
            atomic_fetch_add(&b->prevalid_empty_count, 1);
            goto muted;
        }
        if (raw == NULL && reference == NULL &&
            buffer_reason[0] == 7 && buffer_reason[1] == 7 &&
            prevalid_sentinel(raw_buffer) && prevalid_sentinel(reference_buffer)) {
            atomic_fetch_add(&b->prevalid_quarantine_count, 1);
            atomic_fetch_add(&b->prevalid_sentinel_count, 1);
            goto muted;
        }
    }
    if (!atomic_load_explicit(&b->started, memory_order_relaxed) ||
        raw == NULL || reference == NULL || out == NULL) {
        for (unsigned i = 0; i < 3; i++)
            atomic_store(&b->bad_buffer_reason[i], buffer_reason[i]);
        capture_input_metadata(b, 0, raw_buffer);
        capture_input_metadata(b, 1, reference_buffer);
        atomic_store(&b->bad_buffer_state, (uint32_t)position->state);
        atomic_store(&b->bad_buffer_started, atomic_load(&b->started));
        poison(b, FAIL_BUFFER);
        goto muted;
    }
    const struct spa_io_clock *clock = &position->clock;
    if (clock->rate.num == 0 || clock->rate.denom != RATE * clock->rate.num ||
        clock->id == SPA_ID_INVALID) {
        poison(b, FAIL_CLOCK);
        goto muted;
    }
    if (b->have_clock) {
        if (clock->id != b->first_clock_id ||
            clock->rate.num != b->first_rate_num ||
            clock->rate.denom != b->first_rate_denom ||
            clock->xrun != b->last_xrun || clock->position != b->next_position) {
            atomic_store(&b->bad_gap_kind, 1);
            atomic_store(&b->bad_gap_expected_position, b->next_position);
            atomic_store(&b->bad_gap_observed_position, clock->position);
            atomic_store(&b->bad_gap_expected_xrun, b->last_xrun);
            atomic_store(&b->bad_gap_observed_xrun, clock->xrun);
            atomic_store(&b->bad_gap_expected_clock, b->first_clock_id);
            atomic_store(&b->bad_gap_observed_clock, clock->id);
            poison(b, FAIL_GAP);
            goto muted;
        }
    } else {
        b->have_clock = true;
        b->first_clock_id = clock->id;
        b->first_rate_num = clock->rate.num;
        b->first_rate_denom = clock->rate.denom;
    }
    uint64_t expected_seq = b->producer_seq_base + clock->position / QUANTUM;
    if (raw_seq != expected_seq || reference_seq != expected_seq ||
        raw_seq != reference_seq ||
        (b->have_port_seq &&
         (raw_seq != b->last_raw_seq + 1 || reference_seq != b->last_reference_seq + 1))) {
        atomic_store(&b->bad_gap_kind, 2);
        atomic_store(&b->bad_gap_expected_raw_seq, expected_seq);
        atomic_store(&b->bad_gap_observed_raw_seq, raw_seq);
        atomic_store(&b->bad_gap_expected_reference_seq, expected_seq);
        atomic_store(&b->bad_gap_observed_reference_seq, reference_seq);
        poison(b, FAIL_GAP);
        goto muted;
    }
    if (!finite_pcm(raw) || !finite_pcm(reference)) {
        poison(b, FAIL_PCM);
        goto muted;
    }
    uint32_t tail = atomic_load_explicit(&b->tail, memory_order_acquire);
    if (head - tail >= QUEUE_SLOTS) {
        poison(b, FAIL_QUEUE);
        goto muted;
    }
    struct frame *frame = &b->frames[head % QUEUE_SLOTS];
    frame->sequence = head;
    frame->position = clock->position;
    frame->duration = clock->duration;
    frame->xrun = clock->xrun;
    frame->clock_id = clock->id;
    frame->rate_num = clock->rate.num;
    frame->rate_denom = clock->rate.denom;
    memcpy(frame->raw, raw, QUANTUM * sizeof(float));
    memcpy(frame->reference, reference, QUANTUM * sizeof(float));
    const float *rec_channels[] = { raw };
    const float *play_channels[] = { reference };
    float *out_channels[] = { out };
    int aec_status = spa_audio_aec_run(b->aec, rec_channels, play_channels, out_channels, QUANTUM);
    if (aec_status < 0 || atomic_exchange(&b->inject_aec_error, false) || !finite_pcm(out)) {
        poison(b, FAIL_AEC);
        goto muted;
    }
    memcpy(frame->clean, out, QUANTUM * sizeof(float));
    b->next_position = clock->position + QUANTUM;
    b->last_xrun = clock->xrun;
    b->last_raw_seq = raw_seq;
    b->last_reference_seq = reference_seq;
    b->have_port_seq = true;
    atomic_store(&b->have_valid_frame, true);
    atomic_store_explicit(&b->head, head + 1, memory_order_release);
    recycle(b, raw_buffer, reference_buffer, clean_buffer);
    if (head == 0 && atomic_load(&b->failure) == FAIL_NONE)
        atomic_store_explicit(&b->armed_ready, true, memory_order_release);
    return;
muted:
    if (out != NULL) memset(out, 0, QUANTUM * sizeof(float));
    invalidate_output(clean_buffer);
    recycle(b, raw_buffer, reference_buffer, clean_buffer);
}
static void on_state(void *userdata, enum pw_filter_state old_state,
                     enum pw_filter_state state, const char *error) {
    struct backend *b = userdata;
    (void)error;
    if (state == PW_FILTER_STATE_ERROR ||
        (old_state == PW_FILTER_STATE_STREAMING && state != PW_FILTER_STATE_STREAMING &&
         atomic_load(&b->have_valid_frame) && !atomic_load(&b->stopping)))
        poison(b, FAIL_GRAPH_STATE);
    if (state == PW_FILTER_STATE_PAUSED || state == PW_FILTER_STATE_STREAMING) {
        uint32_t id = pw_filter_get_node_id(b->filter);
        if (id != PW_ID_INVALID) {
            if (atomic_load(&b->ready)) {
                if (id != b->node_id) poison(b, FAIL_LINK);
            } else {
                b->node_id = id;
                atomic_store_explicit(&b->ready, true, memory_order_release);
            }
        }
    }
}
static bool id_from_dict(const struct spa_dict *props, const char *key, uint32_t *id) {
    const char *text = spa_dict_lookup(props, key);
    if (text == NULL || *text == '\0') return false;
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || *end != '\0' || value > UINT32_MAX) return false;
    *id = (uint32_t)value;
    return true;
}
static bool peer_is_virtual(const struct backend *b, uint32_t id) {
    for (size_t i = 0; i < MAX_NODES; i++)
        if (b->nodes[i].present && b->nodes[i].id == id)
            return b->nodes[i].seen_info && b->nodes[i].virtual_node;
    return false;
}
static uint32_t own_port(const struct backend *b, const char *name) {
    for (size_t i = 0; i < MAX_PORTS; i++)
        if (b->ports[i].present && b->ports[i].node_id == b->node_id &&
            strcmp(b->ports[i].name, name) == 0)
            return b->ports[i].id;
    return SPA_ID_INVALID;
}
static bool find_binding(const struct backend *b, uint32_t port, bool input,
                         struct binding *binding) {
    bool found = false;
    for (size_t i = 0; i < MAX_LINKS; i++) {
        const struct link_fact *link = &b->links[i];
        if (!link->present) continue;
        bool matches = input ? (link->input_node == b->node_id && link->input_port == port)
                             : (link->output_node == b->node_id && link->output_port == port);
        if (!matches) continue;
        if (found) return false;
        uint32_t peer_node = input ? link->output_node : link->input_node;
        if (!peer_is_virtual(b, peer_node)) return false;
        *binding = (struct binding) {
            .local_port = port, .link_id = link->id,
            .peer_node = peer_node,
            .peer_port = input ? link->output_port : link->input_port,
        };
        found = true;
    }
    return found;
}
static void start_wait(struct backend *b, const char *reason,
                       uint32_t raw, uint32_t reference, uint32_t clean) {
    if (b->diagnostic_count++ < 8) {
        unsigned node_count = 0, link_count = 0;
        uint32_t raw_link = SPA_ID_INVALID, peer_node = SPA_ID_INVALID;
        bool peer_seen = false, peer_virtual = false;
        for (size_t i = 0; i < MAX_NODES; i++)
            if (b->nodes[i].present) node_count++;
        for (size_t i = 0; i < MAX_LINKS; i++) {
            if (!b->links[i].present) continue;
            link_count++;
            if (b->links[i].input_node == b->node_id && b->links[i].input_port == raw) {
                raw_link = b->links[i].id;
                peer_node = b->links[i].output_node;
            }
        }
        for (size_t i = 0; i < MAX_NODES; i++)
            if (b->nodes[i].present && b->nodes[i].id == peer_node) {
                peer_seen = true;
                peer_virtual = b->nodes[i].virtual_node;
            }
        fprintf(stderr, "AEC_START_WAIT reason=%s node=%u ports=%u,%u,%u nodes=%u links=%u raw_link=%u peer=%u seen=%d virtual=%d\n",
                reason, b->node_id, raw, reference, clean, node_count, link_count,
                raw_link, peer_node, peer_seen, peer_virtual);
    }
}
static void maybe_start(struct backend *b) {
    if (!b->start_requested || atomic_load(&b->started) ||
        atomic_load(&b->failure) != FAIL_NONE || b->node_id == SPA_ID_INVALID)
        return;
    uint32_t raw = own_port(b, "raw");
    uint32_t reference = own_port(b, "reference");
    uint32_t clean = own_port(b, "clean");
    if (raw == SPA_ID_INVALID || reference == SPA_ID_INVALID || clean == SPA_ID_INVALID) {
        start_wait(b, "ports", raw, reference, clean);
        return;
    }
    struct binding bindings[3];
    if (!find_binding(b, raw, true, &bindings[0])) {
        start_wait(b, "raw-link-or-peer", raw, reference, clean);
        return;
    }
    if (!find_binding(b, reference, true, &bindings[1])) {
        start_wait(b, "reference-link-or-peer", raw, reference, clean);
        return;
    }
    if (!find_binding(b, clean, false, &bindings[2])) {
        start_wait(b, "clean-link-or-peer", raw, reference, clean);
        return;
    }
    memcpy(b->bindings, bindings, sizeof(bindings));
    atomic_store_explicit(&b->links_ready, true, memory_order_release);
    atomic_store(&b->started, true);
    if (pw_filter_set_active(b->filter, true) < 0)
        poison(b, FAIL_GRAPH_STATE);
}
static void on_node_info(void *userdata, const struct pw_node_info *info) {
    struct node_fact *fact = userdata;
    struct backend *b = fact->owner;
    if (!fact->present || info == NULL || info->id != fact->id) {
        poison(b, FAIL_LINK);
        return;
    }
    if ((info->change_mask & PW_NODE_CHANGE_MASK_PROPS) != 0) {
        const char *virtual_text = info->props == NULL ? NULL :
            spa_dict_lookup(info->props, PW_KEY_NODE_VIRTUAL);
        fact->seen_info = true;
        fact->virtual_node = virtual_text != NULL && spa_atob(virtual_text);
        if (atomic_load(&b->started) && !fact->virtual_node) {
            bool required = fact->id == b->node_id;
            for (unsigned i = 0; i < 3; i++)
                required = required || fact->id == b->bindings[i].peer_node;
            if (required) poison(b, FAIL_LINK);
        }
    }
    maybe_start(b);
}
static const struct pw_node_events node_events = {
    .version = PW_VERSION_NODE_EVENTS,
    .info = on_node_info,
};
static void on_global(void *userdata, uint32_t id, uint32_t permissions,
                      const char *type, uint32_t version, const struct spa_dict *props) {
    struct backend *b = userdata;
    (void)permissions; (void)version;
    if (strcmp(type, PW_TYPE_INTERFACE_Node) == 0) {
        for (size_t i = 0; i < MAX_NODES; i++) {
            if (!b->nodes[i].present) {
                struct node_fact *fact = &b->nodes[i];
                fact->owner = b;
                fact->id = id;
                fact->present = true;
                fact->seen_info = false;
                fact->virtual_node = false;
                fact->proxy = pw_registry_bind(b->registry, id, PW_TYPE_INTERFACE_Node,
                                               PW_VERSION_NODE, 0);
                if (fact->proxy == NULL ||
                    pw_node_add_listener(fact->proxy, &fact->listener,
                                         &node_events, fact) < 0)
                    poison(b, FAIL_LINK);
                return;
            }
        }
        poison(b, FAIL_LINK);
    } else if (strcmp(type, PW_TYPE_INTERFACE_Port) == 0) {
        if (props == NULL) { poison(b, FAIL_LINK); return; }
        uint32_t node;
        const char *name = spa_dict_lookup(props, PW_KEY_PORT_NAME);
        if (!id_from_dict(props, PW_KEY_NODE_ID, &node) || name == NULL ||
            strlen(name) >= sizeof(b->ports[0].name)) { poison(b, FAIL_LINK); return; }
        if (atomic_load(&b->started) && node == b->node_id) {
            poison(b, FAIL_LINK); return;
        }
        for (size_t i = 0; i < MAX_PORTS; i++) {
            if (!b->ports[i].present) {
                b->ports[i].id = id;
                b->ports[i].node_id = node;
                memcpy(b->ports[i].name, name, strlen(name) + 1);
                b->ports[i].present = true;
                maybe_start(b);
                return;
            }
        }
        poison(b, FAIL_LINK);
    } else if (strcmp(type, PW_TYPE_INTERFACE_Link) == 0) {
        if (props == NULL) { poison(b, FAIL_LINK); return; }
        struct link_fact link = { .id = id, .present = true };
        if (!id_from_dict(props, PW_KEY_LINK_INPUT_NODE, &link.input_node) ||
            !id_from_dict(props, PW_KEY_LINK_INPUT_PORT, &link.input_port) ||
            !id_from_dict(props, PW_KEY_LINK_OUTPUT_NODE, &link.output_node) ||
            !id_from_dict(props, PW_KEY_LINK_OUTPUT_PORT, &link.output_port)) {
            poison(b, FAIL_LINK); return;
        }
        if (atomic_load(&b->started) &&
            (link.input_node == b->node_id || link.output_node == b->node_id)) {
            poison(b, FAIL_LINK); return;
        }
        for (size_t i = 0; i < MAX_LINKS; i++) {
            if (!b->links[i].present) {
                b->links[i] = link;
                maybe_start(b);
                return;
            }
        }
        poison(b, FAIL_LINK);
    }
}
static void on_global_remove(void *userdata, uint32_t id) {
    struct backend *b = userdata;
    for (size_t i = 0; i < MAX_LINKS; i++)
        if (b->links[i].present && b->links[i].id == id) {
            b->links[i].present = false;
            if (atomic_load(&b->started)) poison(b, FAIL_LINK);
        }
    for (size_t i = 0; i < MAX_PORTS; i++)
        if (b->ports[i].present && b->ports[i].id == id) {
            b->ports[i].present = false;
            if (atomic_load(&b->started)) poison(b, FAIL_LINK);
        }
    for (size_t i = 0; i < MAX_NODES; i++)
        if (b->nodes[i].present && b->nodes[i].id == id) {
            b->nodes[i].present = false;
            if (b->nodes[i].proxy != NULL) {
                pw_proxy_destroy((struct pw_proxy *)b->nodes[i].proxy);
                b->nodes[i].proxy = NULL;
            }
            if (atomic_load(&b->started)) poison(b, FAIL_LINK);
        }
}
static const struct pw_registry_events registry_events = {
    .version = PW_VERSION_REGISTRY_EVENTS,
    .global = on_global,
    .global_remove = on_global_remove,
};

static const struct pw_filter_events filter_events = {
    .version = PW_VERSION_FILTER_EVENTS,
    .state_changed = on_state,
    .process = on_process,
};

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
static bool send_body(struct backend *b, const uint8_t *body, uint32_t length) {
    uint8_t prefix[4];
    put32(prefix, length);
    return write_all_bounded(b->ipc_fd, prefix, sizeof(prefix)) &&
           write_all_bounded(b->ipc_fd, body, length);
}
static bool send_hello(struct backend *b) {
    uint8_t body[36] = { IPC_HELLO, IPC_VERSION, 0, 0 };
    put64(body + 4, b->session);
    put64(body + 12, b->generation);
    put32(body + 20, b->node_id);
    put32(body + 24, RATE);
    put32(body + 28, QUANTUM);
    put32(body + 32, QUANTUM);
    return send_body(b, body, sizeof(body));
}
static bool send_links(struct backend *b) {
    uint8_t body[72] = { IPC_LINKS, IPC_VERSION, 0, 0 };
    put64(body + 4, b->session);
    put64(body + 12, b->generation);
    put32(body + 20, b->node_id);
    for (unsigned i = 0; i < 3; i++) {
        put32(body + 24 + i * 16, b->bindings[i].local_port);
        put32(body + 28 + i * 16, b->bindings[i].link_id);
        put32(body + 32 + i * 16, b->bindings[i].peer_node);
        put32(body + 36 + i * 16, b->bindings[i].peer_port);
    }
    return send_body(b, body, sizeof(body));
}
static bool send_armed(struct backend *b) {
    uint8_t body[24] = { IPC_ARMED, IPC_VERSION, 0, 0 };
    put64(body + 4, b->session);
    put64(body + 12, b->generation);
    put32(body + 20, b->node_id);
    return send_body(b, body, sizeof(body));
}
static bool send_fatal(struct backend *b, uint32_t reason) {
    uint8_t body[32] = { IPC_FATAL, IPC_VERSION, 0, 0 };
    put64(body + 4, b->session);
    put64(body + 12, b->generation);
    put32(body + 20, reason);
    put64(body + 24, atomic_load(&b->head));
    return send_body(b, body, sizeof(body));
}
static bool send_frame(struct backend *b, const struct frame *frame) {
    uint8_t body[IPC_MAX_BODY] = { IPC_FRAME, IPC_VERSION, 0, 0 };
    put64(body + 4, b->session);
    put64(body + 12, b->generation);
    put64(body + 20, frame->sequence);
    put32(body + 28, frame->clock_id);
    put32(body + 32, QUANTUM);
    put64(body + 36, frame->position);
    put64(body + 44, frame->duration);
    put64(body + 52, frame->xrun);
    put32(body + 60, frame->rate_num);
    put32(body + 64, frame->rate_denom);
    put32(body + 68, b->node_id);
    size_t offset = 72;
    for (uint32_t i = 0; i < QUANTUM; i++) {
        memcpy(body + offset, &frame->raw[i], 4); offset += 4;
        memcpy(body + offset, &frame->reference[i], 4); offset += 4;
        memcpy(body + offset, &frame->clean[i], 4); offset += 4;
    }
    return send_body(b, body, (uint32_t)offset);
}
static void *writer_main(void *userdata) {
    struct backend *b = userdata;
    bool fatal_sent = false;
    bool no_progress_reported = false;
    struct timespec no_progress_start = { 0 };
    while (!atomic_load(&b->writer_stop)) {
        if (atomic_load(&b->ready) && !atomic_load(&b->hello_sent)) {
            if (!send_hello(b)) { poison(b, FAIL_IPC); break; }
            atomic_store(&b->hello_sent, true);
        }
        if (atomic_load_explicit(&b->links_ready, memory_order_acquire) &&
            !atomic_load(&b->links_sent)) {
            if (!send_links(b)) { poison(b, FAIL_IPC); break; }
            atomic_store(&b->links_sent, true);
        }
        uint32_t reason = atomic_load(&b->failure);
        if (reason != FAIL_NONE) {
            if (reason == FAIL_GAP) {
                fprintf(stderr,
                    "AEC_GAP kind=%u expected_pos=%llu observed_pos=%llu expected_clock=%u observed_clock=%u expected_xrun=%llu observed_xrun=%llu expected_seq=%llu,%llu observed_seq=%llu,%llu\n",
                    atomic_load(&b->bad_gap_kind),
                    (unsigned long long)atomic_load(&b->bad_gap_expected_position),
                    (unsigned long long)atomic_load(&b->bad_gap_observed_position),
                    atomic_load(&b->bad_gap_expected_clock),
                    atomic_load(&b->bad_gap_observed_clock),
                    (unsigned long long)atomic_load(&b->bad_gap_expected_xrun),
                    (unsigned long long)atomic_load(&b->bad_gap_observed_xrun),
                    (unsigned long long)atomic_load(&b->bad_gap_expected_raw_seq),
                    (unsigned long long)atomic_load(&b->bad_gap_expected_reference_seq),
                    (unsigned long long)atomic_load(&b->bad_gap_observed_raw_seq),
                    (unsigned long long)atomic_load(&b->bad_gap_observed_reference_seq));
            }
            if (reason == FAIL_BUFFER) {
                fprintf(stderr,
                    "AEC_BUFFER_PRECONDITION site=%u reasons=%u,%u,%u state=%u started=%d callback=%u position=%llu\n",
                    atomic_load(&b->bad_buffer_site),
                    atomic_load(&b->bad_buffer_reason[0]),
                    atomic_load(&b->bad_buffer_reason[1]),
                    atomic_load(&b->bad_buffer_reason[2]),
                    atomic_load(&b->bad_buffer_state),
                    atomic_load(&b->bad_buffer_started),
                    atomic_load(&b->bad_buffer_callback),
                    (unsigned long long)atomic_load(&b->bad_buffer_position));
                fprintf(stderr,
                    "AEC_BUFFER_META raw=%u,%u,%u,%u,%u,%llu ref=%u,%u,%u,%u,%u,%llu\n",
                    atomic_load(&b->bad_input_chunk_size[0]),
                    atomic_load(&b->bad_input_chunk_stride[0]),
                    atomic_load(&b->bad_input_chunk_flags[0]),
                    atomic_load(&b->bad_input_header_present[0]),
                    atomic_load(&b->bad_input_header_flags[0]),
                    (unsigned long long)atomic_load(&b->bad_input_header_seq[0]),
                    atomic_load(&b->bad_input_chunk_size[1]),
                    atomic_load(&b->bad_input_chunk_stride[1]),
                    atomic_load(&b->bad_input_chunk_flags[1]),
                    atomic_load(&b->bad_input_header_present[1]),
                    atomic_load(&b->bad_input_header_flags[1]),
                    (unsigned long long)atomic_load(&b->bad_input_header_seq[1]));
            }
            if (reason == FAIL_GRAPH_STATE) {
                fprintf(stderr,
                    "AEC_CLOCK_PRECONDITION position_present=%d duration=%u state=%u rate=%u/%u\n",
                    atomic_load(&b->bad_position_present),
                    atomic_load(&b->bad_position_duration),
                    atomic_load(&b->bad_position_state),
                    atomic_load(&b->bad_position_rate_num),
                    atomic_load(&b->bad_position_rate_denom));
            }
            if (!fatal_sent) { (void)send_fatal(b, reason); fatal_sent = true; }
            break;
        }
        if (atomic_load_explicit(&b->armed_ready, memory_order_acquire) &&
            atomic_load(&b->links_sent) && !atomic_load(&b->armed_sent)) {
            if (!send_armed(b)) { poison(b, FAIL_IPC); break; }
            atomic_store(&b->armed_sent, true);
        }
        uint32_t tail = atomic_load_explicit(&b->tail, memory_order_relaxed);
        uint32_t head = atomic_load_explicit(&b->head, memory_order_acquire);
        if (atomic_load(&b->started) && head == 0 && !no_progress_reported) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (no_progress_start.tv_sec == 0) {
                no_progress_start = now;
            } else if ((now.tv_sec - no_progress_start.tv_sec) * 1000000000LL +
                       now.tv_nsec - no_progress_start.tv_nsec >= 1000000000LL) {
                fprintf(stderr,
                    "AEC_NO_PROGRESS callbacks=%u state=%u duration=%u clock=%u first=%llu last=%llu xrun=%llu started=%d valid=%d\n",
                    atomic_load(&b->callback_count),
                    atomic_load(&b->last_position_state),
                    atomic_load(&b->last_position_duration),
                    atomic_load(&b->last_position_clock_id),
                    (unsigned long long)atomic_load(&b->first_position_tick),
                    (unsigned long long)atomic_load(&b->last_position_tick),
                    (unsigned long long)atomic_load(&b->last_position_xrun),
                    atomic_load(&b->started),
                    atomic_load(&b->have_valid_frame));
                no_progress_reported = true;
            }
        }
        if (tail != head && atomic_load(&b->armed_sent)) {
            if (!send_frame(b, &b->frames[tail % QUEUE_SLOTS])) {
                poison(b, FAIL_IPC);
                break;
            }
            atomic_store_explicit(&b->tail, tail + 1, memory_order_release);
            continue;
        }
        struct timespec pause = { .tv_sec = 0, .tv_nsec = 1000000 };
        nanosleep(&pause, NULL);
    }
    return NULL;
}

static void on_ipc(void *userdata, int fd, uint32_t mask) {
    struct backend *b = userdata;
    if (mask & (SPA_IO_ERR | SPA_IO_HUP)) {
        atomic_store(&b->stopping, true);
        pw_main_loop_quit(b->loop);
        return;
    }
    ssize_t got = recv(fd, b->inbound + b->inbound_used,
                       sizeof(b->inbound) - b->inbound_used, MSG_DONTWAIT);
    if (got <= 0) {
        if (got < 0 && (errno == EAGAIN || errno == EINTR)) return;
        atomic_store(&b->stopping, true);
        pw_main_loop_quit(b->loop);
        return;
    }
    b->inbound_used += (size_t)got;
    if (b->inbound_used != sizeof(b->inbound)) return;
    b->inbound_used = 0;
    if (get32(b->inbound) != 4 || b->inbound[5] != IPC_VERSION ||
        b->inbound[6] != 0 || b->inbound[7] != 0) {
        poison(b, FAIL_IPC);
        pw_main_loop_quit(b->loop);
        return;
    }
    if (b->inbound[4] == IPC_STOP) {
        atomic_store(&b->stopping, true);
        pw_main_loop_quit(b->loop);
    } else if (b->inbound[4] == IPC_START && atomic_load(&b->hello_sent) &&
               !b->start_requested && atomic_load(&b->failure) == FAIL_NONE) {
        b->start_requested = true;
        fprintf(stderr, "AEC_START_RECEIVED node=%u\n", b->node_id);
        maybe_start(b);
    } else if (b->inbound[4] == IPC_INJECT_AEC_ERROR &&
               atomic_load(&b->started) && atomic_load(&b->have_valid_frame) &&
               atomic_load(&b->failure) == FAIL_NONE) {
        atomic_store(&b->inject_aec_error, true);
        fprintf(stderr, "AEC_FAULT_ARMED kind=aec-error session=%016llx\n",
                (unsigned long long)b->session);
    } else {
        poison(b, FAIL_IPC);
        pw_main_loop_quit(b->loop);
    }
}
static void on_term(void *userdata, int signum) {
    struct backend *b = userdata;
    (void)signum;
    atomic_store(&b->stopping, true);
    pw_main_loop_quit(b->loop);
}

static bool load_aec(struct backend *b) {
    b->plugin_library = dlopen(TRANSLATOR_AEC_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (b->plugin_library == NULL) return false;
    spa_handle_factory_enum_func_t factory_enum =
        (spa_handle_factory_enum_func_t)dlsym(b->plugin_library, SPA_HANDLE_FACTORY_ENUM_FUNC_NAME);
    if (factory_enum == NULL) return false;
    const struct spa_handle_factory *factory = NULL;
    uint32_t index = 0;
    bool found = false;
    while (factory_enum(&factory, &index) > 0) {
        if (strcmp(factory->name, SPA_NAME_AEC) == 0) { found = true; break; }
    }
    if (!found) return false;
    size_t bytes = spa_handle_factory_get_size(factory, NULL);
    if (bytes == 0 || bytes > 1024 * 1024) return false;
    b->plugin_handle = calloc(1, bytes);
    if (b->plugin_handle == NULL) return false;
    struct spa_support support[16];
    uint32_t count = pw_get_support(support, 16);
    if (spa_handle_factory_init(factory, b->plugin_handle, NULL, support, count) < 0) {
        free(b->plugin_handle);
        b->plugin_handle = NULL;
        return false;
    }
    if (spa_handle_get_interface(b->plugin_handle, SPA_TYPE_INTERFACE_AUDIO_AEC,
                                 (void **)&b->aec) < 0 || b->aec == NULL) return false;
    struct spa_audio_info_raw info = {
        .format = SPA_AUDIO_FORMAT_F32P, .rate = RATE, .channels = 1,
        .position = { SPA_AUDIO_CHANNEL_MONO },
    };
    static const struct spa_dict_item config_items[] = {
        { "webrtc.noise_suppression", "false" },
        { "webrtc.high_pass_filter", "false" },
    };
    const struct spa_dict config = SPA_DICT_INIT(config_items, 2);
    if (spa_audio_aec_init(b->aec, &config, &info) < 0) return false;
    return b->aec->latency != NULL && strcmp(b->aec->latency, "480/48000") == 0;
}

static void cleanup(struct backend *b) {
    atomic_store(&b->writer_stop, true);
    if (b->writer_started) pthread_join(b->writer, NULL);
    uint32_t quarantined = atomic_load(&b->prevalid_quarantine_count);
    if (quarantined > 0)
        fprintf(stderr, "AEC_PREVALID quarantined=%u empty=%u sentinel=%u\n",
                quarantined, atomic_load(&b->prevalid_empty_count),
                atomic_load(&b->prevalid_sentinel_count));
    struct pw_loop *loop = b->loop == NULL ? NULL : pw_main_loop_get_loop(b->loop);
    if (loop != NULL && b->ipc_source != NULL) pw_loop_destroy_source(loop, b->ipc_source);
    if (loop != NULL && b->term_source != NULL) pw_loop_destroy_source(loop, b->term_source);
    if (b->filter != NULL) pw_filter_destroy(b->filter);
    for (size_t i = 0; i < MAX_NODES; i++)
        if (b->nodes[i].proxy != NULL)
            pw_proxy_destroy((struct pw_proxy *)b->nodes[i].proxy);
    if (b->registry != NULL) pw_proxy_destroy((struct pw_proxy *)b->registry);
    if (b->core != NULL) pw_core_disconnect(b->core);
    if (b->context != NULL) pw_context_destroy(b->context);
    if (b->loop != NULL) pw_main_loop_destroy(b->loop);
    if (b->plugin_handle != NULL) {
        spa_handle_clear(b->plugin_handle);
        free(b->plugin_handle);
    }
    if (b->plugin_library != NULL) dlclose(b->plugin_library);
    if (b->ipc_fd >= 0) close(b->ipc_fd);
    pw_deinit();
}

static bool parse_fd(const char *text, int *fd) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 3 || value > 1024) return false;
    if (fcntl((int)value, F_GETFD) < 0) return false;
    int type = 0;
    socklen_t length = sizeof(type);
    if (getsockopt((int)value, SOL_SOCKET, SO_TYPE, &type, &length) < 0 || type != SOCK_STREAM)
        return false;
    struct sockaddr_storage peer;
    socklen_t peer_size = sizeof(peer);
    if (getpeername((int)value, (struct sockaddr *)&peer, &peer_size) < 0 ||
        peer.ss_family != AF_UNIX) return false;
    *fd = (int)value;
    return true;
}
static bool parse_session(const char *text, uint64_t *session) {
    if (strlen(text) != 16) return false;
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 16);
    if (errno != 0 || *end != '\0' || value == 0) return false;
    *session = (uint64_t)value;
    return true;
}

int translator_aec_backend_run(int argc, const char **argv) {
    if (argc == 2 && strcmp(argv[1], "--abi") == 0) {
        puts("translator-aec-backend wire=1 pipewire=1.0.5 rate=48000 quantum=480 aec=webrtc ns=false latency=480");
        return 0;
    }
    int pw_fd = -1, ipc_fd = -1;
    uint64_t session = 0;
    if (argc != 7 || strcmp(argv[1], "--pipewire-fd") != 0 ||
        strcmp(argv[3], "--ipc-fd") != 0 || strcmp(argv[5], "--session-id") != 0 ||
        !parse_fd(argv[2], &pw_fd) || !parse_fd(argv[4], &ipc_fd) ||
        pw_fd == ipc_fd || !parse_session(argv[6], &session)) {
        fputs("translator-aec-backend: private connected FDs and session-id required\n", stderr);
        return 2;
    }
    struct backend *b = calloc(1, sizeof(*b));
    if (b == NULL) return 2;
    b->ipc_fd = ipc_fd;
    b->session = session;
    b->generation = session;
    b->producer_seq_base = (session & UINT64_C(0x3fffffffffffffff)) |
        UINT64_C(0x4000000000000000);
    b->node_id = PW_ID_INVALID;
    pw_init(NULL, NULL);
    int result = 2;
    if (!load_aec(b)) goto out;
    b->loop = pw_main_loop_new(NULL);
    if (b->loop == NULL) goto out;
    struct pw_loop *loop = pw_main_loop_get_loop(b->loop);
    b->context = pw_context_new(loop, NULL, 0);
    if (b->context == NULL) goto out;
    int owned_pw_fd = pw_fd;
    pw_fd = -1;
    b->core = pw_context_connect_fd(b->context, owned_pw_fd, NULL, 0);
    if (b->core == NULL) goto out;
    b->registry = pw_core_get_registry(b->core, PW_VERSION_REGISTRY, 0);
    if (b->registry == NULL) goto out;
    if (pw_registry_add_listener(b->registry, &b->registry_listener,
                                 &registry_events, b) < 0) goto out;
    char name[64];
    snprintf(name, sizeof(name), "translator-aec-%016llx", (unsigned long long)session);
    b->filter = pw_filter_new(b->core, name,
        pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio",
                          PW_KEY_MEDIA_CATEGORY, "Filter",
                          PW_KEY_MEDIA_ROLE, "DSP",
                          PW_KEY_NODE_NAME, name,
                          PW_KEY_NODE_AUTOCONNECT, "false",
                          PW_KEY_NODE_VIRTUAL, "true", NULL));
    if (b->filter == NULL) goto out;
    pw_filter_add_listener(b->filter, &b->filter_listener, &filter_events, b);
    uint8_t pod_buffer[256];
    struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(pod_buffer, sizeof(pod_buffer));
    const struct spa_pod *meta = spa_pod_builder_add_object(&builder,
        SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
        SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
        SPA_PARAM_META_size, SPA_POD_Int(sizeof(struct spa_meta_header)));
    if (meta == NULL) goto out;
    const struct spa_pod *port_params[] = { meta };
    b->raw_port = pw_filter_add_port(b->filter, PW_DIRECTION_INPUT,
        PW_FILTER_PORT_FLAG_MAP_BUFFERS, 0,
        pw_properties_new(PW_KEY_FORMAT_DSP, "32 bit float mono audio",
                          PW_KEY_PORT_NAME, "raw", NULL), port_params, 1);
    b->reference_port = pw_filter_add_port(b->filter, PW_DIRECTION_INPUT,
        PW_FILTER_PORT_FLAG_MAP_BUFFERS, 0,
        pw_properties_new(PW_KEY_FORMAT_DSP, "32 bit float mono audio",
                          PW_KEY_PORT_NAME, "reference", NULL), port_params, 1);
    b->clean_port = pw_filter_add_port(b->filter, PW_DIRECTION_OUTPUT,
        PW_FILTER_PORT_FLAG_MAP_BUFFERS, 0,
        pw_properties_new(PW_KEY_FORMAT_DSP, "32 bit float mono audio",
                          PW_KEY_PORT_NAME, "clean", NULL), port_params, 1);
    if (b->raw_port == NULL || b->reference_port == NULL || b->clean_port == NULL) goto out;
    int flags = fcntl(ipc_fd, F_GETFL, 0);
    if (flags < 0 || fcntl(ipc_fd, F_SETFL, flags | O_NONBLOCK) < 0) goto out;
    b->ipc_source = pw_loop_add_io(loop, ipc_fd, SPA_IO_IN | SPA_IO_ERR | SPA_IO_HUP,
                                    false, on_ipc, b);
    b->term_source = pw_loop_add_signal(loop, SIGTERM, on_term, b);
    if (b->ipc_source == NULL || b->term_source == NULL) goto out;
    if (pthread_create(&b->writer, NULL, writer_main, b) != 0) goto out;
    b->writer_started = true;
    if (pw_filter_connect(b->filter, PW_FILTER_FLAG_INACTIVE | PW_FILTER_FLAG_RT_PROCESS,
                          NULL, 0) < 0) goto out;
    pw_main_loop_run(b->loop);
    result = atomic_load(&b->failure) == FAIL_NONE ? 0 : 1;
out:
    if (pw_fd >= 0) close(pw_fd);
    cleanup(b);
    free(b);
    return result;
}
