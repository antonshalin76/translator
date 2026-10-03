#define _GNU_SOURCE
#include <alsa/asoundlib.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>

#define BLOCK 480


int install_control_policy(void);
int test_control_policy(const char *name, int high_alias);

struct stream {
    snd_pcm_t *pcm;
    snd_pcm_status_t *status;
    snd_pcm_uframes_t buffer;
    snd_pcm_uframes_t period;
    unsigned channels;
    uint64_t timestamp;
    uint64_t current_timestamp;
    uint64_t max_avail;
    unsigned audio_timestamp_type;
    unsigned audio_timestamp_valid;
    unsigned audio_accuracy_reported;
    unsigned audio_accuracy_ns;
    const char *role;
    snd_pcm_tstamp_t sw_timestamp_mode;
    snd_pcm_tstamp_type_t sw_timestamp_type;
    uint64_t transferred;
    uint64_t unknown_timestamps;
};

static struct {
    const char *role;
    int state;
    uint64_t previous_timestamp;
    int64_t timestamp_seconds;
    int64_t timestamp_nanoseconds;
    uint64_t avail;
    uint64_t avail_max;
    int64_t delay;
    unsigned sw_timestamp_mode;
    unsigned sw_timestamp_type;
    uint64_t transferred;
} observation;

static const char *validate_transfer(int64_t transfer) {
    return transfer == BLOCK ? NULL : "short_transfer";
}

static const char *validate_cursor(uint64_t written, int64_t delay,
                                  uint64_t buffer, uint64_t previous,
                                  int running, int64_t transfer,
                                  uint64_t *played) {
    if (!running) return "pcm_not_running";
    const char *reason = validate_transfer(transfer);
    if (reason) return reason;
    if (delay < 0 || (uint64_t)delay > buffer || (uint64_t)delay > written)
        return "invalid_delay";
    *played = written - (uint64_t)delay;
    if (*played < previous) return "played_regression";
    return NULL;
}

static int validate_timestamp(uint64_t previous, uint64_t current, int running,
                              const char **reason) {
    if (!running) { *reason = "pcm_not_running"; return -EPIPE; }
    if (!current) return 1;
    if (current < previous) {
        *reason = "timestamp_regression";
        return -EINVAL;
    }
    return 0;
}

/* Test exports call the same guards used by the live ALSA loop, with no open. */
int translator_aec_physical_cursor_guard(uint64_t written, int64_t delay,
        uint64_t buffer, uint64_t previous, int running, int64_t transfer,
        uint64_t *played) {
    return validate_cursor(written, delay, buffer, previous, running, transfer, played) ? -1 : 0;
}
int translator_aec_physical_timestamp_guard(uint64_t previous, uint64_t current, int running) {
    const char *reason = NULL;
    return validate_timestamp(previous, current, running, &reason);
}

static uint64_t monotonic_ns(void) {
    struct timespec stamp;
    if (clock_gettime(CLOCK_MONOTONIC, &stamp)) return 0;
    return (uint64_t)stamp.tv_sec * UINT64_C(1000000000) + (uint64_t)stamp.tv_nsec;
}

static int configure(struct stream *stream, snd_pcm_stream_t direction,
                     const char **reason) {
    snd_pcm_hw_params_t *hw = NULL;
    snd_pcm_sw_params_t *sw = NULL;
    snd_pcm_info_t *info = NULL;
    int result;
#define CHECK(call, label) do { result = (call); if (result < 0) { *reason = (label); goto done; } } while (0)
    CHECK(snd_pcm_open(&stream->pcm, "hw:0,0", direction, SND_PCM_NONBLOCK), "exclusive_hw_open");
    if (snd_pcm_type(stream->pcm) != SND_PCM_TYPE_HW) { result = -EINVAL; *reason = "not_direct_hw"; goto done; }
    CHECK(snd_pcm_info_malloc(&info), "info_allocation");
    CHECK(snd_pcm_info(stream->pcm, info), "pcm_identity");
    if (snd_pcm_info_get_card(info) != 0 || snd_pcm_info_get_device(info) != 0 ||
        snd_pcm_info_get_subdevice(info) != 0 ||
        strcmp(snd_pcm_info_get_name(info), "ALC287 Analog") != 0) {
        result = -EINVAL; *reason = "pcm_identity_mismatch"; goto done;
    }
    CHECK(snd_pcm_hw_params_malloc(&hw), "hw_allocation");
    CHECK(snd_pcm_hw_params_any(stream->pcm, hw), "hw_capability");
    CHECK(snd_pcm_hw_params_set_access(stream->pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED), "interleaved_unavailable");
    CHECK(snd_pcm_hw_params_set_format(stream->pcm, hw, SND_PCM_FORMAT_S16_LE), "s16le_unavailable");
    CHECK(snd_pcm_hw_params_set_rate(stream->pcm, hw, 48000, 0), "exact_48k_unavailable");
    stream->channels = snd_pcm_hw_params_test_channels(stream->pcm, hw, 1) == 0 ? 1 : 2;
    CHECK(snd_pcm_hw_params_set_channels(stream->pcm, hw, stream->channels), "channel_layout_unavailable");
    stream->period = BLOCK;
    int period_direction = 0;
    CHECK(snd_pcm_hw_params_set_period_size_near(stream->pcm, hw, &stream->period, &period_direction), "period_unavailable");
    stream->buffer = BLOCK * 8;
    CHECK(snd_pcm_hw_params_set_buffer_size_near(stream->pcm, hw, &stream->buffer), "buffer_unavailable");
    CHECK(snd_pcm_hw_params(stream->pcm, hw), "hw_negotiation");
    unsigned actual_rate = 0, actual_channels = 0;
    snd_pcm_format_t actual_format;
    int rate_direction = 0;
    CHECK(snd_pcm_hw_params_get_rate(hw, &actual_rate, &rate_direction), "rate_readback");
    CHECK(snd_pcm_hw_params_get_channels(hw, &actual_channels), "channels_readback");
    CHECK(snd_pcm_hw_params_get_format(hw, &actual_format), "format_readback");
    CHECK(snd_pcm_hw_params_get_period_size(hw, &stream->period, &period_direction), "period_readback");
    CHECK(snd_pcm_hw_params_get_buffer_size(hw, &stream->buffer), "buffer_readback");
    if (actual_rate != 48000 || rate_direction != 0 || actual_channels != stream->channels ||
        actual_format != SND_PCM_FORMAT_S16_LE || stream->buffer < BLOCK * 4 ||
        stream->buffer > 48000 || !stream->period || stream->period > stream->buffer) {
        result = -EINVAL; *reason = "negotiated_layout_mismatch"; goto done;
    }
    CHECK(snd_pcm_sw_params_malloc(&sw), "sw_allocation");
    CHECK(snd_pcm_sw_params_current(stream->pcm, sw), "sw_current");
    CHECK(snd_pcm_sw_params_set_tstamp_mode(stream->pcm, sw, SND_PCM_TSTAMP_ENABLE), "timestamp_unavailable");
    CHECK(snd_pcm_sw_params_set_tstamp_type(stream->pcm, sw, SND_PCM_TSTAMP_TYPE_MONOTONIC), "monotonic_timestamp_unavailable");
    CHECK(snd_pcm_sw_params_set_avail_min(stream->pcm, sw, BLOCK), "availability_unavailable");
    snd_pcm_uframes_t boundary;
    CHECK(snd_pcm_sw_params_get_boundary(sw, &boundary), "start_boundary_unavailable");
    CHECK(snd_pcm_sw_params_set_start_threshold(stream->pcm, sw, boundary), "explicit_start_unavailable");
    CHECK(snd_pcm_sw_params_set_stop_threshold(stream->pcm, sw, stream->buffer), "xrun_stop_unavailable");
    CHECK(snd_pcm_sw_params(stream->pcm, sw), "sw_negotiation");
    CHECK(snd_pcm_sw_params_current(stream->pcm, sw), "effective_sw_current");
    CHECK(snd_pcm_sw_params_get_tstamp_mode(sw, &stream->sw_timestamp_mode), "effective_sw_timestamp_mode");
    CHECK(snd_pcm_sw_params_get_tstamp_type(sw, &stream->sw_timestamp_type), "effective_sw_timestamp_type");
    CHECK(snd_pcm_status_malloc(&stream->status), "status_allocation");
    CHECK(snd_pcm_prepare(stream->pcm), "initial_prepare");
    result = 0;
done:
    if (info) snd_pcm_info_free(info);
    if (hw) snd_pcm_hw_params_free(hw);
    if (sw) snd_pcm_sw_params_free(sw);
    return result;
#undef CHECK
}

static int observe(struct stream *stream, const char **reason) {
    snd_pcm_audio_tstamp_config_t requested = {
        .type_requested = SND_PCM_AUDIO_TSTAMP_TYPE_DEFAULT,
        .report_delay = 0,
    };
    snd_pcm_status_set_audio_htstamp_config(stream->status, &requested);
    int result = snd_pcm_status(stream->pcm, stream->status);
    if (result < 0) { *reason = "status_unavailable"; return result; }
    snd_htimestamp_t stamp;
    snd_pcm_status_get_htstamp(stream->status, &stamp);
    observation.role = stream->role;
    observation.state = snd_pcm_status_get_state(stream->status);
    observation.previous_timestamp = stream->timestamp;
    observation.timestamp_seconds = stamp.tv_sec;
    observation.timestamp_nanoseconds = stamp.tv_nsec;
    observation.avail = snd_pcm_status_get_avail(stream->status);
    observation.avail_max = snd_pcm_status_get_avail_max(stream->status);
    observation.delay = snd_pcm_status_get_delay(stream->status);
    observation.sw_timestamp_mode = stream->sw_timestamp_mode;
    observation.sw_timestamp_type = stream->sw_timestamp_type;
    observation.transferred = stream->transferred;
    if (snd_pcm_status_get_state(stream->status) != SND_PCM_STATE_RUNNING) {
        *reason = "pcm_not_running"; return -EPIPE;
    }
    uint64_t avail = snd_pcm_status_get_avail(stream->status);
    uint64_t maximum = snd_pcm_status_get_avail_max(stream->status);
    if (avail > stream->buffer || maximum > stream->buffer) {
        *reason = "upstream_availability_loss"; return -EPIPE;
    }
    if (maximum > stream->max_avail) stream->max_avail = maximum;
    if (stamp.tv_sec < 0 || stamp.tv_nsec < 0 || stamp.tv_nsec >= 1000000000L) {
        *reason = "timestamp_invalid"; return -EINVAL;
    }
    uint64_t timestamp = (uint64_t)stamp.tv_sec * UINT64_C(1000000000) + (uint64_t)stamp.tv_nsec;
    stream->current_timestamp = timestamp;
    result = validate_timestamp(stream->timestamp, timestamp, 1, reason);
    if (result < 0) return result;
    if (timestamp) stream->timestamp = timestamp;
    else stream->unknown_timestamps++;
    snd_pcm_audio_tstamp_report_t report;
    memset(&report, 0, sizeof(report));
    snd_pcm_status_get_audio_htstamp_report(stream->status, &report);
    stream->audio_timestamp_type = report.actual_type;
    stream->audio_timestamp_valid = report.valid;
    stream->audio_accuracy_reported = report.accuracy_report;
    stream->audio_accuracy_ns = report.accuracy;
    return 0;
}

static void close_stream(struct stream *stream) {
    if (stream->pcm) { snd_pcm_drop(stream->pcm); snd_pcm_close(stream->pcm); }
    if (stream->status) snd_pcm_status_free(stream->status);
}

#include <pipewire/pipewire.h>
#include <spa/interfaces/audio/aec.h>
#include <spa/support/plugin.h>
#include <spa/utils/names.h>
#include <spa/utils/json.h>
#include <speex/speex_resampler.h>
#include <openssl/sha.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <unistd.h>

#define RING 65536
#define MAX_COMMAND 1048656
#define SETTLING_BLOCKS 200
#define MEASUREMENT_BLOCKS 4500
#define MAX_FIXTURE_FRAMES 480000
#define NS_PER_SECOND UINT64_C(1000000000)

struct physical {
    struct stream capture, playback;
    snd_ctl_t *control;
    void *library;
    struct spa_handle *plugin;
    struct spa_audio_aec *aec;
    SpeexResamplerState *to_provider, *to_playback;
    uint64_t session, generation, adc, sequence, played, first_read_ns;
    uint64_t measure_start, reference_anchor;
    int measuring, capture_enabled, fixture_enabled;
    int16_t ledger[RING], playback_queue[RING];
    uint64_t queue_head, queue_tail;
    int16_t *fixture;
    size_t fixture_frames, fixture_cursor;
    unsigned char commands[MAX_COMMAND + 4];
    size_t command_bytes;
    int16_t positive[MAX_COMMAND / 2];
    size_t positive_frames, positive_cursor;
    char positive_sha[65], fingerprint[65];
    uint32_t capture_gains[2], playback_gains[2];
    unsigned capture_gain_count, playback_gain_count;
    unsigned playback_volume_percent;
    uint64_t last_control_ns;
    uint64_t stop_playback_request, stop_playback_target;
};

static volatile sig_atomic_t terminating;
static void term(int signal_number) { (void)signal_number; terminating = 1; }
static void digest_hex(const void *data, size_t size, char hex[65]) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(data, size, digest);
    for (unsigned i = 0; i < sizeof(digest); i++) snprintf(hex + i * 2, 3, "%02x", digest[i]);
    hex[64] = 0;
}
static int fatal(const char *reason) {
    char encoded[256];
    if (spa_json_encode_string(encoded, sizeof(encoded), reason) <= 0) return 69;
    printf("{\"kind\":\"fatal\",\"reason\":%s}\n", encoded);
    fflush(stdout);
    return 69;
}
static int control_snapshot(struct physical *p, int initial) {
    static const char *names[] = {
        "Internal Mic Phantom Jack", "Mic Jack", "Headphone Jack",
        "Capture Volume", "Digital Capture Volume", "Internal Mic Boost Volume",
        "Speaker Playback Volume", "Speaker Playback Switch", "Capture Switch"
    };
    snd_ctl_elem_list_t *list = NULL;
    snd_ctl_elem_info_t *info = NULL;
    snd_ctl_elem_value_t *value = NULL;
    snd_ctl_elem_id_t *id = NULL;
    unsigned char facts[4096];
    size_t size = 0;
    unsigned found = 0;
    int result = -EINVAL;
    if (snd_ctl_elem_list_malloc(&list) || snd_ctl_elem_info_malloc(&info) ||
        snd_ctl_elem_value_malloc(&value) || snd_ctl_elem_id_malloc(&id)) goto done;
    if (snd_ctl_elem_list(p->control, list)) goto done;
    unsigned count = snd_ctl_elem_list_get_count(list);
    if (!count || count > 512 || snd_ctl_elem_list_alloc_space(list, count) ||
        snd_ctl_elem_list(p->control, list)) goto done;
    for (unsigned i = 0; i < snd_ctl_elem_list_get_used(list); i++) {
        snd_ctl_elem_list_get_id(list, i, id);
        const char *name = snd_ctl_elem_id_get_name(id);
        unsigned selected = 0;
        while (selected < sizeof(names) / sizeof(names[0]) && strcmp(name, names[selected])) selected++;
        if (selected == sizeof(names) / sizeof(names[0])) continue;
        snd_ctl_elem_info_set_id(info, id);
        snd_ctl_elem_value_set_id(value, id);
        if (snd_ctl_elem_info(p->control, info) || snd_ctl_elem_read(p->control, value)) goto done;
        unsigned n = snd_ctl_elem_info_get_count(info);
        snd_ctl_elem_type_t type = snd_ctl_elem_info_get_type(info);
        if (!n || n > 16 || (type != SND_CTL_ELEM_TYPE_BOOLEAN && type != SND_CTL_ELEM_TYPE_INTEGER)) goto done;
        found |= 1u << selected;
        uint32_t numid = snd_ctl_elem_id_get_numid(id);
        int64_t minimum = type == SND_CTL_ELEM_TYPE_INTEGER ? snd_ctl_elem_info_get_min(info) : 0;
        int64_t maximum = type == SND_CTL_ELEM_TYPE_INTEGER ? snd_ctl_elem_info_get_max(info) : 1;
        if (maximum <= minimum || size + sizeof(numid) + sizeof(minimum) + sizeof(maximum) + n * sizeof(int64_t) > sizeof(facts)) goto done;
        memcpy(facts + size, &numid, sizeof(numid)); size += sizeof(numid);
        memcpy(facts + size, &minimum, sizeof(minimum)); size += sizeof(minimum);
        memcpy(facts + size, &maximum, sizeof(maximum)); size += sizeof(maximum);
        unsigned volume_total = 0;
        for (unsigned c = 0; c < n; c++) {
            int64_t actual = type == SND_CTL_ELEM_TYPE_BOOLEAN ?
                snd_ctl_elem_value_get_boolean(value, c) : snd_ctl_elem_value_get_integer(value, c);
            if (actual < minimum || actual > maximum) goto done;
            memcpy(facts + size, &actual, sizeof(actual)); size += sizeof(actual);
            if ((selected == 0 && !actual) || ((selected == 1 || selected == 2) && actual) ||
                ((selected == 7 || selected == 8) && !actual)) goto done;
            if (initial && c < 2 && selected == 3) {
                if (actual < 0 || actual > UINT32_MAX) goto done;
                p->capture_gains[c] = (uint32_t)actual; p->capture_gain_count = c + 1;
            }
            if (initial && c < 2 && selected == 6) {
                if (actual < 0 || actual > UINT32_MAX) goto done;
                p->playback_gains[c] = (uint32_t)actual; p->playback_gain_count = c + 1;
            }
            if (selected == 6) volume_total += (unsigned)((actual - minimum) * 100 / (maximum - minimum));
        }
        if (initial && selected == 6) p->playback_volume_percent = volume_total / n;
    }
    if (!(found & 1) || !(found & (1u << 1)) || !(found & (1u << 2)) || !(found & (1u << 3)) ||
        !(found & (1u << 6)) || !(found & (1u << 7)) || !(found & (1u << 8)) || !size) goto done;
    char digest[65]; digest_hex(facts, size, digest);
    if (initial) memcpy(p->fingerprint, digest, sizeof(digest));
    else if (strcmp(p->fingerprint, digest)) goto done;
    result = 0;
done:
    if (list) { snd_ctl_elem_list_free_space(list); snd_ctl_elem_list_free(list); }
    if (info) snd_ctl_elem_info_free(info);
    if (value) snd_ctl_elem_value_free(value);
    if (id) snd_ctl_elem_id_free(id);
    return result;
}

static int load_physical_aec(struct physical *p) {
    pw_init(NULL, NULL);
    p->library = dlopen(TRANSLATOR_AEC_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (!p->library) return -EINVAL;
    spa_handle_factory_enum_func_t enumerate = dlsym(p->library, SPA_HANDLE_FACTORY_ENUM_FUNC_NAME);
    if (!enumerate) return -EINVAL;
    const struct spa_handle_factory *factory = NULL;
    uint32_t index = 0;
    while (enumerate(&factory, &index) > 0)
        if (!strcmp(factory->name, SPA_NAME_AEC)) break;
    if (!factory || strcmp(factory->name, SPA_NAME_AEC)) return -EINVAL;
    size_t bytes = spa_handle_factory_get_size(factory, NULL);
    if (!bytes || bytes > 1024 * 1024) return -EINVAL;
    p->plugin = calloc(1, bytes);
    if (!p->plugin) return -ENOMEM;
    struct spa_support support[16];
    uint32_t count = pw_get_support(support, 16);
    if (spa_handle_factory_init(factory, p->plugin, NULL, support, count) < 0 ||
        spa_handle_get_interface(p->plugin, SPA_TYPE_INTERFACE_AUDIO_AEC, (void **)&p->aec) < 0) return -EINVAL;
    struct spa_audio_info_raw info = {
        .format = SPA_AUDIO_FORMAT_F32P, .rate = 48000, .channels = 1,
        .position = { SPA_AUDIO_CHANNEL_MONO }
    };
    const struct spa_dict_item properties[] = {
        { "webrtc.noise_suppression", "false" }, { "webrtc.high_pass_filter", "false" }
    };
    const struct spa_dict config = SPA_DICT_INIT(properties, 2);
    if (spa_audio_aec_init(p->aec, &config, &info) < 0 ||
        !p->aec->latency || strcmp(p->aec->latency, "480/48000")) return -EINVAL;
    int error = 0;
    p->to_provider = speex_resampler_init(1, 48000, 16000, SPEEX_RESAMPLER_QUALITY_DESKTOP, &error);
    if (!p->to_provider || error) return -EINVAL;
    p->to_playback = speex_resampler_init(1, 16000, 48000, SPEEX_RESAMPLER_QUALITY_DESKTOP, &error);
    return !p->to_playback || error ? -EINVAL : 0;
}
static int load_fixture(struct physical *p) {
    int fd = open("/stage/far.pcm", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return -errno;
    struct stat st;
    int result = -EINVAL;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size != 48000 * 2) goto done;
    p->fixture_frames = 48000;
    p->fixture = malloc((size_t)st.st_size);
    if (!p->fixture) { result = -ENOMEM; goto done; }
    size_t at = 0;
    while (at < (size_t)st.st_size) {
        ssize_t got = read(fd, (unsigned char *)p->fixture + at, (size_t)st.st_size - at);
        if (got <= 0) goto done;
        at += (size_t)got;
    }
    double power = 0;
    for (size_t i = 0; i < p->fixture_frames; i++) {
        if (p->fixture[i] == INT16_MIN || p->fixture[i] == INT16_MAX) goto done;
        power += (double)p->fixture[i] * p->fixture[i] / (32768.0 * 32768.0);
    }
    double dbfs = 10.0 * log10(power / p->fixture_frames);
    if (!isfinite(dbfs) || fabs(dbfs + 20.0) > 0.01) goto done;
    result = 0;
done:
    close(fd);
    return result;
}
static uint32_t le32(const unsigned char *b) {
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
static uint64_t le64(const unsigned char *b) { return le32(b) | (uint64_t)le32(b + 4) << 32; }
static void ack(uint64_t request, int success) {
    printf("{\"kind\":\"ack\",\"request\":%" PRIu64 ",\"success\":%s}\n", request, success ? "true" : "false");
}
static int process_command(struct physical *p, const unsigned char *body, uint32_t size) {
    if (size < 80 || size > MAX_COMMAND) return -EINVAL;
    uint32_t command = le32(body), pcm_bytes = le32(body + 12);
    uint64_t request = le64(body + 4);
    if (!request || pcm_bytes != size - 80 || pcm_bytes % 2) return -EINVAL;
    if (command != 3 && command != 7 && control_snapshot(p, 0)) return -EINVAL;
    const int16_t *pcm = (const int16_t *)(body + 80);
    int result = 0;
    switch (command) {
    case 1:
        if (pcm_bytes || p->measuring) return -EINVAL;
        p->measure_start = p->adc; p->measuring = 1; p->fixture_enabled = 1;
        break;
    case 2: {
        if (!pcm_bytes || pcm_bytes % 640 || p->positive_cursor < p->positive_frames ||
            pcm_bytes > sizeof(p->positive)) return -EINVAL;
        char hash[65]; digest_hex(pcm, pcm_bytes, hash);
        if (memcmp(hash, body + 16, 64)) return -EINVAL;
        memcpy(p->positive, pcm, pcm_bytes);
        memcpy(p->positive_sha, hash, 65);
        p->positive_frames = pcm_bytes / 2; p->positive_cursor = 0;
        p->capture_enabled = 1;
        break;
    }
    case 3: {
        if (!pcm_bytes || pcm_bytes % 640 || pcm_bytes / 2 > RING / 3) return -EINVAL;
        uint32_t in = pcm_bytes / 2, out = (uint32_t)(RING - (p->queue_head - p->queue_tail));
        int16_t converted[RING];
        if (in * 3 > out) return -ENOBUFS;
        if (speex_resampler_process_int(p->to_playback, 0, pcm, &in, converted, &out) ||
            in != pcm_bytes / 2 || out != in * 3) return -EINVAL;
        for (uint32_t i = 0; i < out; i++) p->playback_queue[p->queue_head++ % RING] = converted[i];
        p->fixture_enabled = 0;
        break;
    }
    case 4:
        if (pcm_bytes || p->stop_playback_request) return -EINVAL;
        p->fixture_enabled = 0; p->queue_tail = p->queue_head;
        p->stop_playback_request = request;
        p->stop_playback_target = p->playback.transferred;
        return 0; /* ACK only after the kernel reports this buffer consumed. */
    case 5:
        if (pcm_bytes || p->positive_cursor < p->positive_frames) return -EINVAL;
        p->capture_enabled = 0;
        break;
    case 6:
        if (pcm_bytes) return -EINVAL;
        p->capture_enabled = 1;
        break;
    case 7:
        if (pcm_bytes || p->positive_cursor < p->positive_frames) return -EINVAL;
        break;
    default: result = -EINVAL;
    }
    ack(request, result == 0);
    return result;
}
static int receive_commands(struct physical *p) {
    if (p->command_bytes == sizeof(p->commands)) return -ENOBUFS;
    ssize_t got = read(STDIN_FILENO, p->commands + p->command_bytes, sizeof(p->commands) - p->command_bytes);
    if (!got) return 1;
    if (got < 0 && errno != EAGAIN && errno != EINTR) return -errno;
    if (got > 0) p->command_bytes += (size_t)got;
    while (p->command_bytes >= 4) {
        uint32_t size = le32(p->commands);
        if (size < 80 || size > MAX_COMMAND) return -EINVAL;
        if (p->command_bytes < (size_t)size + 4) break;
        uint32_t command = le32(p->commands + 4);
        if (command != 3 && p->adc % 960) break;
        if (command == 7 && p->positive_cursor < p->positive_frames) break;
        int result = process_command(p, p->commands + 4, size);
        if (result) return result;
        memmove(p->commands, p->commands + 4 + size, p->command_bytes - 4 - size);
        p->command_bytes -= 4 + size;
    }
    return 0;
}
static void array_s16(const int16_t *samples, size_t count) {
    putchar('[');
    for (size_t i = 0; i < count; i++) printf("%s%d", i ? "," : "", samples[i]);
    putchar(']');
}
static void gains(const uint32_t *values, unsigned count) {
    putchar('[');
    for (unsigned i = 0; i < count; i++) printf("%s%u", i ? "," : "", values[i]);
    putchar(']');
}
static void ready(const struct physical *p) {
    printf("{\"kind\":\"ready\",\"identity\":{\"graph\":{\"kind\":\"native\",\"session_id\":%" PRIu64 ",\"generation\":%" PRIu64
           ",\"physical_device_id\":\"alsa-hw:0,0:PCH:ALC287 Analog\",\"dsp_config_id\":\"spa-webrtc-f32p48k-mono-ns0-hpf0-speex1.2.1\"},"
           "\"card\":0,\"card_id\":\"PCH\",\"pcm_name\":\"ALC287 Analog\",\"capture_channels\":%u,\"playback_channels\":%u,"
           "\"capture_buffer\":%lu,\"playback_buffer\":%lu,\"source_port\":\"analog-input-internal-mic\",\"sink_port\":\"analog-output-speaker\","
           "\"control_fingerprint\":\"%s\",\"capture_gains\":", p->session, p->generation, p->capture.channels, p->playback.channels,
           p->capture.buffer, p->playback.buffer, p->fingerprint);
    gains(p->capture_gains, p->capture_gain_count); printf(",\"playback_gains\":");
    gains(p->playback_gains, p->playback_gain_count);
    printf(",\"capture_muted\":false,\"playback_muted\":false,\"playback_volume_percent\":%u,\"capture_origin_monotonic_ns\":%" PRIu64 "}}\n", p->playback_volume_percent, p->first_read_ns);
}
static unsigned phase(const struct physical *p) {
    if (!p->measuring) return 0;
    uint64_t blocks = (p->adc - p->measure_start) / BLOCK;
    if (blocks < SETTLING_BLOCKS) return 0;
    blocks -= SETTLING_BLOCKS;
    if (blocks < 1500) return (unsigned)(blocks / 500) + 1;
    return blocks < MEASUREMENT_BLOCKS ? 4 : 5;
}
static int write_block(struct physical *p, const int16_t mono[BLOCK], const char **reason) {
    int16_t interleaved[BLOCK * 2];
    for (unsigned i = 0; i < BLOCK; i++)
        for (unsigned c = 0; c < p->playback.channels; c++) interleaved[i * p->playback.channels + c] = mono[i];
    snd_pcm_sframes_t written = snd_pcm_writei(p->playback.pcm, interleaved, BLOCK);
    const char *problem = validate_transfer(written);
    if (problem) { *reason = problem; return -EPIPE; }
    for (unsigned i = 0; i < BLOCK; i++) p->ledger[p->playback.transferred++ % RING] = mono[i];
    return 0;
}
static int frame(struct physical *p, const int16_t interleaved[BLOCK * 2], uint64_t stamp, uint64_t bracket, const char **reason) {
    int16_t raw[BLOCK], reference[BLOCK], clean[BLOCK], provider[160];
    float rec[BLOCK], play[BLOCK], output[BLOCK];
    if (p->played < BLOCK || p->playback.transferred - p->played > p->playback.buffer ||
        p->playback.transferred - (p->played - BLOCK) > RING) {
        *reason = "consumed_reference_unavailable"; return -EINVAL;
    }
    for (unsigned i = 0; i < BLOCK; i++) {
        int total = 0;
        for (unsigned c = 0; c < p->capture.channels; c++) total += interleaved[i * p->capture.channels + c];
        raw[i] = (int16_t)(total / (int)p->capture.channels);
        reference[i] = p->ledger[(p->played - BLOCK + i) % RING];
        rec[i] = raw[i] / 32768.0f; play[i] = reference[i] / 32768.0f;
    }
    const float *rec_channels[] = {rec}, *play_channels[] = {play};
    float *out_channels[] = {output};
    if (spa_audio_aec_run(p->aec, rec_channels, play_channels, out_channels, BLOCK) < 0) {
        *reason = "aec_process_failed"; return -EINVAL;
    }
    for (unsigned i = 0; i < BLOCK; i++) {
        if (!isfinite(output[i]) || output[i] <= -1 || output[i] >= 1) { *reason = "invalid_aec_pcm"; return -EINVAL; }
        clean[i] = (int16_t)lrintf(output[i] * 32768.0f);
    }
    uint32_t in = BLOCK, out = 160;
    if (speex_resampler_process_int(p->to_provider, 0, clean, &in, provider, &out) || in != BLOCK || out != 160) {
        *reason = "provider_transport_conversion_failed"; return -EINVAL;
    }
    unsigned scoring_phase = phase(p);
    printf("{\"kind\":\"frame\",\"session_id\":%" PRIu64 ",\"generation\":%" PRIu64 ",\"sequence\":%" PRIu64
           ",\"adc_start\":%" PRIu64 ",\"adc_end\":%" PRIu64 ",\"capture_monotonic_ns\":%" PRIu64
           ",\"bracket_ns\":%" PRIu64 ",\"written\":%" PRIu64 ",\"played\":%" PRIu64
           ",\"reference_start\":%" PRIu64 ",\"reference_end\":%" PRIu64 ",\"capture_timestamp\":",
           p->session, p->generation, p->sequence++, p->adc, p->adc + BLOCK, stamp, bracket, p->playback.transferred,
           p->played, p->played - BLOCK, p->played);
    if (p->capture.current_timestamp) printf("%" PRIu64, p->capture.current_timestamp); else printf("null");
    printf(",\"playback_timestamp\":");
    if (p->playback.current_timestamp) printf("%" PRIu64, p->playback.current_timestamp); else printf("null");
    printf(",\"phase\":%u,\"raw\":", scoring_phase); array_s16(raw, BLOCK);
    printf(",\"reference\":"); array_s16(reference, BLOCK);
    printf(",\"clean\":"); array_s16(clean, BLOCK);
    printf(",\"provider_pcm\":");
    int injected = p->positive_cursor < p->positive_frames;
    if (injected) {
        if (p->positive_frames - p->positive_cursor < 160) return -EINVAL;
        memcpy(provider, p->positive + p->positive_cursor, sizeof(provider));
        p->positive_cursor += 160;
    }
    array_s16(provider, p->capture_enabled ? 160 : 0);
    printf(",\"origin\":{\"kind\":\"%s\"", injected ? "injected_positive" : "physical");
    if (injected) printf(",\"fixture_sha256\":\"%s\"", p->positive_sha);
    printf("},\"channels\":[");
    for (unsigned c = 0; c < p->capture.channels; c++) {
        uint64_t squared = 0; int64_t sum = 0; unsigned clipped = 0, peak = 0;
        for (unsigned i = 0; i < BLOCK; i++) {
            int sample = interleaved[i * p->capture.channels + c];
            unsigned absolute = sample < 0 ? (unsigned)-sample : (unsigned)sample;
            if (absolute > peak) peak = absolute;
            clipped += sample == INT16_MIN || sample == INT16_MAX;
            sum += sample; squared += (uint64_t)((int64_t)sample * sample);
        }
        printf("%s{\"clipped\":%u,\"peak\":%u,\"sum\":%" PRId64 ",\"squared_sum\":%" PRIu64 "}", c ? "," : "", clipped, peak, sum, squared);
    }
    printf("]}\n");
    if (fflush(stdout) || ferror(stdout)) { *reason = "transport_failed"; return -EPIPE; }
    p->adc += BLOCK;
    return 0;
}
int translator_aec_physical_run(int argc, const char *const argv[]) {
    if (argc != 3 || strcmp(argv[1], "--physical") || strlen(argv[2]) != 16) return fatal("invalid_physical_invocation");
    for (unsigned i = 0; i < 16; i++)
        if (!((argv[2][i] >= '0' && argv[2][i] <= '9') || (argv[2][i] >= 'a' && argv[2][i] <= 'f')))
            return fatal("invalid_session");
    char *end = NULL; errno = 0;
    uint64_t session = strtoull(argv[2], &end, 16);
    if (errno || !end || *end || !session) return fatal("invalid_session");
    struct physical *p = calloc(1, sizeof(*p));
    if (!p) return fatal("allocation_failed");
    p->session = session; p->generation = monotonic_ns();
    p->capture.role = "capture"; p->playback.role = "playback";
    struct rlimit core = {0, 0};
    const char *reason = "physical_acquisition_failed";
    int result = 69;
    snd_ctl_card_info_t *card = NULL;
    if (setrlimit(RLIMIT_CORE, &core) || install_control_policy() ||
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK) < 0 ||
        snd_ctl_open(&p->control, "hw:0", SND_CTL_READONLY | SND_CTL_NONBLOCK) ||
        snd_ctl_card_info_malloc(&card) || snd_ctl_card_info(p->control, card) ||
        strcmp(snd_ctl_card_info_get_id(card), "PCH") ||
        strcmp(snd_ctl_card_info_get_driver(card), "HDA-Intel") ||
        control_snapshot(p, 1) || configure(&p->capture, SND_PCM_STREAM_CAPTURE, &reason) ||
        configure(&p->playback, SND_PCM_STREAM_PLAYBACK, &reason) ||
        load_fixture(p) || load_physical_aec(p)) goto done;
    signal(SIGTERM, term); signal(SIGINT, term);
    int16_t silence[BLOCK] = {0}, interleaved[BLOCK * 2] = {0};
    for (unsigned i = 0; i < 4; i++) if (write_block(p, silence, &reason)) goto done;
    if (snd_pcm_start(p->playback.pcm) || snd_pcm_start(p->capture.pcm)) goto done;
    uint64_t progress = monotonic_ns();
    p->last_control_ns = progress;
    while (!terminating) {
        int command = receive_commands(p);
        if (command == 1) { result = 0; break; }
        if (command < 0) { reason = "command_contract_failed"; break; }
        uint64_t before = monotonic_ns();
        if (!before || before - progress > UINT64_C(200000000)) { reason = "capture_progress_deadline"; break; }
        if (before - p->last_control_ns >= UINT64_C(100000000)) {
            if (control_snapshot(p, 0)) { reason = "physical_configuration_changed"; break; }
            p->last_control_ns = before;
        }
        if (observe(&p->capture, &reason) || observe(&p->playback, &reason)) break;
        uint64_t played = 0;
        if (validate_cursor(p->playback.transferred, snd_pcm_status_get_delay(p->playback.status),
                            p->playback.buffer, p->played, 1, BLOCK, &played)) { reason = "played_cursor_invalid"; break; }
        p->played = played;
        if (p->stop_playback_request && p->played >= p->stop_playback_target) {
            ack(p->stop_playback_request, 1);
            p->stop_playback_request = 0;
        }
        if (snd_pcm_status_get_avail(p->capture.status) < BLOCK || p->played < BLOCK) {
            struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN};
            if (poll(&input, 1, 1) < 0 && errno != EINTR) { reason = "poll_failed"; break; }
            continue;
        }
        snd_pcm_sframes_t got = snd_pcm_readi(p->capture.pcm, interleaved, BLOCK);
        if (validate_transfer(got)) { reason = "short_capture_transfer"; break; }
        uint64_t after = monotonic_ns();
        p->capture.transferred += BLOCK;
        if (!p->first_read_ns) { p->first_read_ns = after; ready(p); }
        if (frame(p, interleaved, after, after - before, &reason)) break;
        progress = after;
        if (snd_pcm_status_get_avail(p->playback.status) < BLOCK) continue;
        int16_t playback[BLOCK] = {0};
        unsigned current_phase = phase(p);
        for (unsigned i = 0; i < BLOCK; i++) {
            if (p->queue_tail < p->queue_head) playback[i] = p->playback_queue[p->queue_tail++ % RING];
            else if (p->fixture_enabled && current_phase >= 4)
                playback[i] = p->fixture[p->fixture_cursor++ % p->fixture_frames];
        }
        if (write_block(p, playback, &reason)) break;
    }
    if (terminating) result = 0;
done:
    memset(p->commands, 0, sizeof(p->commands)); memset(p->positive, 0, sizeof(p->positive));
    close_stream(&p->capture); close_stream(&p->playback);
    if (p->to_provider) speex_resampler_destroy(p->to_provider);
    if (p->to_playback) speex_resampler_destroy(p->to_playback);
    if (p->plugin) { spa_handle_clear(p->plugin); free(p->plugin); }
    if (p->library) dlclose(p->library);
    if (p->control) snd_ctl_close(p->control);
    if (card) snd_ctl_card_info_free(card);
    free(p->fixture); free(p);
    return result ? fatal(reason) : 0;
}
