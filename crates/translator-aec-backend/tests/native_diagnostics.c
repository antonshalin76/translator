#include "../src/native.c"
#include <assert.h>

int main(void) {
    struct backend b = { 0 };
    struct spa_chunk chunk = { .size = 123, .stride = 4, .flags = 0 };
    struct spa_data data = { .chunk = &chunk };
    struct spa_buffer spa = { .n_datas = 1, .datas = &data };
    struct pw_buffer pw = { .buffer = &spa };

    atomic_store(&b.callback_count, 17);
    atomic_store(&b.last_position_tick, 123456);
    poison_at(&b, FAIL_BUFFER, 101);
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
    return 0;
}
