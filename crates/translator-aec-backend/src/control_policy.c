#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sound/asound.h>

/* Public libseccomp 2.5.5 ABI; development headers are not installed. */
struct scmp_arg_cmp {
    unsigned int arg;
    int op;
    uint64_t datum_a;
    uint64_t datum_b;
};

static const struct control_operation {
    const char *name;
    unsigned long request;
    int mutates;
} operations[] = {
    {"ELEM_WRITE", SNDRV_CTL_IOCTL_ELEM_WRITE, 1},
    {"ELEM_LOCK", SNDRV_CTL_IOCTL_ELEM_LOCK, 1},
    {"ELEM_UNLOCK", SNDRV_CTL_IOCTL_ELEM_UNLOCK, 1},
    {"SUBSCRIBE_EVENTS", SNDRV_CTL_IOCTL_SUBSCRIBE_EVENTS, 1},
    {"ELEM_ADD", SNDRV_CTL_IOCTL_ELEM_ADD, 1},
    {"ELEM_REPLACE", SNDRV_CTL_IOCTL_ELEM_REPLACE, 1},
    {"ELEM_REMOVE", SNDRV_CTL_IOCTL_ELEM_REMOVE, 1},
    {"TLV_WRITE", SNDRV_CTL_IOCTL_TLV_WRITE, 1},
    {"TLV_COMMAND", SNDRV_CTL_IOCTL_TLV_COMMAND, 1},
    {"POWER", SNDRV_CTL_IOCTL_POWER, 1},
    {"RAWMIDI_PREFER_SUBDEVICE", SNDRV_CTL_IOCTL_RAWMIDI_PREFER_SUBDEVICE, 1},
    {"CARD_INFO", SNDRV_CTL_IOCTL_CARD_INFO, 0},
    {"PVERSION", SNDRV_CTL_IOCTL_PVERSION, 0},
    {"PCM_PREFER_SUBDEVICE", SNDRV_CTL_IOCTL_PCM_PREFER_SUBDEVICE, 0},
};

int install_control_policy(void) {
    void *library = dlopen("/lib/x86_64-linux-gnu/libseccomp.so.2", RTLD_NOW | RTLD_LOCAL);
    if (!library) return -ENOSYS;
    void *(*initialize)(uint32_t) = dlsym(library, "seccomp_init");
    void (*release)(void *) = dlsym(library, "seccomp_release");
    int (*resolve)(const char *) = dlsym(library, "seccomp_syscall_resolve_name");
    int (*add)(void *, uint32_t, int, unsigned int, const struct scmp_arg_cmp *) = dlsym(library, "seccomp_rule_add_array");
    int (*load)(void *) = dlsym(library, "seccomp_load");
    if (!initialize || !release || !resolve || !add || !load) {
        dlclose(library);
        return -ENOSYS;
    }
    void *context = initialize(UINT32_C(0x7fff0000));
    if (!context) { dlclose(library); return -ENOMEM; }
    int syscall_number = resolve("ioctl");
    int result = syscall_number < 0 ? -ENOSYS : 0;
    for (unsigned index = 0; !result && index < sizeof(operations) / sizeof(operations[0]); index++) {
        if (!operations[index].mutates) continue;
        /* Kernel ioctl command is 32-bit: high-bit aliases must not bypass. */
        struct scmp_arg_cmp comparison = {
            .arg = 1, .op = 7, .datum_a = UINT32_MAX,
            .datum_b = (uint32_t)operations[index].request,
        };
        result = add(context, UINT32_C(0x00050000) | EPERM, syscall_number, 1, &comparison);
    }
    if (!result) result = load(context);
    release(context);
    dlclose(library);
    return result;
}

int test_control_policy(const char *name, int high_alias) {
    const struct control_operation *operation = NULL;
    for (unsigned index = 0; index < sizeof(operations) / sizeof(operations[0]); index++)
        if (strcmp(operations[index].name, name) == 0) operation = &operations[index];
    if (!operation) return 69;
    int result = install_control_policy();
    if (result) {
        printf("{\"status\":\"UNAVAILABLE\",\"reason\":\"control_policy_unavailable\",\"error\":%d,\"aec_proof\":false}\n", result);
        return 69;
    }
    unsigned long request = operation->request;
    if (high_alias) request |= UINT64_C(1) << 32;
    errno = 0;
    int actual = ioctl(-1, request, NULL);
    int observed_errno = errno;
    int expected_errno = operation->mutates ? EPERM : EBADF;
    int verified = actual == -1 && observed_errno == expected_errno;
    printf("{\"status\":\"%s\",\"errno\":%d,\"hardware_opened\":false,\"aec_proof\":false}\n",
           verified ? "CONTROL_POLICY_VERIFIED" : "UNAVAILABLE", observed_errno);
    return verified ? 0 : 69;
}
