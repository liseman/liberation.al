/* continuum.h in plain C: the ABI version matches, a call with a null argument fails with a message, and calls
 * made during process exit (an atexit handler, after the thread's storage is gone) return quietly instead of
 * aborting. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "continuum.h"

/* Runs after main returns: the library's per-thread error storage may already be destroyed. */
static void at_exit(void) {
    cnt_client *client = NULL;
    if (cnt_connect(NULL, NULL, &client) != CNT_INVALID_ARGUMENT) _Exit(3);
    (void)cnt_last_error_message();
    (void)cnt_last_broker_code();
    if (cnt_client_close(NULL, "exit") != CNT_INVALID_ARGUMENT) _Exit(3);
}

int main(void) {
    if (atexit(at_exit) != 0) return 1;
    if (cnt_abi_version() != CNT_ABI_VERSION) {
        fprintf(stderr, "ABI %u, header %u\n", cnt_abi_version(), (unsigned)CNT_ABI_VERSION);
        return 1;
    }
    cnt_client *client = NULL;
    cnt_status s = cnt_connect(NULL, NULL, &client);
    if (s != CNT_INVALID_ARGUMENT || client != NULL || strlen(cnt_last_error_message()) == 0) {
        fprintf(stderr, "a null socket should be CNT_INVALID_ARGUMENT with a message, got %d\n", (int)s);
        return 1;
    }
    printf("c_check: libcontinuum %s, ABI %u\n", cnt_version(), cnt_abi_version());
    return 0;
}
