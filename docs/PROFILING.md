# Service CPU profiling

`PACK_PROFILE` opts a Python pack console service into a finite in-process CPU profile (#1743). It is off by
default. The package's startup hook checks the environment; the off path imports no profiler or configuration,
starts no thread, and writes nothing. The service retains its existing command, PID, arguments, and signal handling.

Enable it only on the services selected for a coordinated quiet window. On systemd, the operator can add temporary
per-unit environment overrides and restart one or two services at a time, then remove those overrides after the
capture. No profiler command installs, restarts, or changes a service. Ordinary startup on other platforms remains
unchanged; requesting CPU stack profiling without Linux procfs logs an actionable warning and starts the service
without profiling. Do not change ptrace or perf policy to obtain a capture.

| Environment override | Default | Meaning |
| --- | --- | --- |
| `PACK_PROFILE` | unset | `1` enables this service; a comma-separated service list selects matching pack console names |
| `PACK_PROFILE_SERVICE` | inferred from `pack-<name>` | Explicit generic service name for a Python module entry point |
| `PACK_PROFILE_DELAY_SECONDS` | 30 | Warmup before the measured window, from 0 to 600 seconds |
| `PACK_PROFILE_SECONDS` | 120 | Finite measured window, from 1 to 3,600 seconds; collection stops automatically |
| `PACK_PROFILE_INTERVAL_SECONDS` | 0.05 | Stack/counter sampling interval, from 0.01 to 1 second |
| `PACK_PROFILE_FLUSH_SECONDS` | 30 | Periodic cumulative artifact publication, from 1 to 300 seconds |

Administration commands `pack-services` and `pack-config` are excluded. A restarted service gets a new capture;
the sampler stops itself after its finite window, so turning sampling off requires no further service restart.
Remove the diagnostic override to keep subsequent restarts unprofiled. Invalid overrides warn without blocking
the underlying service.

Artifacts live under configured `paths.logs/profiles/`. JSON snapshots and collapsed stacks are atomic mode-0600
files, rotated to a bounded number per service. They contain code locations, CPU counters, and capture/process
identities, never source text, locals, arguments, transcripts, frames, or observed runtime records. Keep them local;
publish only aggregate profiling conclusions. A normal exit flushes through `atexit`; abrupt termination can leave
the latest periodic snapshot incomplete. Check the JSON `final` flag before treating a capture as complete.
`SIGTERM`, `os._exit`, and crashes do not guarantee `atexit` execution. A forked Python child does not inherit a
running sampler thread; these artifacts cover their identified process only. The passive survey still accounts
for descendants, whose CPU must not be silently attributed to the parent's Python functions.

## What the numbers mean

Linux `/proc/self/task/<tid>/stat` supplies each native thread's user-plus-system CPU ticks. Python thread IDs map
`sys._current_frames()` stacks to native IDs. Each positive CPU delta is weighted onto that thread's sampled stack only when its sampled Linux state is `R`
(running or runnable, not proof of execution on a CPU). For each Python thread, its counter/state read is immediately
followed by a fresh frame lookup before the next task's I/O (`thread-cpu-delta-running-state-v3`). These reads are
adjacent, not atomic. CPU from a visible Python thread sampled outside `R`
remains in `nonrunning_unattributed_ticks`, not on a waiting function. It is never redistributed to running stacks.
Sleeping threads with no CPU delta contribute zero CPU. Native calls with a visible Python caller contribute to that sampled caller;
native worker threads without a Python frame remain explicitly unattributed.

OS-counter deltas are CPU accounting at the kernel's clock-tick resolution. Per-function self and inclusive values
are **statistical attribution**, not exact invocation timings. A stack seen at a tick can differ from the stack
that consumed CPU earlier in that interval. The running-state filter reduces idle-endpoint attribution, but does not
recover prior short-lived work or remove GIL/scheduling bias. Earlier captures without the filter remain labelled
`all-endpoint-states`; reports keep their method separate from `running-only` captures. Python scheduling/GIL contention can delay collection. Short functions,
short-lived threads, and native worker pools need care. Never present inclusive totals as additive self CPU, or a
wall-time sample count as CPU time. Record sampling interval, elapsed window, and warmup with every result.

The JSON separates sampler CPU, Python-attributed CPU, native/unmapped CPU, non-running endpoint CPU, overflow, new/exited/unreadable thread
gaps, and the signed process-minus-observed-thread residual. Process and thread counter reads are not atomic, so
small residuals can reflect read skew; a large residual means attribution is incomplete. Report it rather than
normalizing away missing CPU. The collapsed stack weights are CPU clock ticks, not raw sample counts. Python stacks carry a native-thread-ID
and start-tick root so identical functions on different threads remain distinguishable; global function rankings
still aggregate across threads. Older artifacts without `stack_scope: "native-thread"` lack that dimension.

`tools/profile_services.py --seconds 60 --output <private-json-path>` passively measures service process CPU,
including descendants, with no restarts or broker/catalog/media reads. Run it at low priority with the existing
venv interpreter. One core means one CPU-second per elapsed second; 100% here means one core, not the whole machine.
PID/start-time identities prevent PID reuse from combining unrelated processes. New and exited processes carry
partial-window evidence. Do not add process totals to their constituent thread totals or include child CPU counters
again on the parent.

To combine completed profiles with an existing passive survey without reading live processes, use
`python -m tools.profile_services --profiles <profile-directory> --survey <survey-json> --output <report-json>`.
With `--profiles` alone it summarizes profiles only; adding `--seconds` explicitly requests a new passive survey.
The tool selects the newest completed capture per service/PID/process-start identity and emits a Markdown table with
separate survey/profile cores and the top ten estimated self/inclusive Python function costs. Their independent
measurement windows remain separate. Multiple process captures receive separate function sections and rates;
independent process windows are not summed into a service total. Completed process captures can outlive their
process, so use a contemporaneous survey for current service membership. Only artifacts still in the bounded
rotation can be reported; absent process captures remain unmeasured. Invalid artifacts and absent completed
captures are actionable errors.

## Where short wakeups wait

Services that wake many times a second and sleep again before the next sample (listen, see-captions, and
see-zones in the #3162 captures) leave most of their CPU in `nonrunning_unattributed_ticks`, so the function table
cannot say where that polling or per-message overhead lives. A separate wait-site table answers it (#3180,
`wait_method: "voluntary-switch-delta-at-nonrunning-endpoint-v1"`). Each sample also reads every thread's
`voluntary_ctxt_switches` from `/proc/self/task/<tid>/status`. A thread blocks once per voluntary switch, so the
delta since its previous sample counts its wakeups. When a Python thread's endpoint is not running, those wakeups
and its non-running CPU ticks go to that endpoint's stack in `wait_sites`; the leaf frame carries the current line,
which is the blocking call (a `time.sleep`, a `Queue.get` inside `threading.py`, a `select`, a socket read).
Wakeups at running endpoints and on threads without a Python frame are totalled as `running_endpoint_wakeups`
and `native_wakeups`, and each thread row gains `voluntary_switches`. The sampler's own switches are excluded.

This table is wait evidence, not CPU attribution. Its ticks are never added to the function estimates or the
collapsed stacks, and `nonrunning_unattributed_ticks` keeps the same total. Like the CPU stacks, it is a statistical
endpoint: a thread that waits at several sites between two samples has all of that interval's wakeups recorded at
the site it was sampled at. A loop that blocks at one site (the common polling shape) keeps all of its wakeups there.
Waiting for the GIL is also a voluntary switch, so contended threads add some wakeups of their own. A thread
polling with a 10 ms sleep shows at its sleep line with about 100 wakeups per second. Read it as "this site wakes N
times a second and costs M non-running cores", then look for a longer timeout, a blocking wait, or batching.
`tools/profile_services.py --profiles` prints the top ten sites by wakeups per second, innermost frame first, and
the top ten threads by wakeups per second. Captures from before #3180 report the table as unavailable.

The sampler's own CPU is `excluded_sampler_ticks`. The #3162 captures measured about 0.055 cores per profiled
service at the default 0.05 s interval, before wait sampling added a second procfs read per thread per sample; check
that field when a capture's budget matters, and raise `PACK_PROFILE_INTERVAL_SECONDS` to lower it.

## Interpreting all-service results

Use simultaneous process counters for overall service ranking and finite profiles for attribution. Capture the
hottest services first, then the remaining Python services. Wait for startup/model loading to settle and label any
window with a workload change. For each service report cores, the top ten functions by estimated self CPU (with
inclusive CPU alongside), native/unattributed CPU, sampler cost, and concrete redundant work supported by stacks.
Repeated decoder calls, color conversion, serialization, and polling are hypotheses until actual profiles show
their cost. Coordinate findings with work already changing those paths.

Rust services still receive exact process CPU accounting from the passive survey. Their startup/purge phase logs
measure wall durations and workload/backlog counters, not steady-state per-function CPU. If native profiling is
unavailable under the machine's existing perf policy, mark native function attribution unavailable. Do not fill the
top-ten list with guessed functions or turn wall timings into CPU percentages.
