# Service CPU profiling

Two tools find where CPU goes: a passive survey of every service's process CPU, and an opt-in sampler inside a
Python service. Start with the survey.

## Survey every service

```sh
nice -n 19 .venv/bin/python tools/profile_services.py --seconds 60 --output <private-json-path>
```

It reads process counters only: no restarts and no broker, catalog, or media reads. It counts each service's
descendants, Rust daemons included. One core means one CPU-second per elapsed second.

## Profile one Python service

Set `PACK_PROFILE` in the environment of a `pack-*` service and restart it. The sampler waits out a warmup, samples
for a fixed window, and stops by itself. It is off by default, and the off path loads nothing. It needs Linux
procfs; elsewhere the service starts without profiling and logs a warning.

| Variable | Default | Meaning |
|---|---|---|
| `PACK_PROFILE` | unset | `1` enables this service; a comma-separated list selects services by name |
| `PACK_PROFILE_SERVICE` | From `pack-<name>` | The service name, for a module entry point |
| `PACK_PROFILE_DELAY_SECONDS` | 30 | Warmup before the window |
| `PACK_PROFILE_SECONDS` | 120 | The measured window, 1 to 3,600 |
| `PACK_PROFILE_INTERVAL_SECONDS` | 0.05 | Sampling interval |
| `PACK_PROFILE_FLUSH_SECONDS` | 30 | How often the snapshot is rewritten |

Profile one or two services at a time, in a quiet window, and remove the variable afterward. Do not change ptrace
or perf policy to get a capture.

Snapshots go to `paths.logs/profiles/` as JSON and collapsed stacks, mode 0600, a bounded number per service. They
hold code locations and counters only, never source, arguments, or observed data. Keep them local and publish only
conclusions. Check the JSON's `final` flag: a killed service leaves its last periodic snapshot.

To report on finished captures:

```sh
.venv/bin/python -m tools.profile_services --profiles <profile-directory> --survey <survey-json> --output <report-json>
```

It prints each service's cores and its top ten functions by estimated self CPU.

## Reading the numbers

- **Function costs are statistical.** The sampler weights each thread's CPU ticks onto the stack it sees at the
  sample. That stack may not be the one that used the CPU. Short functions and short-lived threads are
  under-sampled.
- **Only running threads are attributed.** CPU from a thread that is not runnable at the sample goes to
  `nonrunning_unattributed_ticks`, never to a waiting function.
- **Native threads without a Python frame are unattributed.** Report that CPU as unattributed; do not guess.
- **Inclusive totals are not additive.** Never present them as self CPU, or a sample count as CPU time.
- **Check the residual.** A large gap between process CPU and observed thread CPU means attribution is incomplete.
- **The sampler costs CPU too** (`excluded_sampler_ticks`). Raise the interval to lower it.

A service that wakes many times a second and sleeps again leaves most of its CPU unattributed. The `wait_sites`
table covers that case: for each blocking call site, how often threads woke there. Read it as "this site wakes N
times a second", then look for a longer timeout, a blocking wait, or batching. It is wait evidence, not CPU
attribution.

Record the interval, window, and warmup with every result. Treat repeated decodes, conversions, serialization, and
polling as hypotheses until a profile shows their cost.
