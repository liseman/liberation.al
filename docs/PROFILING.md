# How to find where CPU goes

Two tools find where CPU goes: a passive survey of every service's process CPU
([tools/profile_services.py](../tools/profile_services.py)), and an opt-in sampler inside a Python service
(`src/liberational/service_profile.py`). Start with the survey.

## Before you begin

- Both live tools read Linux `/proc`. On non-Linux systems, the live survey exits with an error, and a service with
  `PACK_PROFILE` set logs a warning and starts without profiling. Offline `--profiles` reporting works anywhere.
- Do not change `ptrace` or `perf` policy to capture a profile.
- Keep profile snapshots and JSON reports under `data/` or another local private path, and publish only conclusions.

## Survey every service

Run the passive survey at lowest priority:

```sh
nice -n 19 .venv/bin/python -m tools.profile_services --seconds 60 --output <private-json-path>
```

It reads `/proc` process counters only: no restarts and no broker, catalog, or media reads. It counts each service's
descendants, Rust daemons included. One core means one CPU-second per elapsed second. Processes born or exiting between
polls have incomplete coverage, and short-lived processes can be missed.

## Profile one Python service

1. Set `PACK_PROFILE=<service>` in the environment of one or two `pack-*` services during a quiet window, then restart
   those services. `pack-services` and `pack-config` never start the sampler.
2. Wait for the warmup and measurement window to finish (`PACK_PROFILE_DELAY_SECONDS` plus `PACK_PROFILE_SECONDS`), or
   stop the service cleanly with `SIGTERM`. The sampler stops on its own and writes `.json` and `.collapsed` stack
   files (mode `0600`, at most three snapshots per service) under `paths.logs/profiles/`.
3. Remove `PACK_PROFILE` from the service environment and restart the service so future runs take the zero-overhead off
   path.
4. Summarize the finished captures alongside your saved survey:

   ```sh
   .venv/bin/python -m tools.profile_services --profiles <profile-dir> --survey <survey-json> --output <report-json>
   ```

`--profiles` reads only snapshots with `"final": true`; a hard kill leaves only periodic `"final": false` snapshots. It
prints each service's cores, its top ten Python locations by estimated self CPU, its top ten wait sites, and its top
ten native threads by wakeups per second. Omit `--survey` and pass `--seconds 60` to run a fresh survey in the same
pass (`--survey` and `--seconds` cannot be combined).

## Reference

### Environment variables

Out-of-range numeric values log a warning and skip profiling (`src/liberational/profile_bootstrap.py`).

| Variable | Default | Range | Meaning |
|---|---|---|---|
| `PACK_PROFILE` | unset | `1`, `true`, `yes`, `on`, or comma-separated names | Enables profiling on every Python service, or only the named services (with or without `pack-`) |
| `PACK_PROFILE_SERVICE` | From `pack-<name>` | String | Overrides the service name for a `python -m` entry point |
| `PACK_PROFILE_DELAY_SECONDS` | `30` | `0` to `600` | Warmup before the measured window |
| `PACK_PROFILE_SECONDS` | `120` | `1` to `3600` | Length of the measured window |
| `PACK_PROFILE_INTERVAL_SECONDS` | `0.05` | `0.01` to `1.0` | Stack sampling interval in seconds |
| `PACK_PROFILE_FLUSH_SECONDS` | `30` | `1` to `300` | How often periodic snapshots are rewritten |

Snapshots hold code locations and counters only, never source text, arguments, or observed data.

### Reading the numbers

- **Function costs are statistical.** The sampler weights each thread's CPU ticks onto the stack it sees at the sample.
  That stack may not be the one that used the CPU, and native C or Rust work called from Python can be charged to its
  visible Python caller. Short functions and short-lived threads are under-sampled.
- **Survey and profile windows are independent.** Never add survey cores to profile cores, and never sum per-PID rates
  captured over different windows.
- **Only running threads are attributed.** CPU from a thread that is not runnable at the sample goes to
  `nonrunning_unattributed_ticks`, never to a waiting function.
- **Native threads without a Python frame are unattributed.** Report that CPU as unattributed; do not guess.
- **Inclusive totals overlap.** Never present inclusive cores as self CPU, or a sample count as CPU time.
- **Check the residual.** A large gap between process CPU and observed thread CPU means attribution is incomplete.
- **The sampler costs CPU too** (`excluded_sampler_ticks`). Raise `PACK_PROFILE_INTERVAL_SECONDS` to lower it.
- **Wait sites are wakeup evidence, not CPU attribution.** A service that wakes many times a second and sleeps again
  leaves most of its CPU unattributed; `wait_sites` records how often threads woke at each blocking call site so you
  can look for a longer timeout, a blocking wait, or batching.

Record the interval, window, and warmup with every result. Treat repeated decodes, conversions, serialization, and
polling as hypotheses until a profile shows their cost.
