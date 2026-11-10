//! One capture worker (see `continuum_capture::worker`): started by the capture supervisor, one per device.

use std::process::ExitCode;

use continuum_capture::worker::{STOP, WorkerEnd, install_stop_signals, parse_args, run};

fn main() -> ExitCode {
    continuum_log::init("capture-worker");
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("{e}");
            eprintln!("usage: continuum-capture-worker --manifest P --device NAME --node NODE --socket S");
            return ExitCode::from(WorkerEnd::BadSetup as u8);
        }
    };
    install_stop_signals();
    ExitCode::from(run(&args, &STOP) as u8)
}
