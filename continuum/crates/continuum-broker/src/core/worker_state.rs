//! Worker checkpoints as one keyed catalog row: saves and deletes go through the writer, which checks the
//! inputs' revisions in the same transaction that stores the row; loads are read on the reader pool.

use continuum_core::schema::continuum::ErrorCode;

use super::{Event, ReadJob, Sched, SessionId, Urgent, error_code};
use crate::protocol::{self, worker_state::Ask, worker_state::Done};

impl Sched {
    /// A `SaveWorkerState`, `LoadWorkerState`, or `ForgetWorkerState`.
    pub(super) fn worker_state(&mut self, session: SessionId, request_id: u64, ask: Ask) {
        if let Ask::Load { owner, contract } = ask {
            let events = self.events.clone();
            let job: ReadJob = Box::new(move |catalog| {
                let result = catalog
                    .load_worker_state(&owner, &contract)
                    .map(Done::Loaded)
                    .map_err(|e| (error_code(&e), e.to_string()));
                let _ = events.send(Event::WorkerStateDone { session, request_id, result });
            });
            if let Err(why) = self.read(job) {
                self.reply_error(session, request_id, ErrorCode::Unavailable, why);
            }
            return;
        }
        if let Err(e) = self.urgent.try_send(Urgent::WorkerState { session, request_id, ask }) {
            self.refuse_busy(session, request_id, &e);
        }
    }

    /// What the catalog said, to the session that asked.
    pub(super) fn worker_state_done(
        &mut self,
        session: SessionId,
        request_id: u64,
        result: Result<Done, (ErrorCode, String)>,
    ) {
        match result {
            Ok(done) => {
                let bytes = protocol::worker_state::reply(&mut self.builder, request_id, &done);
                self.push(session, bytes);
            }
            Err((code, msg)) => self.reply_error(session, request_id, code, &msg),
        }
    }
}

/// The writer's half: a save or a delete, in one catalog transaction.
pub(super) fn write(catalog: &dyn continuum_catalog::Catalog, ask: Ask) -> Result<Done, (ErrorCode, String)> {
    let result = match ask {
        Ask::Save(state) => catalog.save_worker_state(&state).map(|()| Done::Saved),
        Ask::Forget { owner, contract } => catalog.forget_worker_state(&owner, &contract).map(Done::Forgotten),
        Ask::Load { owner, contract } => catalog.load_worker_state(&owner, &contract).map(Done::Loaded),
    };
    result.map_err(|e| (error_code(&e), e.to_string()))
}
