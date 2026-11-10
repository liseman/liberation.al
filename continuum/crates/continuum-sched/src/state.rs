//! Broker system state the scheduler acts on (docs/CONTINUUM.md, decision 11).

use continuum_core::VoiceTurnState;

/// A system state change. The broker broadcasts it to subscribers; the scheduler reacts to the ones that change
/// what may run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SystemState {
    /// The voice turn. `Listening` and `Speaking` preempt P2 and P3 leases on the GPU and Neural Engine at once.
    VoiceTurn(VoiceTurnState),
}
