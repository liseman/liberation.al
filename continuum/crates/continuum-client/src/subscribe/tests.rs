use super::*;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Duration;

fn shared() -> Shared {
    let (writes, _outgoing) = mpsc::unbounded_channel();
    Shared {
        writes,
        pending: Mutex::new(HashMap::new()),
        next_id: AtomicU64::new(1),
        subscriptions: Mutex::new(HashMap::new()),
        next_subscription: AtomicU64::new(1),
        subscription_overflow: SubscriptionOverflow::Close,
        purges: Mutex::new(HashMap::new()),
        quotes: Mutex::new(HashMap::new()),
        promises: Mutex::new(HashMap::new()),
        next_promise_token: AtomicU64::new(1),
        timeout: Duration::from_secs(1),
        closed: AtomicBool::new(false),
        voice: Mutex::new(fb::VoiceTurnState::Idle),
        runtime: tokio::runtime::Handle::current(),
    }
}

fn full_route() -> (Route, mpsc::Receiver<SubEvent>) {
    let (tx, events) = mpsc::channel(1);
    tx.try_send(SubEvent::CaughtUp { position: 1 }).unwrap();
    let (ended, _) = tokio::sync::watch::channel(false);
    (Route { tx, ended: Arc::new(ended) }, events)
}

#[tokio::test]
async fn unsubscribe_after_route_clone_does_not_close_the_connection_for_stale_overflow() {
    let shared = shared();
    let (route, _events) = full_route();
    shared.subscriptions.lock().unwrap().insert(7, route);
    // Reconstruct route()'s exact interleaving without a timing-dependent thread race: it cloned the
    // sender, then another task unsubscribed before the full-queue decision.
    let cloned = shared.subscriptions.lock().unwrap().get(&7).unwrap().clone();
    assert!(matches!(cloned.tx.try_send(SubEvent::CaughtUp { position: 2 }), Err(mpsc::error::TrySendError::Full(_))));
    assert!(overflow_is_current(&shared, 7, &cloned.tx));
    assert!(forget(&shared, 7));
    assert!(*cloned.ended.borrow());
    assert!(
        !overflow_is_current(&shared, 7, &cloned.tx),
        "an unsubscribed full queue is no longer a connection overflow"
    );
}

#[tokio::test]
async fn stale_full_sender_does_not_close_a_replacement_subscription() {
    let shared = shared();
    let (route, _old_events) = full_route();
    let cloned = route.clone();
    shared.subscriptions.lock().unwrap().insert(7, route);
    let (replacement, _new_events) = full_route();
    let replacement_sender = replacement.tx.clone();
    shared.subscriptions.lock().unwrap().insert(7, replacement);
    assert!(!overflow_is_current(&shared, 7, &cloned.tx));
    assert!(overflow_is_current(&shared, 7, &replacement_sender));
}
