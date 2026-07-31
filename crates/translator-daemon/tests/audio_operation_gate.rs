use std::sync::{
    Arc, Barrier,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use translator_daemon::{AudioOperationGate, AudioOperationState};
use uuid::Uuid;

#[test]
fn gate_starts_idle_and_production_excludes_every_other_operation() {
    let gate = AudioOperationGate::new();
    assert_eq!(gate.state(), AudioOperationState::Idle);

    let production = gate.acquire_production().unwrap();

    assert_eq!(gate.state(), AudioOperationState::Production);
    assert!(gate.acquire_production().is_err());
    assert!(gate.acquire_manual().is_err());
    assert!(gate.acquire_human_round_trip(Uuid::new_v4()).is_err());

    drop(production);
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn human_round_trip_owns_the_gate_with_its_session_id() {
    let gate = AudioOperationGate::new();
    let session_id = Uuid::new_v4();

    let lease = gate.acquire_human_round_trip(session_id).unwrap();

    assert_eq!(
        gate.state(),
        AudioOperationState::HumanRoundTrip { session_id }
    );
    assert!(gate.acquire_production().is_err());
    assert!(gate.acquire_manual().is_err());

    drop(lease);
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn manual_operation_uses_the_production_slot_and_excludes_self_test() {
    let gate = AudioOperationGate::new();

    let manual = gate.acquire_manual().unwrap();

    assert_eq!(gate.state(), AudioOperationState::Production);
    assert!(gate.acquire_manual().is_err());
    assert!(gate.acquire_production().is_err());
    assert!(gate.acquire_human_round_trip(Uuid::new_v4()).is_err());

    drop(manual);
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn stale_explicit_release_cannot_clear_a_newer_owner() {
    let gate = AudioOperationGate::new();
    let mut first = gate.acquire_manual().unwrap();

    assert!(first.release());
    let second_session = Uuid::new_v4();
    let second = gate.acquire_human_round_trip(second_session).unwrap();

    assert!(!first.release());
    assert_eq!(
        gate.state(),
        AudioOperationState::HumanRoundTrip {
            session_id: second_session
        }
    );

    drop(first);
    assert_eq!(
        gate.state(),
        AudioOperationState::HumanRoundTrip {
            session_id: second_session
        }
    );
    drop(second);
    assert_eq!(gate.state(), AudioOperationState::Idle);
}

#[test]
fn daemon_stop_invalidates_active_lease_and_permanently_denies_admission() {
    let gate = AudioOperationGate::new();
    let mut active = gate.acquire_production().unwrap();

    gate.begin_stopping();

    assert_eq!(gate.state(), AudioOperationState::Stopping);
    assert!(!active.release());
    drop(active);
    assert_eq!(gate.state(), AudioOperationState::Stopping);
    assert!(gate.acquire_production().is_err());
    assert!(gate.acquire_manual().is_err());
    assert!(gate.acquire_human_round_trip(Uuid::new_v4()).is_err());
}

#[test]
fn concurrent_admission_has_exactly_one_winner() {
    let gate = Arc::new(AudioOperationGate::new());
    let start = Arc::new(Barrier::new(3));
    let hold_winner = Arc::new(AtomicBool::new(true));
    let (result_tx, result_rx) = mpsc::channel();
    let mut workers = Vec::new();

    for acquire_manual in [false, true] {
        let gate = Arc::clone(&gate);
        let start = Arc::clone(&start);
        let hold_winner = Arc::clone(&hold_winner);
        let result_tx = result_tx.clone();
        workers.push(std::thread::spawn(move || {
            start.wait();
            let result = if acquire_manual {
                gate.acquire_manual()
            } else {
                gate.acquire_production()
            };
            result_tx.send(result.is_ok()).unwrap();
            if let Ok(_lease) = result {
                while hold_winner.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
        }));
    }

    start.wait();
    let wins = [result_rx.recv().unwrap(), result_rx.recv().unwrap()]
        .into_iter()
        .filter(|won| *won)
        .count();
    assert_eq!(wins, 1);
    assert_eq!(gate.state(), AudioOperationState::Production);

    hold_winner.store(false, Ordering::Release);
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(gate.state(), AudioOperationState::Idle);
}
