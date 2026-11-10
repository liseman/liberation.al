//! Every query and result round-trips through its bytes, and a full result stays an inline payload.

use continuum_search::{Filter, Hit, Key, MAX_K, TextQuery, VectorQuery, decode_hits, encode_hits};
use proptest::prelude::*;
use uuid::Uuid;

fn uuid() -> impl Strategy<Value = Uuid> {
    any::<u128>().prop_map(Uuid::from_u128)
}

fn filter() -> impl Strategy<Value = Filter> {
    (
        proptest::option::of((0u64..u64::MAX - 1).prop_flat_map(|a| (Just(a), a + 1..u64::MAX))),
        prop::collection::vec(uuid(), 0..4),
        proptest::option::of(any::<bool>()),
    )
        .prop_map(|(during, subjects, labelled)| Filter { during, subjects, labelled })
}

fn hit() -> impl Strategy<Value = Hit> {
    (uuid(), proptest::option::of(uuid()), proptest::option::of(uuid()), any::<u64>(), any::<u64>(), -1e6f32..1e6)
        .prop_map(|(payload_id, entity_id, parent_id, start_ns, end_ns, score)| Hit {
            key: Key { payload_id, entity_id, parent_id, start_ns, end_ns },
            score,
        })
}

proptest! {
    #[test]
    fn a_vector_query_round_trips(
        model in "[a-z0-9_.-]{1,12}",
        version in "[a-z0-9_.-]{1,12}",
        k in 1..=MAX_K,
        vector in prop::collection::vec(-1e6f32..1e6, 1..600),
        filter in filter(),
    ) {
        let q = VectorQuery { model, version, k, vector, filter };
        prop_assert_eq!(VectorQuery::decode(&q.encode().unwrap()).unwrap(), q);
    }

    #[test]
    fn a_text_query_round_trips(text in ".{0,200}", k in 1..=MAX_K, filter in filter()) {
        let q = TextQuery { text, k, filter };
        prop_assert_eq!(TextQuery::decode(&q.encode().unwrap()).unwrap(), q);
    }

    #[test]
    fn hits_round_trip(hits in prop::collection::vec(hit(), 0..50)) {
        prop_assert_eq!(decode_hits(&encode_hits(&hits).unwrap()).unwrap(), hits);
    }

    #[test]
    fn a_full_result_stays_an_inline_payload(hits in prop::collection::vec(hit(), MAX_K as usize)) {
        // 4 KiB was the broker's inline limit when MAX_K was set; results stay that small.
        prop_assert!(encode_hits(&hits).unwrap().len() <= 4096);
    }
}

#[test]
fn a_query_that_would_only_fail_at_the_worker_is_refused_when_encoded() {
    let q = |vector: Vec<f32>, during| VectorQuery {
        model: "m".into(),
        version: "v".into(),
        k: 5,
        vector,
        filter: Filter { during, ..Filter::default() },
    };
    assert!(q(vec![], None).encode().is_err());
    assert!(q(vec![1.0, f32::NAN], None).encode().is_err());
    assert!(q(vec![1.0], Some((9, 5))).encode().is_err());
    assert!(q(vec![1.0], Some((5, 5))).encode().is_err());
    assert!(q(vec![1.0], Some((5, 9))).encode().is_ok());
}
