//! Knowledge-graph merge — bitemporal-aware custom merge for triples; LWW + ConflictEvent for entities.
//!
//! Design:
//!
//! ### KG entities (locked review #5)
//! LWW on top-level property keys + ConflictEvent on disagreement. Concurrent
//! `upsert_entity(name, type, properties)` calls with the same `(name, type)`
//! pair dedupe to the lower-HLC entity-id; properties merge per-key with LWW
//! and surface a ConflictEvent if two devices set different values for the
//! same key.
//!
//! ### KG triples (the meat — bitemporal merge)
//!
//! Two time axes:
//! - Wall time (when the event happened in the real world): `valid_from`, `valid_to`
//! - Sync time (when a device learned about the event): the HLC on the sync event
//!
//! Three scenarios:
//!
//! **Scenario A (locked review #2): Concurrent assertions of the same fact.**
//! Two devices independently learn the same triple. COALESCE on `(subject, predicate,
//! object)`: take `min(valid_from)`, `max(valid_to or NULL)`, `max(confidence)`.
//! No ConflictEvent; same fact recorded twice is one fact.
//!
//! **Scenario B: Concurrent invalidation + re-assertion.**
//! Device A invalidates a triple at t=T_A; device B re-asserts the same triple
//! at t=T_B > T_A on a fresh assertion. Preserve BOTH as branches in the temporal
//! timeline. The `replaces`/`replaced_by` chain (locked review #4) links them.
//! No ConflictEvent — the temporal merge is informationally clean.
//!
//! **Scenario C (locked review #3): Concurrent partial-invalidation.**
//! Two devices invalidate the same triple at DIFFERENT `valid_to` timestamps.
//! Operational answer: `min(valid_to)` (conservative — "ended no later than"
//! both claims). ConflictEvent surfaced so assistant can ask user to confirm.

use crate::SyncResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use simonides_core::sync::{ApplyOutcome, ConflictEvent, Hlc};

/// Serde adapter: encodes `serde_json::Value` as a JSON string so that
/// non-self-describing formats (bincode) can round-trip it. The in-memory
/// type stays `serde_json::Value`; only the wire representation changes.
mod json_as_string {
    use serde::{Deserialize, Deserializer, Serializer};
    use serde_json::Value;
    pub fn serialize<S: Serializer>(v: &Value, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&v.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Value, D::Error> {
        let raw = String::deserialize(d)?;
        serde_json::from_str(&raw).map_err(serde::de::Error::custom)
    }
}

/// A KG triple record with the v0.3 schema additions (replaces / replaced_by / hlc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleRecord {
    pub id: String,
    pub subject_id: String,
    pub predicate: String,
    pub object_id: String,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub confidence: f32,
    pub hlc: Hlc,
    pub replaces: Option<String>,
    pub replaced_by: Option<String>,
}

impl TripleRecord {
    /// The identity-class key — two triples sharing this key represent the same
    /// real-world fact (potentially at different times in the temporal timeline).
    pub fn fact_key(&self) -> (String, String, String) {
        (self.subject_id.clone(), self.predicate.clone(), self.object_id.clone())
    }
}

#[derive(Debug, Clone)]
pub enum TripleMergeOutcome {
    /// Scenario A: coalesce to a single triple with merged validity range.
    Coalesced(TripleRecord),
    /// Scenario B: preserve both as a temporal-timeline pair (caller persists both).
    PreserveBoth { earlier: TripleRecord, later: TripleRecord },
    /// Scenario C: concurrent partial-invalidation. Operational answer is `min(valid_to)`;
    /// caller persists the result and emits the ConflictEvent.
    ConcurrentInvalidation {
        merged: TripleRecord,
        conflict: ConflictEvent,
    },
    /// Different fact-keys — keep both as distinct facts.
    Distinct,
    /// Identical inputs.
    NoChange,
}

/// Merge two views of (potentially) the same triple.
pub fn merge_triples(local: &TripleRecord, remote: &TripleRecord) -> TripleMergeOutcome {
    if local.fact_key() != remote.fact_key() {
        return TripleMergeOutcome::Distinct;
    }

    if triples_byte_identical(local, remote) {
        return TripleMergeOutcome::NoChange;
    }

    // Both have valid_to set (both are invalidations) and the values differ →
    // Scenario C: concurrent partial-invalidation.
    if let (Some(l_to), Some(r_to)) = (local.valid_to, remote.valid_to) {
        if l_to != r_to && local.hlc.concurrent_with(&remote.hlc) {
            let min_to = if l_to < r_to { l_to } else { r_to };
            let mut merged = local.clone();
            merged.valid_to = Some(min_to);
            merged.confidence = local.confidence.max(remote.confidence);

            let conflict = ConflictEvent::KgTripleConcurrentInvalidation {
                triple_subject: local.subject_id.clone(),
                triple_predicate: local.predicate.clone(),
                triple_object: local.object_id.clone(),
                local_valid_to: l_to,
                remote_valid_to: r_to,
                operational_choice: min_to,
            };

            return TripleMergeOutcome::ConcurrentInvalidation { merged, conflict };
        }
    }

    // Scenario B: one side is an invalidation (has valid_to set), the other is a
    // FRESH assertion (no replaces link, valid_from > the invalidation's valid_to).
    // Preserve both as branches in the temporal timeline.
    if is_scenario_b(local, remote) {
        let (earlier, later) = order_by_valid_from(local, remote);
        return TripleMergeOutcome::PreserveBoth { earlier, later };
    }

    // Scenario A: concurrent assertions of the same fact (or any other same-key
    // case that isn't B or C). Coalesce.
    TripleMergeOutcome::Coalesced(coalesce_triples(local, remote))
}

fn triples_byte_identical(a: &TripleRecord, b: &TripleRecord) -> bool {
    a.id == b.id
        && a.subject_id == b.subject_id
        && a.predicate == b.predicate
        && a.object_id == b.object_id
        && a.valid_from == b.valid_from
        && a.valid_to == b.valid_to
        && (a.confidence - b.confidence).abs() < f32::EPSILON
        && a.hlc == b.hlc
        && a.replaces == b.replaces
        && a.replaced_by == b.replaced_by
}

/// Scenario B test: one record is closed (valid_to set), the other is a fresh
/// assertion starting AFTER the close. (Scenario B in the tests.)
fn is_scenario_b(a: &TripleRecord, b: &TripleRecord) -> bool {
    let one_closed_one_open = |x: &TripleRecord, y: &TripleRecord| -> bool {
        x.valid_to.is_some() && y.valid_to.is_none()
    };
    if one_closed_one_open(a, b) {
        if let (Some(closed_to), Some(open_from)) = (a.valid_to, b.valid_from) {
            return open_from >= closed_to;
        }
    }
    if one_closed_one_open(b, a) {
        if let (Some(closed_to), Some(open_from)) = (b.valid_to, a.valid_from) {
            return open_from >= closed_to;
        }
    }
    false
}

fn order_by_valid_from(a: &TripleRecord, b: &TripleRecord) -> (TripleRecord, TripleRecord) {
    let a_from = a.valid_from.unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
    let b_from = b.valid_from.unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
    if a_from <= b_from { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) }
}

/// Coalesce two same-fact triples per Scenario A locked rule:
/// `min(valid_from)`, `max(valid_to or NULL)`, `max(confidence)`.
fn coalesce_triples(a: &TripleRecord, b: &TripleRecord) -> TripleRecord {
    let valid_from = match (a.valid_from, b.valid_from) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    };
    // max(valid_to or NULL): NULL represents "still valid (open upper bound)" and
    // should DOMINATE any concrete date (more permissive = max).
    let valid_to = match (a.valid_to, b.valid_to) {
        (None, _) | (_, None) => None,
        (Some(x), Some(y)) => Some(x.max(y)),
    };
    let confidence = a.confidence.max(b.confidence);

    // Pick the earlier-HLC id for deterministic coalescing across devices.
    let (id, hlc) = if a.hlc.total_order_cmp(&b.hlc) == std::cmp::Ordering::Less {
        (a.id.clone(), b.hlc)
    } else {
        (b.id.clone(), a.hlc)
    };

    TripleRecord {
        id,
        subject_id: a.subject_id.clone(),
        predicate: a.predicate.clone(),
        object_id: a.object_id.clone(),
        valid_from,
        valid_to,
        confidence,
        hlc,
        replaces: a.replaces.clone().or_else(|| b.replaces.clone()),
        replaced_by: a.replaced_by.clone().or_else(|| b.replaced_by.clone()),
    }
}

// ─── KG entity merge ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    pub id: String,
    pub name: String,
    pub entity_type: Option<String>,
    /// Stored as a JSON string over the wire so bincode can encode it.
    #[serde(with = "json_as_string")]
    pub properties: serde_json::Value,
    pub hlc: Hlc,
}

impl EntityRecord {
    pub fn identity_key(&self) -> (String, Option<String>) {
        (self.name.clone(), self.entity_type.clone())
    }
}

#[derive(Debug, Clone)]
pub enum EntityMergeOutcome {
    Merged {
        record: EntityRecord,
        property_conflicts: Vec<ConflictEvent>,
    },
    Distinct,
    NoChange,
}

/// Merge two views of (potentially) the same KG entity. Per locked review #5:
/// LWW on top-level property keys + ConflictEvent on disagreement.
pub fn merge_entities(local: &EntityRecord, remote: &EntityRecord) -> EntityMergeOutcome {
    if local.identity_key() != remote.identity_key() {
        return EntityMergeOutcome::Distinct;
    }
    if local.hlc == remote.hlc && local.properties == remote.properties {
        return EntityMergeOutcome::NoChange;
    }

    // Pick the earlier-HLC id for deterministic dedupe.
    let earlier_id = if local.hlc.total_order_cmp(&remote.hlc) == std::cmp::Ordering::Less {
        local.id.clone()
    } else {
        remote.id.clone()
    };
    let later_hlc = if local.hlc.total_order_cmp(&remote.hlc) == std::cmp::Ordering::Greater {
        local.hlc
    } else {
        remote.hlc
    };

    let (merged_props, conflicts) = merge_property_maps(
        &local.properties,
        &remote.properties,
        &local.hlc,
        &remote.hlc,
        &earlier_id,
    );

    EntityMergeOutcome::Merged {
        record: EntityRecord {
            id: earlier_id,
            name: local.name.clone(),
            entity_type: local.entity_type.clone(),
            properties: merged_props,
            hlc: later_hlc,
        },
        property_conflicts: conflicts,
    }
}

fn merge_property_maps(
    a: &serde_json::Value,
    b: &serde_json::Value,
    a_hlc: &Hlc,
    b_hlc: &Hlc,
    entity_id: &str,
) -> (serde_json::Value, Vec<ConflictEvent>) {
    use serde_json::Value;
    let (Value::Object(am), Value::Object(bm)) = (a, b) else {
        // Non-object properties — LWW the whole blob; no per-key surfacing.
        return if a_hlc.total_order_cmp(b_hlc) == std::cmp::Ordering::Less {
            (b.clone(), vec![])
        } else {
            (a.clone(), vec![])
        };
    };

    let mut merged = serde_json::Map::new();
    let mut conflicts = Vec::new();

    let mut all_keys: std::collections::BTreeSet<&String> = am.keys().collect();
    all_keys.extend(bm.keys());

    for key in all_keys {
        let av = am.get(key);
        let bv = bm.get(key);
        match (av, bv) {
            (Some(x), None) => { merged.insert(key.clone(), x.clone()); }
            (None, Some(y)) => { merged.insert(key.clone(), y.clone()); }
            (Some(x), Some(y)) if x == y => { merged.insert(key.clone(), x.clone()); }
            (Some(x), Some(y)) => {
                // Disagreement on the same key — LWW + surface conflict.
                let winner = if a_hlc.total_order_cmp(b_hlc) == std::cmp::Ordering::Greater { x } else { y };
                let winning_hlc = if a_hlc.total_order_cmp(b_hlc) == std::cmp::Ordering::Greater { *a_hlc } else { *b_hlc };
                merged.insert(key.clone(), winner.clone());
                conflicts.push(ConflictEvent::KgEntityPropertyDisagreement {
                    entity_id: entity_id.to_string(),
                    property_key: key.clone(),
                    local_value: x.clone(),
                    remote_value: y.clone(),
                    winning_hlc,
                });
            }
            (None, None) => {}
        }
    }

    (serde_json::Value::Object(merged), conflicts)
}

pub fn aggregate_kg_outcomes(
    triple_outcomes: Vec<TripleMergeOutcome>,
    entity_outcomes: Vec<EntityMergeOutcome>,
) -> SyncResult<ApplyOutcome> {
    let mut conflicts = Vec::new();
    for outcome in triple_outcomes {
        if let TripleMergeOutcome::ConcurrentInvalidation { conflict, .. } = outcome {
            conflicts.push(conflict);
        }
    }
    for outcome in entity_outcomes {
        if let EntityMergeOutcome::Merged { property_conflicts, .. } = outcome {
            for c in property_conflicts {
                conflicts.push(c);
            }
        }
    }
    if conflicts.is_empty() {
        Ok(ApplyOutcome::Clean)
    } else {
        Ok(ApplyOutcome::ConflictSurfaced { conflicts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn hlc(wall: u64, logical: u32, node_seed: u8) -> Hlc {
        let mut node_id = [0u8; 16];
        node_id[0] = node_seed;
        Hlc { wall_time: wall, logical, node_id }
    }

    fn dt(year: i32, month: u32, day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0).unwrap()
    }

    #[allow(clippy::too_many_arguments)] // test fixture: one argument per record field
    fn triple(
        id: &str,
        subj: &str,
        pred: &str,
        obj: &str,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        conf: f32,
        h: Hlc,
    ) -> TripleRecord {
        TripleRecord {
            id: id.to_string(),
            subject_id: subj.to_string(),
            predicate: pred.to_string(),
            object_id: obj.to_string(),
            valid_from: from,
            valid_to: to,
            confidence: conf,
            hlc: h,
            replaces: None,
            replaced_by: None,
        }
    }

    #[test]
    fn scenario_a_concurrent_assertions_coalesce() {
        // Worked example: device A says Mira lives_in Dili from 2024-01-01,
        // device B says Mira lives_in Dili from 2024-02-15. Same fact recorded
        // twice → coalesce; take min(valid_from) = 2024-01-01.
        let a = triple("t1", "mira", "lives_in", "dili",
            Some(dt(2024, 1, 1)), None, 0.9, hlc(100, 0, 0xAA));
        let b = triple("t2", "mira", "lives_in", "dili",
            Some(dt(2024, 2, 15)), None, 0.85, hlc(110, 0, 0xBB));
        let outcome = merge_triples(&a, &b);
        match outcome {
            TripleMergeOutcome::Coalesced(merged) => {
                assert_eq!(merged.valid_from, Some(dt(2024, 1, 1)));
                assert_eq!(merged.valid_to, None);
                assert!((merged.confidence - 0.9).abs() < 0.01);
            }
            _ => panic!("expected Coalesced"),
        }
    }

    #[test]
    fn scenario_b_invalidation_plus_re_assertion_preserves_both() {
        // Worked example: pre-state was Mira lives_in Dili (open). Device A
        // invalidated at 2026-04-01. Device B re-asserted lives_in Dili
        // (valid_from=2026-05-15, open). Preserve both branches.
        let invalidation = triple("t1", "mira", "lives_in", "dili",
            Some(dt(2024, 1, 1)), Some(dt(2026, 4, 1)), 0.9, hlc(100, 0, 0xAA));
        let re_assertion = triple("t2", "mira", "lives_in", "dili",
            Some(dt(2026, 5, 15)), None, 1.0, hlc(110, 0, 0xBB));
        let outcome = merge_triples(&invalidation, &re_assertion);
        match outcome {
            TripleMergeOutcome::PreserveBoth { earlier, later } => {
                assert_eq!(earlier.valid_to, Some(dt(2026, 4, 1)));
                assert_eq!(later.valid_from, Some(dt(2026, 5, 15)));
                assert_eq!(later.valid_to, None);
            }
            _ => panic!("expected PreserveBoth"),
        }
    }

    #[test]
    fn scenario_c_concurrent_partial_invalidation_min_plus_conflict() {
        // Worked example: device A invalidates with ended=2026-03-01;
        // device B invalidates with ended=2026-04-15. HLCs concurrent.
        // Operational answer: min(valid_to) = 2026-03-01. ConflictEvent.
        let a = triple("t1", "sarah", "works_on", "project-x",
            Some(dt(2025, 1, 1)), Some(dt(2026, 3, 1)), 0.9, hlc(100, 0, 0xAA));
        let b = triple("t1", "sarah", "works_on", "project-x",
            Some(dt(2025, 1, 1)), Some(dt(2026, 4, 15)), 0.9, hlc(100, 0, 0xBB));
        let outcome = merge_triples(&a, &b);
        match outcome {
            TripleMergeOutcome::ConcurrentInvalidation { merged, conflict } => {
                assert_eq!(merged.valid_to, Some(dt(2026, 3, 1)));
                match conflict {
                    ConflictEvent::KgTripleConcurrentInvalidation {
                        operational_choice, ..
                    } => assert_eq!(operational_choice, dt(2026, 3, 1)),
                    _ => panic!("wrong conflict variant"),
                }
            }
            _ => panic!("expected ConcurrentInvalidation"),
        }
    }

    #[test]
    fn distinct_facts_kept_separate() {
        let a = triple("t1", "mira", "lives_in", "dili", None, None, 0.9, hlc(100, 0, 0));
        let b = triple("t2", "mira", "lives_in", "usa", None, None, 0.9, hlc(100, 0, 0));
        let outcome = merge_triples(&a, &b);
        assert!(matches!(outcome, TripleMergeOutcome::Distinct));
    }

    #[test]
    fn entity_property_disagreement_surfaces_conflict() {
        let a = EntityRecord {
            id: "e1".into(),
            name: "Mira".into(),
            entity_type: Some("person".into()),
            properties: serde_json::json!({ "role": "founder", "city": "Dili" }),
            hlc: hlc(100, 0, 0xAA),
        };
        let b = EntityRecord {
            id: "e2".into(),
            name: "Mira".into(),
            entity_type: Some("person".into()),
            properties: serde_json::json!({ "role": "CEO", "city": "Dili" }),
            hlc: hlc(110, 0, 0xBB),
        };
        let outcome = merge_entities(&a, &b);
        match outcome {
            EntityMergeOutcome::Merged { record, property_conflicts } => {
                assert_eq!(property_conflicts.len(), 1);
                // remote has later HLC → CEO wins on disagreement
                assert_eq!(record.properties.get("role"), Some(&serde_json::json!("CEO")));
                // Non-conflicting field "city" carries through
                assert_eq!(record.properties.get("city"), Some(&serde_json::json!("Dili")));
            }
            _ => panic!("expected Merged"),
        }
    }

    #[test]
    fn entity_property_merge_combines_disjoint_keys() {
        // Worked example (entity merge): device A adds
        // {role: founder}, device B adds {city: Dili}. Properties merge.
        let a = EntityRecord {
            id: "e1".into(),
            name: "Mira".into(),
            entity_type: Some("person".into()),
            properties: serde_json::json!({ "role": "founder" }),
            hlc: hlc(100, 0, 0xAA),
        };
        let b = EntityRecord {
            id: "e2".into(),
            name: "Mira".into(),
            entity_type: Some("person".into()),
            properties: serde_json::json!({ "city": "Dili" }),
            hlc: hlc(110, 0, 0xBB),
        };
        let outcome = merge_entities(&a, &b);
        match outcome {
            EntityMergeOutcome::Merged { record, property_conflicts } => {
                assert!(property_conflicts.is_empty());
                assert_eq!(record.properties.get("role"), Some(&serde_json::json!("founder")));
                assert_eq!(record.properties.get("city"), Some(&serde_json::json!("Dili")));
            }
            _ => panic!("expected Merged"),
        }
    }
}
