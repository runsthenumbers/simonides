//! Delta encoding + per-table routing — M6 per-table CRDT integration.
//!
//! `PalaceDelta` is the wire format for `SyncDelta.payload` (bincode-encoded).
//! `dispatch_delta` routes each table's incoming records through the
//! per-table merge functions — merge math is in the existing merge modules;
//! this module is the integration bridge that calls them.

use serde::{Deserialize, Serialize};

use simonides_core::sync::{ApplyOutcome, Hlc, SyncTable};

use crate::{
    aggregate_outcomes,
    drawer_merge::{apply_drawer_merge, resolve_delete_vs_edit, DrawerMergeInputs},
    kg_merge::{
        aggregate_kg_outcomes, merge_entities, merge_triples, EntityMergeOutcome, EntityRecord,
        TripleMergeOutcome, TripleRecord,
    },
    palace_merge::{aggregate_palace_outcomes, merge_palace_records, PalaceMergeOutcome, PalaceRecord},
    tunnel_merge::{aggregate_tunnel_outcomes, merge_tunnels, TunnelMergeOutcome, TunnelRecord},
    SyncError, SyncResult,
};

// ─── Wire types ───────────────────────────────────────────────────────────────

/// Full per-table payload encoded inside `SyncDelta.payload` (bincode).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PalaceDelta {
    pub drawers: Vec<DrawerDeltaItem>,
    /// Palace structure records (wings, rooms, closets).
    pub palace_records: Vec<PalaceRecord>,
    pub tunnels: Vec<TunnelRecord>,
    pub triples: Vec<TripleRecord>,
    pub entities: Vec<EntityRecord>,
}

/// One drawer's contribution to a delta from a remote peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawerDeltaItem {
    pub drawer_id: String,
    pub content: String,
    /// Serialised Automerge document bytes (`doc.save()`).
    /// Empty Vec means the remote has no AM doc yet; receiver creates one.
    pub automerge_bytes: Vec<u8>,
    pub hlc: Hlc,
    pub wing: String,
    pub room: String,
    pub provenance: String,
    pub tombstoned: bool,
    pub name: Option<String>,
}

/// Encode a `PalaceDelta` for `SyncDelta.payload`.
pub fn encode_delta(delta: &PalaceDelta) -> SyncResult<Vec<u8>> {
    bincode::serialize(delta).map_err(|e| SyncError::Internal(format!("encode_delta: {e}")))
}

/// Decode `SyncDelta.payload` back to a `PalaceDelta`.
pub fn decode_delta(bytes: &[u8]) -> SyncResult<PalaceDelta> {
    bincode::deserialize(bytes).map_err(|e| SyncError::InvalidDelta(format!("decode_delta: {e}")))
}

// ─── Local-state inputs ───────────────────────────────────────────────────────

/// Current local state for every row referenced by an incoming `PalaceDelta`.
/// The caller (the store's `apply_remote_delta`) loads these from the DB.
#[derive(Debug, Default)]
pub struct LocalState {
    pub triples: Vec<TripleRecord>,
    pub entities: Vec<EntityRecord>,
    pub palace_records: Vec<PalaceRecord>,
    pub tunnels: Vec<TunnelRecord>,
    pub drawers: Vec<DrawerLocalState>,
}

/// Local state for a single drawer row.
#[derive(Debug)]
pub struct DrawerLocalState {
    pub drawer_id: String,
    pub content: String,
    /// Automerge doc bytes from `crdt_docs`, or None if never sync-merged before.
    pub automerge_bytes: Option<Vec<u8>>,
    pub tombstoned: bool,
    pub wing: String,
    pub room: String,
    pub provenance: String,
    pub name: Option<String>,
}

// ─── Dispatch output ──────────────────────────────────────────────────────────

/// Everything the caller needs to write back to the DB after a successful merge.
#[derive(Debug)]
pub struct DispatchResult {
    pub outcome: ApplyOutcome,
    pub triple_writes: Vec<TripleRecord>,
    pub entity_writes: Vec<EntityRecord>,
    /// (record, SyncTable) — palace structure records to upsert.
    pub palace_writes: Vec<(PalaceRecord, SyncTable)>,
    pub tunnel_writes: Vec<TunnelRecord>,
    pub drawer_writes: Vec<DrawerWrite>,
}

/// A merged drawer to write back to the DB.
#[derive(Debug)]
pub struct DrawerWrite {
    pub drawer_id: String,
    pub content: String,
    /// Merged Automerge doc bytes — goes into `crdt_docs`.
    pub automerge_bytes: Vec<u8>,
    pub tombstoned: bool,
    pub wing: String,
    pub room: String,
    pub provenance: String,
    pub name: Option<String>,
    /// Present when divergence was detected and a sibling was materialized.
    pub sibling: Option<SiblingDrawer>,
}

/// A sibling drawer created when Automerge produced content neither side authored.
#[derive(Debug)]
pub struct SiblingDrawer {
    pub sibling_id: String,
    /// The Automerge-synthesized merged content.
    pub content: String,
    pub tunnel_id: String,
}

// ─── The real dispatch_delta ──────────────────────────────────────────────────

/// Route an incoming `PalaceDelta` through the per-table merge functions.
///
/// Pure computation — no I/O.  Caller loads `local` from DB before calling and
/// applies `DispatchResult` writes inside a single DB transaction after returning.
pub fn dispatch_delta(delta: &PalaceDelta, local: &LocalState) -> SyncResult<DispatchResult> {
    let mut all_outcomes: Vec<ApplyOutcome> = Vec::new();

    let (triple_writes, triple_outcomes, entity_writes, entity_outcomes) =
        route_kg(&delta.triples, &delta.entities, local);
    all_outcomes.push(aggregate_kg_outcomes(triple_outcomes, entity_outcomes)?);

    let (palace_writes, palace_outcomes) = route_palace(&delta.palace_records, local);
    all_outcomes.push(aggregate_palace_outcomes(palace_outcomes)?);

    let (tunnel_writes, tunnel_outcomes) = route_tunnels(&delta.tunnels, local);
    all_outcomes.push(aggregate_tunnel_outcomes(tunnel_outcomes)?);

    let (drawer_writes, drawer_outcomes) = route_drawers(&delta.drawers, local)?;
    all_outcomes.extend(drawer_outcomes);

    Ok(DispatchResult {
        outcome: aggregate_outcomes(all_outcomes),
        triple_writes,
        entity_writes,
        palace_writes,
        tunnel_writes,
        drawer_writes,
    })
}

// ─── Per-table routing ────────────────────────────────────────────────────────

fn route_kg(
    remote_triples: &[TripleRecord],
    remote_entities: &[EntityRecord],
    local: &LocalState,
) -> (Vec<TripleRecord>, Vec<TripleMergeOutcome>, Vec<EntityRecord>, Vec<EntityMergeOutcome>) {
    let mut triple_writes = Vec::new();
    let mut triple_outcomes = Vec::new();

    for remote in remote_triples {
        let local_match = local.triples.iter().find(|t| t.fact_key() == remote.fact_key());
        if let Some(local_t) = local_match {
            let outcome = merge_triples(local_t, remote);
            match &outcome {
                TripleMergeOutcome::Coalesced(r) => triple_writes.push(r.clone()),
                TripleMergeOutcome::PreserveBoth { earlier, later } => {
                    // Two validity windows of one fact = two rows. The palace
                    // `triples_fact_key` UNIQUE index is
                    // (subject_id, predicate, object_id, valid_from) as of schema v5,
                    // and `write_triple`'s INSERT OR REPLACE keys on `id`; earlier and
                    // later carry distinct ids and distinct valid_from, so both land —
                    // neither overwrites the other. (The old single-row limitation this
                    // note used to describe was the PK data-loss bug, now fixed.)
                    triple_writes.push(earlier.clone());
                    triple_writes.push(later.clone());
                }
                TripleMergeOutcome::ConcurrentInvalidation { merged, .. } => {
                    triple_writes.push(merged.clone());
                }
                TripleMergeOutcome::NoChange | TripleMergeOutcome::Distinct => {}
            }
            triple_outcomes.push(outcome);
        } else {
            // No local match → new fact; write remote verbatim.
            triple_writes.push(remote.clone());
        }
    }

    let mut entity_writes = Vec::new();
    let mut entity_outcomes = Vec::new();

    for remote in remote_entities {
        let local_match = local
            .entities
            .iter()
            .find(|e| e.identity_key() == remote.identity_key());
        if let Some(local_e) = local_match {
            let outcome = merge_entities(local_e, remote);
            if let EntityMergeOutcome::Merged { ref record, .. } = outcome {
                entity_writes.push(record.clone());
            }
            entity_outcomes.push(outcome);
        } else {
            entity_writes.push(remote.clone());
        }
    }

    (triple_writes, triple_outcomes, entity_writes, entity_outcomes)
}

fn route_palace(
    remote_records: &[PalaceRecord],
    local: &LocalState,
) -> (Vec<(PalaceRecord, SyncTable)>, Vec<PalaceMergeOutcome>) {
    let mut writes = Vec::new();
    let mut outcomes = Vec::new();

    for remote in remote_records {
        let local_match = local
            .palace_records
            .iter()
            .find(|r| r.record_id == remote.record_id && r.table == remote.table);
        if let Some(local_r) = local_match {
            let outcome = merge_palace_records(local_r, remote);
            if matches!(outcome, PalaceMergeOutcome::TakeRemote) {
                writes.push((remote.clone(), remote.table));
            }
            outcomes.push(outcome);
        } else {
            writes.push((remote.clone(), remote.table));
        }
    }

    (writes, outcomes)
}

fn route_tunnels(
    remote_tunnels: &[TunnelRecord],
    local: &LocalState,
) -> (Vec<TunnelRecord>, Vec<TunnelMergeOutcome>) {
    let mut writes = Vec::new();
    let mut outcomes = Vec::new();

    for remote in remote_tunnels {
        let local_match = local
            .tunnels
            .iter()
            .find(|t| t.triple_key() == remote.triple_key());
        if let Some(local_t) = local_match {
            let outcome = merge_tunnels(local_t, remote);
            match &outcome {
                TunnelMergeOutcome::Merged(t) => writes.push(t.clone()),
                TunnelMergeOutcome::Distinct => writes.push(remote.clone()),
                TunnelMergeOutcome::NoChange => {}
            }
            outcomes.push(outcome);
        } else {
            writes.push(remote.clone());
        }
    }

    (writes, outcomes)
}

fn route_drawers(
    remote_drawers: &[DrawerDeltaItem],
    local: &LocalState,
) -> SyncResult<(Vec<DrawerWrite>, Vec<ApplyOutcome>)> {
    let mut writes = Vec::new();
    let mut outcomes = Vec::new();

    for remote in remote_drawers {
        let local_match = local.drawers.iter().find(|d| d.drawer_id == remote.drawer_id);

        if let Some(local_d) = local_match {
            if local_d.tombstoned {
                // Edit (remote) vs delete (local tombstone): edit wins, ghost preserved.
                let outcome =
                    resolve_delete_vs_edit(&remote.drawer_id, remote.hlc, remote.hlc);
                let am_bytes = merge_am_docs(
                    local_d.automerge_bytes.as_deref(),
                    &remote.automerge_bytes,
                    &remote.content,
                )?;
                writes.push(DrawerWrite {
                    drawer_id: remote.drawer_id.clone(),
                    content: local_d.content.clone(), // tombstoned content preserved
                    automerge_bytes: am_bytes,
                    tombstoned: true,
                    wing: local_d.wing.clone(),
                    room: local_d.room.clone(),
                    provenance: local_d.provenance.clone(),
                    name: local_d.name.clone(),
                    sibling: None,
                });
                outcomes.push(outcome);
            } else {
                // Both live: Automerge merge + divergence detection.
                let (am_bytes, merged_content) = merge_am_docs_with_content(
                    local_d.automerge_bytes.as_deref(),
                    &remote.automerge_bytes,
                    &remote.content,
                )?;
                let inputs = DrawerMergeInputs {
                    drawer_id: &remote.drawer_id,
                    local_pre: &local_d.content,
                    remote_pre: &remote.content,
                    merged: &merged_content,
                };
                let outcome = apply_drawer_merge(&inputs, remote.hlc)?;

                let sibling = match &outcome {
                    ApplyOutcome::DivergencePreserved {
                        sibling_drawer_id,
                        tunnel_id,
                        ..
                    } => Some(SiblingDrawer {
                        sibling_id: sibling_drawer_id.clone(),
                        content: merged_content.clone(),
                        tunnel_id: tunnel_id.clone(),
                    }),
                    _ => None,
                };

                // On divergence: original keeps local (pre-merge) content; sibling gets merged.
                let content = if sibling.is_some() {
                    local_d.content.clone()
                } else {
                    merged_content
                };

                writes.push(DrawerWrite {
                    drawer_id: remote.drawer_id.clone(),
                    content,
                    automerge_bytes: am_bytes,
                    tombstoned: false,
                    wing: remote.wing.clone(),
                    room: remote.room.clone(),
                    provenance: local_d.provenance.clone(),
                    name: remote.name.clone().or_else(|| local_d.name.clone()),
                    sibling,
                });
                outcomes.push(outcome);
            }
        } else {
            // No local match: insert remote drawer as a new entry.
            let am_bytes = if remote.automerge_bytes.is_empty() {
                make_am_doc(&remote.content)?
            } else {
                remote.automerge_bytes.clone()
            };
            writes.push(DrawerWrite {
                drawer_id: remote.drawer_id.clone(),
                content: remote.content.clone(),
                automerge_bytes: am_bytes,
                tombstoned: remote.tombstoned,
                wing: remote.wing.clone(),
                room: remote.room.clone(),
                provenance: remote.provenance.clone(),
                name: remote.name.clone(),
                sibling: None,
            });
            outcomes.push(ApplyOutcome::Clean);
        }
    }

    Ok((writes, outcomes))
}

// ─── Automerge helpers ────────────────────────────────────────────────────────

/// Merge two Automerge documents and return the merged bytes (no content extraction).
fn merge_am_docs(
    local_bytes: Option<&[u8]>,
    remote_bytes: &[u8],
    remote_content_fallback: &str,
) -> SyncResult<Vec<u8>> {
    Ok(merge_am_docs_with_content(local_bytes, remote_bytes, remote_content_fallback)?.0)
}

/// Merge two Automerge documents and return (merged_bytes, merged_content_string).
fn merge_am_docs_with_content(
    local_bytes: Option<&[u8]>,
    remote_bytes: &[u8],
    remote_content_fallback: &str,
) -> SyncResult<(Vec<u8>, String)> {
    use automerge::transaction::Transactable;
    use automerge::{AutoCommit, ReadDoc, ScalarValue, Value};

    let mut local_doc: AutoCommit = match local_bytes {
        Some(b) if !b.is_empty() => AutoCommit::load(b)
            .map_err(|e| SyncError::Automerge(format!("load local doc: {e}")))?,
        _ => {
            // No existing AM doc for this drawer: bootstrap from remote content.
            let mut doc = AutoCommit::new();
            doc.put(automerge::ROOT, "content", remote_content_fallback)
                .map_err(|e| SyncError::Automerge(format!("bootstrap put: {e}")))?;
            doc
        }
    };

    if !remote_bytes.is_empty() {
        let mut remote_doc = AutoCommit::load(remote_bytes)
            .map_err(|e| SyncError::Automerge(format!("load remote doc: {e}")))?;
        local_doc
            .merge(&mut remote_doc)
            .map_err(|e| SyncError::Automerge(format!("merge: {e}")))?;
    }

    let merged_content = match local_doc
        .get(automerge::ROOT, "content")
        .map_err(|e| SyncError::Automerge(format!("get content key: {e}")))?
    {
        Some((Value::Scalar(sv), _)) => match sv.as_ref() {
            ScalarValue::Str(s) => s.to_string(),
            _ => remote_content_fallback.to_string(),
        },
        _ => remote_content_fallback.to_string(),
    };

    Ok((local_doc.save(), merged_content))
}

/// Create a fresh Automerge document seeded with the given text content.
fn make_am_doc(content: &str) -> SyncResult<Vec<u8>> {
    use automerge::transaction::Transactable;
    use automerge::AutoCommit;
    let mut doc = AutoCommit::new();
    doc.put(automerge::ROOT, "content", content)
        .map_err(|e| SyncError::Automerge(format!("make_am_doc: {e}")))?;
    Ok(doc.save())
}
