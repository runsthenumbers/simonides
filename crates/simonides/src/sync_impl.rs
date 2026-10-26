//! `impl SyncableStore for PalaceStore` — M6 per-table CRDT integration.
//!
//! Routes `apply_remote_delta` through `simonides_sync::delta::dispatch_delta`,
//! which calls the per-table merge functions from the sync crate.
//! All writes land in a single atomic DB transaction. Automerge doc bytes
//! are stored in `crdt_docs` (inside the sqlcipher DB — never a plaintext sidecar).

use std::sync::Arc;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::params;

use simonides_core::{SimonidesError, SimonidesResult};
use simonides_core::sync::{
    ApplyOutcome, Hlc, PeerClock, SubscriptionId, SyncDelta, SyncSnapshot, SyncTable,
    SyncableStore, WriteEvent, WriteSink,
};

use simonides_sync::delta::{
    decode_delta, dispatch_delta, encode_delta, DrawerDeltaItem, DrawerLocalState, DrawerWrite,
    LocalState, PalaceDelta,
};
use simonides_sync::kg_merge::{EntityRecord, TripleRecord};
use simonides_sync::tunnel_merge::TunnelRecord;

use crate::{PalaceError, PalaceStore};

impl From<PalaceError> for SimonidesError {
    fn from(e: PalaceError) -> Self {
        SimonidesError::Storage(e.to_string())
    }
}

impl SyncableStore for PalaceStore {
    fn subscribe_writes(&self, sink: Arc<dyn WriteSink>) -> SimonidesResult<SubscriptionId> {
        let id = SubscriptionId(self.next_sub_id.fetch_add(1, AtomicOrdering::SeqCst));
        self.write_sinks.lock().unwrap().push((id, sink));
        Ok(id)
    }

    fn unsubscribe(&self, id: SubscriptionId) -> SimonidesResult<()> {
        self.write_sinks.lock().unwrap().retain(|(sid, _)| *sid != id);
        Ok(())
    }

    fn apply_remote_delta(&self, delta: SyncDelta) -> SimonidesResult<ApplyOutcome> {
        // 1. Decode payload.
        let palace_delta =
            decode_delta(&delta.payload).map_err(|e| SimonidesError::Storage(e.to_string()))?;

        // 2. Load local state, dispatch (pure merge math), and apply writes —
        //    all inside one Mutex acquisition to prevent TOCTOU.
        let (outcome, drawer_ids) = {
            let mut guard = self.conn.lock().unwrap();

            let local = load_local_state(&guard, &palace_delta)
                .map_err(|e| SimonidesError::Storage(e.to_string()))?;

            let result = dispatch_delta(&palace_delta, &local)
                .map_err(|e| SimonidesError::Storage(e.to_string()))?;

            let now = chrono::Utc::now().to_rfc3339();
            let tx = guard
                .transaction()
                .map_err(|e| SimonidesError::Storage(format!("apply_remote_delta tx begin: {e}")))?;

            // Entities BEFORE triples: `triples.subject_id`/`object_id` are FKs into
            // `entities` and the connection runs with `PRAGMA foreign_keys = ON`, so
            // the referenced rows must exist before the triple insert.
            for entity in &result.entity_writes {
                write_entity(&tx, entity, &now).map_err(SimonidesError::from)?;
            }
            // A remote entity id is NOT the local id for the same canonical name:
            // `write_entity` updates the matching local row IN PLACE (preserving its
            // primary key so local triples' FKs survive). A remote triple therefore
            // carries the *sender's* entity ids, which mean nothing here — they are
            // remapped onto the local rows by canonical name, or the triple would
            // violate the FK. Built from the full remote entity set (not just the
            // write-set) so NoChange entity merges are covered too.
            let id_remap = build_entity_id_remap(&tx, &palace_delta).map_err(SimonidesError::from)?;
            for triple in &result.triple_writes {
                write_triple(&tx, &remap_triple(triple, &id_remap)).map_err(SimonidesError::from)?;
            }
            for tunnel in &result.tunnel_writes {
                write_tunnel(&tx, tunnel).map_err(SimonidesError::from)?;
            }
            // Palace structure writes (M6 scope: palace_writes is always empty in practice).
            // Full implementation in M7 when palace tables are synced.

            let drawer_ids: Vec<String> =
                result.drawer_writes.iter().map(|dw| dw.drawer_id.clone()).collect();
            for dw in &result.drawer_writes {
                write_drawer(&tx, dw, &now).map_err(SimonidesError::from)?;
            }

            tx.commit()
                .map_err(|e| SimonidesError::Storage(format!("apply_remote_delta commit: {e}")))?;

            (result.outcome, drawer_ids)
        };

        // 3. Notify write subscribers (outside the Mutex).
        {
            let sinks = self.write_sinks.lock().unwrap();
            if !sinks.is_empty() {
                let hlc = self.current_hlc();
                for id in &drawer_ids {
                    let event =
                        WriteEvent { table: SyncTable::Drawer, record_id: id.clone(), hlc };
                    for (_, sink) in sinks.iter() {
                        sink.on_write(event.clone());
                    }
                }
            }
        }

        Ok(outcome)
    }

    fn produce_delta(&self, _since: &PeerClock) -> SimonidesResult<SyncDelta> {
        // M6: full snapshot. Incremental delta (filter by HLC since last peer sync) is M7.
        let snap = self.snapshot()?;
        Ok(SyncDelta { payload: snap.payload, source_hlc: snap.snapshot_hlc })
    }

    fn snapshot(&self) -> SimonidesResult<SyncSnapshot> {
        let (delta, snapshot_hlc) = {
            let guard = self.conn.lock().unwrap();
            let triples =
                load_all_triples(&guard).map_err(|e| SimonidesError::Storage(e.to_string()))?;
            let entities =
                load_all_entities(&guard).map_err(|e| SimonidesError::Storage(e.to_string()))?;
            let tunnels =
                load_all_tunnels(&guard).map_err(|e| SimonidesError::Storage(e.to_string()))?;
            let drawers =
                load_all_drawers(&guard).map_err(|e| SimonidesError::Storage(e.to_string()))?;
            let hlc = hlc_now();
            let delta = PalaceDelta { drawers, palace_records: vec![], tunnels, triples, entities };
            (delta, hlc)
        };

        let payload = encode_delta(&delta).map_err(|e| SimonidesError::Storage(e.to_string()))?;
        Ok(SyncSnapshot { payload, snapshot_hlc })
    }

    fn current_hlc(&self) -> Hlc {
        hlc_now()
    }
}

// ─── HLC helper ───────────────────────────────────────────────────────────────

pub(crate) fn hlc_now() -> Hlc {
    let wall_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    Hlc { wall_time, logical: 0, node_id: [0u8; 16] }
}

// ─── Load local state ─────────────────────────────────────────────────────────

fn load_local_state(
    conn: &rusqlite::Connection,
    delta: &PalaceDelta,
) -> Result<LocalState, PalaceError> {
    let mut local = LocalState::default();

    for remote in &delta.triples {
        let (s, p, o) = remote.fact_key();
        // Merge partner (deterministic): prefer the row that shares the remote's
        // EXACT window (so a re-synced window coalesces in place); otherwise fall
        // back to the latest window for the fact key (so a remote invalidation or
        // fresh re-assertion is compared against the standing local row — the
        // Scenario-B / partial-invalidation path). A bare unordered LIMIT 1 here
        // would hand `merge_triples` an arbitrary window once a fact key holds more
        // than one — exactly the undefined-selection hazard behind the PK bug.
        let partner = match query_triple_window(conn, &s, &p, &o, remote.valid_from)? {
            Some(t) => Some(t),
            None => query_latest_triple(conn, &s, &p, &o)?,
        };
        if let Some(t) = partner {
            local.triples.push(t);
        }
    }

    for remote in &delta.entities {
        let (name, etype) = remote.identity_key();
        if let Some(e) = query_entity_by_name(conn, &name)? {
            if e.entity_type == etype {
                local.entities.push(e);
            }
        }
    }

    for remote in &delta.tunnels {
        let (from, to, rel) = remote.triple_key();
        if let Some(t) = query_tunnel(conn, &from, &to, &rel)? {
            local.tunnels.push(t);
        }
    }

    for remote in &delta.drawers {
        if let Some(d) = query_drawer_local_state(conn, &remote.drawer_id)? {
            local.drawers.push(d);
        }
    }

    Ok(local)
}

/// Map every remote entity id in `delta` to the id of the local row that now
/// holds the same canonical name (post `write_entity`). Only differing ids are
/// recorded; an entity the local store has never seen keeps its remote id,
/// which is exactly the row `write_entity` just inserted.
fn build_entity_id_remap(
    tx: &rusqlite::Transaction<'_>,
    delta: &PalaceDelta,
) -> Result<std::collections::HashMap<String, String>, PalaceError> {
    let mut remap = std::collections::HashMap::new();
    for remote in &delta.entities {
        let local_id: Option<String> = tx
            .query_row(
                "SELECT id FROM entities WHERE canonical_name = ?1 COLLATE NOCASE",
                params![remote.name],
                |r| r.get(0),
            )
            .ok();
        if let Some(local_id) = local_id {
            if local_id != remote.id {
                remap.insert(remote.id.clone(), local_id);
            }
        }
    }
    Ok(remap)
}

/// Rewrite a triple's subject/object ids through `remap` (identity when the
/// remote id already matches the local row).
fn remap_triple(
    triple: &TripleRecord,
    remap: &std::collections::HashMap<String, String>,
) -> TripleRecord {
    if remap.is_empty() {
        return triple.clone();
    }
    let mut out = triple.clone();
    if let Some(local) = remap.get(&out.subject_id) {
        out.subject_id = local.clone();
    }
    if let Some(local) = remap.get(&out.object_id) {
        out.object_id = local.clone();
    }
    out
}

/// Column list for every triple read, in the order `row_to_triple` decodes.
const TRIPLE_SELECT: &str = "SELECT id, subject_id, predicate, object_id, valid_from, valid_to, \
     confidence, hlc_wall, hlc_logical, hlc_node_id, replaces, replaced_by FROM triples";

/// Run `sql` (which must SELECT the `TRIPLE_SELECT` columns) and return the first
/// row as a `TripleRecord`, or `None` if the query matched nothing.
fn first_triple(
    conn: &rusqlite::Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> Result<Option<TripleRecord>, PalaceError> {
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| PalaceError::Storage(format!("query_triple prepare: {e}")))?;
    let mut rows = stmt
        .query_map(params, row_to_triple)
        .map_err(|e| PalaceError::Storage(format!("query_triple query: {e}")))?;
    match rows.next() {
        Some(r) => Ok(Some(r.map_err(|e| PalaceError::Storage(e.to_string()))?)),
        None => Ok(None),
    }
}

/// The currently-OPEN window for a fact key — the row with `valid_to IS NULL`,
/// newest `valid_from` first. This is the row `kg_add_triple`'s reuse branch and
/// `kg_invalidate` both want: the fact as it stands right now.
///
/// **The `ORDER BY` is load-bearing.** Once a fact key holds more than one
/// window (which the widened `triples_fact_key` index now permits), a bare
/// `LIMIT 1` with no ordering would hand back an arbitrary row — the same
/// undefined-selection hazard that made the PK bug possible. `None` means no
/// window is currently open (every window closed, or the fact key is unknown).
pub(crate) fn query_current_triple(
    conn: &rusqlite::Connection,
    subject: &str,
    predicate: &str,
    object: &str,
) -> Result<Option<TripleRecord>, PalaceError> {
    let sql = format!(
        "{TRIPLE_SELECT} \
         WHERE subject_id = ?1 AND predicate = ?2 AND object_id = ?3 AND valid_to IS NULL \
         ORDER BY valid_from DESC LIMIT 1"
    );
    first_triple(conn, &sql, params![subject, predicate, object])
}

/// The row for one EXACT validity window — `(subject, predicate, object,
/// valid_from)`. `valid_from = None` addresses the NULL-`valid_from` window
/// (SQLite `=` never matches NULL, so the `None` arm uses `IS NULL`). The merge
/// path uses this to coalesce a re-synced window in place instead of duplicating it.
pub(crate) fn query_triple_window(
    conn: &rusqlite::Connection,
    subject: &str,
    predicate: &str,
    object: &str,
    valid_from: Option<chrono::DateTime<chrono::Utc>>,
) -> Result<Option<TripleRecord>, PalaceError> {
    match valid_from {
        Some(vf) => {
            let sql = format!(
                "{TRIPLE_SELECT} \
                 WHERE subject_id = ?1 AND predicate = ?2 AND object_id = ?3 AND valid_from = ?4 \
                 LIMIT 1"
            );
            first_triple(conn, &sql, params![subject, predicate, object, vf.to_rfc3339()])
        }
        None => {
            // NULLs are DISTINCT in a UNIQUE index, so several NULL-valid_from rows
            // can coexist; order by id DESC so the pick is deterministic, not arbitrary.
            let sql = format!(
                "{TRIPLE_SELECT} \
                 WHERE subject_id = ?1 AND predicate = ?2 AND object_id = ?3 AND valid_from IS NULL \
                 ORDER BY id DESC LIMIT 1"
            );
            first_triple(conn, &sql, params![subject, predicate, object])
        }
    }
}

/// The latest row for a fact key by `valid_from` (newest first), regardless of
/// openness, with a deterministic `id DESC` tiebreak (UUIDv7 → creation order).
/// Used to distinguish "already invalidated" from "unknown fact", to link a new
/// window's `replaces` to its predecessor, and as the merge path's fallback
/// partner when no exact-window row matches. `None` iff the fact key is unknown.
pub(crate) fn query_latest_triple(
    conn: &rusqlite::Connection,
    subject: &str,
    predicate: &str,
    object: &str,
) -> Result<Option<TripleRecord>, PalaceError> {
    let sql = format!(
        "{TRIPLE_SELECT} \
         WHERE subject_id = ?1 AND predicate = ?2 AND object_id = ?3 \
         ORDER BY valid_from DESC, id DESC LIMIT 1"
    );
    first_triple(conn, &sql, params![subject, predicate, object])
}

fn row_to_triple(row: &rusqlite::Row<'_>) -> rusqlite::Result<TripleRecord> {
    let wall: i64 = row.get(7)?;
    let logical: i64 = row.get(8)?;
    let node_bytes: Vec<u8> = row.get(9)?;
    let mut node_id = [0u8; 16];
    let len = node_bytes.len().min(16);
    node_id[..len].copy_from_slice(&node_bytes[..len]);

    Ok(TripleRecord {
        id: row.get(0)?,
        subject_id: row.get(1)?,
        predicate: row.get(2)?,
        object_id: row.get(3)?,
        valid_from: row
            .get::<_, Option<String>>(4)?
            .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok()),
        valid_to: row
            .get::<_, Option<String>>(5)?
            .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok()),
        confidence: row.get::<_, f64>(6)? as f32,
        hlc: Hlc { wall_time: wall as u64, logical: logical as u32, node_id },
        replaces: row.get(10)?,
        replaced_by: row.get(11)?,
    })
}

fn query_entity_by_name(
    conn: &rusqlite::Connection,
    name: &str,
) -> Result<Option<EntityRecord>, PalaceError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, canonical_name, entity_type, \
             COALESCE(properties, '{}'), \
             COALESCE(hlc_wall, 0), COALESCE(hlc_logical, 0), hlc_node_id \
             FROM entities \
             WHERE canonical_name = ?1 COLLATE NOCASE \
             LIMIT 1",
        )
        .map_err(|e| PalaceError::Storage(format!("query_entity prepare: {e}")))?;
    let mut rows = stmt
        .query_map(params![name], row_to_entity)
        .map_err(|e| PalaceError::Storage(format!("query_entity query: {e}")))?;
    match rows.next() {
        Some(r) => Ok(Some(r.map_err(|e| PalaceError::Storage(e.to_string()))?)),
        None => Ok(None),
    }
}

fn row_to_entity(row: &rusqlite::Row<'_>) -> rusqlite::Result<EntityRecord> {
    let wall: i64 = row.get(4)?;
    let logical: i64 = row.get(5)?;
    let node_bytes: Option<Vec<u8>> = row.get(6)?;
    let mut node_id = [0u8; 16];
    if let Some(b) = node_bytes {
        let len = b.len().min(16);
        node_id[..len].copy_from_slice(&b[..len]);
    }
    let props_str: String = row.get(3)?;
    let properties = serde_json::from_str(&props_str)
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    Ok(EntityRecord {
        id: row.get(0)?,
        name: row.get(1)?,
        entity_type: row.get(2)?,
        properties,
        hlc: Hlc { wall_time: wall as u64, logical: logical as u32, node_id },
    })
}

pub(crate) fn query_tunnel(
    conn: &rusqlite::Connection,
    from: &str,
    to: &str,
    relation: &str,
) -> Result<Option<TunnelRecord>, PalaceError> {
    let has_tunnels: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tunnels'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    if has_tunnels == 0 {
        return Ok(None);
    }

    let mut stmt = conn
        .prepare(
            "SELECT id, from_drawer_id, to_drawer_id, relation, confidence, \
             hlc_wall, hlc_logical, hlc_node_id, tombstoned \
             FROM tunnels \
             WHERE from_drawer_id = ?1 AND to_drawer_id = ?2 AND relation = ?3 \
             LIMIT 1",
        )
        .map_err(|e| PalaceError::Storage(format!("query_tunnel prepare: {e}")))?;
    let mut rows = stmt
        .query_map(params![from, to, relation], row_to_tunnel)
        .map_err(|e| PalaceError::Storage(format!("query_tunnel query: {e}")))?;
    match rows.next() {
        Some(r) => Ok(Some(r.map_err(|e| PalaceError::Storage(e.to_string()))?)),
        None => Ok(None),
    }
}

pub(crate) fn row_to_tunnel(row: &rusqlite::Row<'_>) -> rusqlite::Result<TunnelRecord> {
    let wall: i64 = row.get(5)?;
    let logical: i64 = row.get(6)?;
    let node_bytes: Vec<u8> = row.get(7)?;
    let mut node_id = [0u8; 16];
    let len = node_bytes.len().min(16);
    node_id[..len].copy_from_slice(&node_bytes[..len]);

    Ok(TunnelRecord {
        id: row.get(0)?,
        from_drawer_id: row.get(1)?,
        to_drawer_id: row.get(2)?,
        relation: row.get(3)?,
        confidence: row.get::<_, f64>(4)? as f32,
        hlc: Hlc { wall_time: wall as u64, logical: logical as u32, node_id },
        tombstoned: row.get::<_, i64>(8)? != 0,
    })
}

fn query_drawer_local_state(
    conn: &rusqlite::Connection,
    drawer_id: &str,
) -> Result<Option<DrawerLocalState>, PalaceError> {
    let row: Option<(String, String, String, String, Option<String>, i64)> = conn
        .query_row(
            "SELECT d.wing, d.room, d.content, d.provenance, d.name, \
             CASE d.state WHEN 'tombstoned' THEN 1 ELSE 0 END \
             FROM drawers d WHERE d.id = ?1",
            params![drawer_id],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            },
        )
        .ok();

    let Some((wing, room, content, provenance, name, tombstoned_int)) = row else {
        return Ok(None);
    };

    let automerge_bytes: Option<Vec<u8>> = conn
        .query_row(
            "SELECT doc_bytes FROM crdt_docs WHERE drawer_id = ?1",
            params![drawer_id],
            |r| r.get(0),
        )
        .ok();

    Ok(Some(DrawerLocalState {
        drawer_id: drawer_id.to_string(),
        content,
        automerge_bytes,
        tombstoned: tombstoned_int != 0,
        wing,
        room,
        provenance,
        name,
    }))
}

// ─── Load-all helpers for snapshot ───────────────────────────────────────────

fn load_all_triples(conn: &rusqlite::Connection) -> Result<Vec<TripleRecord>, PalaceError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, subject_id, predicate, object_id, valid_from, valid_to, confidence, \
             hlc_wall, hlc_logical, hlc_node_id, replaces, replaced_by \
             FROM triples",
        )
        .map_err(|e| PalaceError::Storage(format!("load_all_triples: {e}")))?;
    let rows: Result<Vec<_>, _> = stmt.query_map([], row_to_triple)
        .map_err(|e| PalaceError::Storage(format!("load_all_triples query: {e}")))?
        .map(|r| r.map_err(|e| PalaceError::Storage(e.to_string())))
        .collect();
    rows
}

fn load_all_entities(conn: &rusqlite::Connection) -> Result<Vec<EntityRecord>, PalaceError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, canonical_name, entity_type, \
             COALESCE(properties, '{}'), \
             COALESCE(hlc_wall, 0), COALESCE(hlc_logical, 0), hlc_node_id \
             FROM entities",
        )
        .map_err(|e| PalaceError::Storage(format!("load_all_entities: {e}")))?;
    let rows: Result<Vec<_>, _> = stmt.query_map([], row_to_entity)
        .map_err(|e| PalaceError::Storage(format!("load_all_entities query: {e}")))?
        .map(|r| r.map_err(|e| PalaceError::Storage(e.to_string())))
        .collect();
    rows
}

fn load_all_tunnels(conn: &rusqlite::Connection) -> Result<Vec<TunnelRecord>, PalaceError> {
    let has_tunnels: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tunnels'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    if has_tunnels == 0 {
        return Ok(vec![]);
    }
    let mut stmt = conn
        .prepare(
            "SELECT id, from_drawer_id, to_drawer_id, relation, confidence, \
             hlc_wall, hlc_logical, hlc_node_id, tombstoned \
             FROM tunnels",
        )
        .map_err(|e| PalaceError::Storage(format!("load_all_tunnels: {e}")))?;
    let rows: Result<Vec<_>, _> = stmt.query_map([], row_to_tunnel)
        .map_err(|e| PalaceError::Storage(format!("load_all_tunnels query: {e}")))?
        .map(|r| r.map_err(|e| PalaceError::Storage(e.to_string())))
        .collect();
    rows
}

fn load_all_drawers(conn: &rusqlite::Connection) -> Result<Vec<DrawerDeltaItem>, PalaceError> {
    let hlc_zero = Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] };
    let mut stmt = conn
        .prepare(
            "SELECT d.id, d.wing, d.room, d.content, d.provenance, d.name, \
             CASE d.state WHEN 'tombstoned' THEN 1 ELSE 0 END, \
             cd.doc_bytes \
             FROM drawers d LEFT JOIN crdt_docs cd ON cd.drawer_id = d.id",
        )
        .map_err(|e| PalaceError::Storage(format!("load_all_drawers: {e}")))?;
    let mapped = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, i64>(6)?,
            row.get::<_, Option<Vec<u8>>>(7)?,
        ))
    })
    .map_err(|e| PalaceError::Storage(format!("load_all_drawers query: {e}")))?
    .map(|r| {
        r.map_err(|e| PalaceError::Storage(e.to_string())).map(
            |(id, wing, room, content, provenance, name, tombstoned_int, doc_bytes)| {
                DrawerDeltaItem {
                    drawer_id: id,
                    content,
                    automerge_bytes: doc_bytes.unwrap_or_default(),
                    hlc: hlc_zero,
                    wing,
                    room,
                    provenance,
                    tombstoned: tombstoned_int != 0,
                    name,
                }
            },
        )
    });
    let rows: Result<Vec<_>, _> = mapped.collect();
    rows
}

// ─── Write helpers (operate inside a Transaction) ────────────────────────────

pub(crate) fn write_triple(
    tx: &rusqlite::Transaction<'_>,
    triple: &TripleRecord,
) -> Result<(), PalaceError> {
    let node_bytes: Vec<u8> = triple.hlc.node_id.to_vec();
    tx.execute(
        "INSERT OR REPLACE INTO triples \
         (id, subject_id, predicate, object_id, valid_from, valid_to, confidence, \
          hlc_wall, hlc_logical, hlc_node_id, replaces, replaced_by, source_drawer_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, NULL)",
        params![
            triple.id,
            triple.subject_id,
            triple.predicate,
            triple.object_id,
            triple.valid_from.map(|dt| dt.to_rfc3339()),
            triple.valid_to.map(|dt| dt.to_rfc3339()),
            triple.confidence as f64,
            triple.hlc.wall_time as i64,
            triple.hlc.logical as i64,
            node_bytes,
            triple.replaces,
            triple.replaced_by,
        ],
    )
    .map_err(|e| PalaceError::Storage(format!("write_triple: {e}")))?;
    Ok(())
}

fn write_entity(
    tx: &rusqlite::Transaction<'_>,
    entity: &EntityRecord,
    now: &str,
) -> Result<(), PalaceError> {
    let node_bytes: Vec<u8> = entity.hlc.node_id.to_vec();
    let props_str = entity.properties.to_string();

    let existing_id: Option<String> = tx
        .query_row(
            "SELECT id FROM entities WHERE canonical_name = ?1 COLLATE NOCASE",
            params![entity.name],
            |r| r.get(0),
        )
        .ok();

    if let Some(eid) = existing_id {
        // Update in-place to preserve the row's primary key (avoids FK breakage in triples).
        tx.execute(
            "UPDATE entities \
             SET entity_type = ?1, properties = ?2, \
                 hlc_wall = ?3, hlc_logical = ?4, hlc_node_id = ?5, updated_at = ?6 \
             WHERE id = ?7",
            params![
                entity.entity_type,
                props_str,
                entity.hlc.wall_time as i64,
                entity.hlc.logical as i64,
                node_bytes,
                now,
                eid,
            ],
        )
        .map_err(|e| PalaceError::Storage(format!("write_entity update: {e}")))?;
    } else {
        tx.execute(
            "INSERT INTO entities \
             (id, canonical_name, entity_type, source_drawer_id, \
              properties, hlc_wall, hlc_logical, hlc_node_id, created_at, updated_at) \
             VALUES (?1, ?2, ?3, NULL, ?4, ?5, ?6, ?7, ?8, ?8)",
            params![
                entity.id,
                entity.name,
                entity.entity_type,
                props_str,
                entity.hlc.wall_time as i64,
                entity.hlc.logical as i64,
                node_bytes,
                now,
            ],
        )
        .map_err(|e| PalaceError::Storage(format!("write_entity insert: {e}")))?;
    }
    Ok(())
}

pub(crate) fn write_tunnel(
    tx: &rusqlite::Transaction<'_>,
    tunnel: &TunnelRecord,
) -> Result<(), PalaceError> {
    let node_bytes: Vec<u8> = tunnel.hlc.node_id.to_vec();
    tx.execute(
        "INSERT OR REPLACE INTO tunnels \
         (id, from_drawer_id, to_drawer_id, relation, confidence, \
          hlc_wall, hlc_logical, hlc_node_id, tombstoned) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            tunnel.id,
            tunnel.from_drawer_id,
            tunnel.to_drawer_id,
            tunnel.relation,
            tunnel.confidence as f64,
            tunnel.hlc.wall_time as i64,
            tunnel.hlc.logical as i64,
            node_bytes,
            if tunnel.tombstoned { 1i64 } else { 0i64 },
        ],
    )
    .map_err(|e| PalaceError::Storage(format!("write_tunnel: {e}")))?;
    Ok(())
}

fn write_drawer(
    tx: &rusqlite::Transaction<'_>,
    dw: &DrawerWrite,
    now: &str,
) -> Result<(), PalaceError> {
    // Upsert the drawer row. reindex_pending is always set to 1 (merge = stale indexes).
    tx.execute(
        "INSERT INTO drawers \
         (id, wing, room, content, provenance, state, reindex_pending, created_at, name) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, ?7, ?8) \
         ON CONFLICT(id) DO UPDATE SET \
             content         = excluded.content, \
             wing            = excluded.wing, \
             room            = excluded.room, \
             state           = excluded.state, \
             name            = COALESCE(excluded.name, drawers.name), \
             reindex_pending = 1",
        params![
            dw.drawer_id,
            dw.wing,
            dw.room,
            dw.content,
            dw.provenance,
            if dw.tombstoned { "tombstoned" } else { "live" },
            now,
            dw.name,
        ],
    )
    .map_err(|e| PalaceError::Storage(format!("write_drawer upsert: {e}")))?;

    // Upsert AM bytes into crdt_docs (encrypted at rest — inside sqlcipher DB).
    tx.execute(
        "INSERT INTO crdt_docs (drawer_id, doc_bytes, updated_at) \
         VALUES (?1, ?2, ?3) \
         ON CONFLICT(drawer_id) DO UPDATE SET \
             doc_bytes  = excluded.doc_bytes, \
             updated_at = excluded.updated_at",
        params![dw.drawer_id, dw.automerge_bytes, now],
    )
    .map_err(|e| PalaceError::Storage(format!("write_drawer crdt_docs: {e}")))?;

    // If divergence produced a sibling, insert it and its divergence_from tunnel.
    if let Some(sibling) = &dw.sibling {
        tx.execute(
            "INSERT OR IGNORE INTO drawers \
             (id, wing, room, content, provenance, state, reindex_pending, created_at, name) \
             VALUES (?1, ?2, ?3, ?4, 'simonides-sync (merge)', 'live', 1, ?5, NULL)",
            params![sibling.sibling_id, dw.wing, dw.room, sibling.content, now],
        )
        .map_err(|e| PalaceError::Storage(format!("write_drawer sibling insert: {e}")))?;

        tx.execute(
            "INSERT OR IGNORE INTO tunnels \
             (id, from_drawer_id, to_drawer_id, relation, confidence, \
              hlc_wall, hlc_logical, hlc_node_id, tombstoned) \
             VALUES (?1, ?2, ?3, 'divergence_from', 1.0, 0, 0, ZEROBLOB(16), 0)",
            params![sibling.tunnel_id, sibling.sibling_id, dw.drawer_id],
        )
        .map_err(|e| PalaceError::Storage(format!("write_drawer divergence tunnel: {e}")))?;
    }

    Ok(())
}
