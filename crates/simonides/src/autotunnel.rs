//! Auto-tunnel generation — the deterministic, idempotent derivation pass that
//! FILLS the (already-built) tunnel machinery from KG + vectors + metadata.
//!
//! Discipline (mirrors closets, the other regenerable projection):
//!  * Every derived edge is `auto:`-namespaced. MANUAL edges (any relation not
//!    starting with `auto:`) are invariant — never read as candidates, never
//!    rewritten, never tombstoned here.
//!  * Re-running RECONCILES: the live `auto:` edge set converges to exactly what
//!    the current data derives. Unchanged edges are left byte-identical (no HLC
//!    advance); edges no longer derivable are tombstoned; newly derivable edges
//!    are created. A second run over unchanged data is a write-layer no-op.
//!  * Deterministic: fixed sort orders everywhere, so the same store always
//!    yields the same edge set regardless of row/iteration order.
//!
//! Sources built (spec §1):
//!  1. Entity co-occurrence  -> `auto:coentity:<entity>`  (selectivity guard:
//!     an entity links its drawers only if 2 <= drawer-freq <= K).
//!  3. Semantic similarity   -> `auto:similar`            (per-drawer top-N
//!     nearest neighbours above cosine tau).
//!  2. Provenance/structural -> `auto:same_thread`        (same wing+room;
//!     bounded group size). NOTE: `auto:same_source` and `auto:chunk_of` are
//!     intentionally NOT emitted for this corpus — see the branch flag: migrated
//!     provenance is the author (coarse, not a source_file), and chunks do not
//!     span drawers in this schema, so neither carries distinct signal here.

use crate::sync_impl;
use simonides_sync::tunnel_merge::TunnelRecord;
use crate::{PalaceError, PalaceStore, Result};
use rusqlite::{params, Connection};
use std::collections::{BTreeSet, HashMap, HashSet};
use uuid::Uuid;

/// The namespace every derived edge carries; manual edges never do.
pub const AUTO_PREFIX: &str = "auto:";

// ── Spec knobs (spec defaults; tunable) ──────────────────────────────────────
const ENTITY_FREQ_MIN: usize = 2;
const ENTITY_FREQ_MAX: usize = 20; // K — the load-bearing hub guard
const SIM_TOP_N: usize = 5;
const SIM_TAU: f32 = 0.82;
const FANOUT_CAP: usize = 30; // max live auto: edges touching any one drawer
const THREAD_GROUP_MAX: usize = 50; // same_thread selectivity: skip huge rooms
const CONF_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Default)]
pub struct AutoTunnelReport {
    pub drawers: u64,
    pub candidates: u64,
    pub accepted: u64,
    pub coentity: u64,
    pub similar: u64,
    pub same_thread: u64,
    pub made_live: u64,
    pub reused: u64,
    pub tombstoned: u64,
    pub writes: u64,
    pub drawers_capped: u64,
}

impl std::fmt::Display for AutoTunnelReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "auto-tunnels: drawers={} candidates={} accepted={} \
             (coentity={} similar={} same_thread={}) made_live={} reused={} \
             tombstoned={} writes={} capped_drawers={}",
            self.drawers,
            self.candidates,
            self.accepted,
            self.coentity,
            self.similar,
            self.same_thread,
            self.made_live,
            self.reused,
            self.tombstoned,
            self.writes,
            self.drawers_capped
        )
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Coentity,
    Similar,
    SameThread,
}

#[derive(Clone)]
struct Candidate {
    from: String,
    to: String,
    relation: String,
    confidence: f32,
    priority: f32,
    kind: Kind,
}

fn stor(e: rusqlite::Error) -> PalaceError {
    PalaceError::Storage(e.to_string())
}

impl PalaceStore {
    /// Regenerate all `auto:` tunnels from the current KG + vectors + metadata.
    /// Deterministic and idempotent; manual edges are untouched. Returns a
    /// report of what was created, reused, and tombstoned.
    pub fn generate_auto_tunnels(&self) -> Result<AutoTunnelReport> {
        let mut guard = self.conn.lock().unwrap();

        // ── READ PHASE (immutable; everything collected into owned data) ──
        let live: Vec<(String, String, String)> = {
            let conn: &Connection = &guard;
            read_live_drawers(conn)?
        };
        let live_ids: HashSet<String> = live.iter().map(|(id, _, _)| id.clone()).collect();

        let mut candidates: Vec<Candidate> = Vec::new();
        {
            let conn: &Connection = &guard;
            build_coentity(conn, &live_ids, &mut candidates)?;
            build_similar(conn, &live_ids, &mut candidates)?;
        }
        build_same_thread(&live, &mut candidates);

        let existing_live_auto: Vec<(String, String, String, String)> = {
            let conn: &Connection = &guard;
            read_existing_live_auto(conn)?
        };

        let mut rep = AutoTunnelReport {
            drawers: live.len() as u64,
            candidates: candidates.len() as u64,
            ..Default::default()
        };

        // ── FAN-OUT CAP: deterministic greedy degree-bound ──
        candidates.sort_by(|a, b| {
            b.priority
                .total_cmp(&a.priority)
                .then_with(|| a.from.cmp(&b.from))
                .then_with(|| a.to.cmp(&b.to))
                .then_with(|| a.relation.cmp(&b.relation))
        });
        let mut degree: HashMap<String, usize> = HashMap::new();
        let mut capped: HashSet<String> = HashSet::new();
        let mut accepted: Vec<Candidate> = Vec::new();
        for c in &candidates {
            let df = *degree.get(&c.from).unwrap_or(&0);
            let dt = *degree.get(&c.to).unwrap_or(&0);
            if df < FANOUT_CAP && dt < FANOUT_CAP {
                *degree.entry(c.from.clone()).or_insert(0) += 1;
                *degree.entry(c.to.clone()).or_insert(0) += 1;
                match c.kind {
                    Kind::Coentity => rep.coentity += 1,
                    Kind::Similar => rep.similar += 1,
                    Kind::SameThread => rep.same_thread += 1,
                }
                accepted.push(c.clone());
            } else {
                if df >= FANOUT_CAP {
                    capped.insert(c.from.clone());
                }
                if dt >= FANOUT_CAP {
                    capped.insert(c.to.clone());
                }
            }
        }
        rep.accepted = accepted.len() as u64;
        rep.drawers_capped = capped.len() as u64;

        let target: HashSet<(String, String, String)> = accepted
            .iter()
            .map(|c| (c.from.clone(), c.to.clone(), c.relation.clone()))
            .collect();

        // ── WRITE PHASE: one transaction; reconcile toward `target` ──
        let tx = guard
            .transaction()
            .map_err(|e| PalaceError::Storage(format!("generate_auto_tunnels tx begin: {e}")))?;

        // Create / reuse every accepted edge (OR-set: reuse the existing row's
        // id for this triple key so re-runs update in place, never duplicate).
        for c in &accepted {
            let existing = sync_impl::query_tunnel(&tx, &c.from, &c.to, &c.relation)?;
            match existing {
                // Already live and unchanged -> true no-op (no HLC advance).
                Some(rec)
                    if !rec.tombstoned && (rec.confidence - c.confidence).abs() < CONF_EPS =>
                {
                    rep.reused += 1;
                }
                Some(rec) => {
                    let revived = rec.tombstoned;
                    let record = TunnelRecord {
                        id: rec.id,
                        from_drawer_id: c.from.clone(),
                        to_drawer_id: c.to.clone(),
                        relation: c.relation.clone(),
                        confidence: c.confidence,
                        hlc: sync_impl::hlc_now(),
                        tombstoned: false,
                    };
                    sync_impl::write_tunnel(&tx, &record)?;
                    rep.writes += 1;
                    if revived {
                        rep.made_live += 1;
                    } else {
                        rep.reused += 1; // confidence refreshed in place
                    }
                }
                None => {
                    let record = TunnelRecord {
                        id: Uuid::now_v7().to_string(),
                        from_drawer_id: c.from.clone(),
                        to_drawer_id: c.to.clone(),
                        relation: c.relation.clone(),
                        confidence: c.confidence,
                        hlc: sync_impl::hlc_now(),
                        tombstoned: false,
                    };
                    sync_impl::write_tunnel(&tx, &record)?;
                    rep.writes += 1;
                    rep.made_live += 1;
                }
            }
        }

        // Tombstone any previously-live auto: edge no longer derivable.
        for (id, from, to, relation) in &existing_live_auto {
            if !target.contains(&(from.clone(), to.clone(), relation.clone())) {
                let hlc = sync_impl::hlc_now();
                tx.execute(
                    "UPDATE tunnels SET tombstoned = 1, \
                     hlc_wall = ?1, hlc_logical = ?2, hlc_node_id = ?3 WHERE id = ?4",
                    params![hlc.wall_time as i64, hlc.logical as i64, hlc.node_id.to_vec(), id],
                )
                .map_err(stor)?;
                rep.tombstoned += 1;
                rep.writes += 1;
            }
        }

        tx.commit()
            .map_err(|e| PalaceError::Storage(format!("generate_auto_tunnels tx commit: {e}")))?;

        Ok(rep)
    }
}

// ── read helpers ─────────────────────────────────────────────────────────────

fn read_live_drawers(conn: &Connection) -> Result<Vec<(String, String, String)>> {
    let mut st = conn
        .prepare("SELECT id, wing, room FROM drawers WHERE state = 'live'")
        .map_err(stor)?;
    let rows = st
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .map_err(stor)?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(stor)?);
    }
    Ok(out)
}

/// Source 1: entity co-occurrence via the KG. An entity's drawers are the
/// distinct `source_drawer_id`s of triples referencing it (subject or object) —
/// the co-occurrence the graph actually knows. Selectivity guard: 2..=K drawers.
fn build_coentity(
    conn: &Connection,
    live: &HashSet<String>,
    out: &mut Vec<Candidate>,
) -> Result<()> {
    let names: HashMap<String, String> = {
        let mut st = conn
            .prepare("SELECT id, canonical_name FROM entities")
            .map_err(stor)?;
        let rows = st
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(stor)?;
        let mut m = HashMap::new();
        for r in rows {
            let (i, n) = r.map_err(stor)?;
            m.insert(i, n);
        }
        m
    };

    let mut ent_drawers: HashMap<String, BTreeSet<String>> = HashMap::new();
    {
        let mut st = conn
            .prepare(
                "SELECT subject_id, object_id, source_drawer_id \
                 FROM triples WHERE source_drawer_id IS NOT NULL",
            )
            .map_err(stor)?;
        let rows = st
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(stor)?;
        for r in rows {
            let (s, o, d) = r.map_err(stor)?;
            if !live.contains(&d) {
                continue;
            }
            ent_drawers.entry(s).or_default().insert(d.clone());
            ent_drawers.entry(o).or_default().insert(d);
        }
    }

    let mut ents: Vec<&String> = ent_drawers.keys().collect();
    ents.sort();
    for eid in ents {
        let drawers = &ent_drawers[eid];
        let freq = drawers.len();
        if !(ENTITY_FREQ_MIN..=ENTITY_FREQ_MAX).contains(&freq) {
            continue;
        }
        let name = names.get(eid).cloned().unwrap_or_else(|| eid.clone());
        let relation = format!("{AUTO_PREFIX}coentity:{name}");
        // Rarer shared entity = more selective = higher priority.
        let priority = 2.0 + (1.0 - (freq as f32) / (ENTITY_FREQ_MAX as f32));
        let ds: Vec<&String> = drawers.iter().collect(); // BTreeSet -> sorted
        for i in 0..ds.len() {
            for j in (i + 1)..ds.len() {
                out.push(Candidate {
                    from: ds[i].clone(),
                    to: ds[j].clone(),
                    relation: relation.clone(),
                    confidence: 1.0,
                    priority,
                    kind: Kind::Coentity,
                });
            }
        }
    }
    Ok(())
}

/// Source 3: semantic similarity. One vector per drawer (first chunk), top-N
/// nearest neighbours above cosine tau. Directed d->neighbour; retrieval is
/// bidirectional so a single direction suffices.
fn build_similar(conn: &Connection, live: &HashSet<String>, out: &mut Vec<Candidate>) -> Result<()> {
    let mut vecs: Vec<(String, Vec<f32>)> = Vec::new();
    {
        let mut st = conn
            .prepare(
                "SELECT drawer_id, embedding FROM chunks \
                 WHERE embedding IS NOT NULL ORDER BY drawer_id, chunk_index",
            )
            .map_err(stor)?;
        let rows = st
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .map_err(stor)?;
        let mut seen: HashSet<String> = HashSet::new();
        for r in rows {
            let (d, blob) = r.map_err(stor)?;
            if !live.contains(&d) || seen.contains(&d) {
                continue;
            }
            seen.insert(d.clone());
            vecs.push((d, crate::embedder::blob_to_vec(&blob)));
        }
    }
    vecs.sort_by(|a, b| a.0.cmp(&b.0));

    for i in 0..vecs.len() {
        let mut scored: Vec<(f32, &String)> = Vec::new();
        for j in 0..vecs.len() {
            if i == j {
                continue;
            }
            let s = crate::embedder::cosine_similarity(&vecs[i].1, &vecs[j].1);
            if s >= SIM_TAU {
                scored.push((s, &vecs[j].0));
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        for (s, dj) in scored.into_iter().take(SIM_TOP_N) {
            out.push(Candidate {
                from: vecs[i].0.clone(),
                to: dj.clone(),
                relation: format!("{AUTO_PREFIX}similar"),
                confidence: s,
                priority: 1.0 + s,
                kind: Kind::Similar,
            });
        }
    }
    Ok(())
}

/// Source 2 (structural, bounded): drawers sharing a (wing, room) thread.
/// Least selective, so lowest priority — the fan-out cap trims big threads.
fn build_same_thread(live: &[(String, String, String)], out: &mut Vec<Candidate>) {
    let mut groups: HashMap<(String, String), Vec<String>> = HashMap::new();
    for (id, wing, room) in live {
        groups
            .entry((wing.clone(), room.clone()))
            .or_default()
            .push(id.clone());
    }
    let mut keys: Vec<&(String, String)> = groups.keys().collect();
    keys.sort();
    for k in keys {
        let mut ds = groups[k].clone();
        ds.sort();
        let n = ds.len();
        if !(2..=THREAD_GROUP_MAX).contains(&n) {
            continue;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                out.push(Candidate {
                    from: ds[i].clone(),
                    to: ds[j].clone(),
                    relation: format!("{AUTO_PREFIX}same_thread"),
                    confidence: 1.0,
                    priority: 0.1,
                    kind: Kind::SameThread,
                });
            }
        }
    }
}

fn read_existing_live_auto(
    conn: &Connection,
) -> Result<Vec<(String, String, String, String)>> {
    let has: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tunnels'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    if has == 0 {
        return Ok(vec![]);
    }
    let mut st = conn
        .prepare(
            "SELECT id, from_drawer_id, to_drawer_id, relation FROM tunnels \
             WHERE tombstoned = 0 AND relation LIKE 'auto:%'",
        )
        .map_err(stor)?;
    let rows = st
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .map_err(stor)?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(stor)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PalaceStore;
    use tempfile::TempDir;

    const KEY: &str = "autotunnel-test-fixed-key";

    fn store() -> (PalaceStore, TempDir) {
        let tmp = TempDir::new().unwrap();
        let s = PalaceStore::open_in_memory(tmp.path(), KEY).unwrap();
        (s, tmp)
    }

    fn drawer(s: &PalaceStore, wing: &str, room: &str, content: &str) -> String {
        s.insert(wing, room, content, "test").unwrap().id
    }

    /// Record that entity `name` is mentioned in drawer `did`. Uses a unique
    /// per-mention `marker` object so the fact-key is distinct and the triple's
    /// `source_drawer_id` = `did` (the co-occurrence signal the generator reads).
    fn mention(s: &PalaceStore, name: &str, did: &str, marker: &str) {
        let e = s.kg_upsert_entity(name, None, None).unwrap();
        let m = s.kg_upsert_entity(marker, None, None).unwrap();
        s.kg_add_triple_by_ids(&e, "mentioned_in", &m, Some(did))
            .unwrap();
    }

    fn set_embedding(s: &PalaceStore, did: &str, v: &[f32]) {
        let blob = crate::embedder::vec_to_blob(v);
        let g = s.conn.lock().unwrap();
        g.execute(
            "UPDATE chunks SET embedding = ?1 WHERE drawer_id = ?2",
            params![blob, did],
        )
        .unwrap();
    }

    fn live_auto(s: &PalaceStore) -> Vec<(String, String, String)> {
        let g = s.conn.lock().unwrap();
        let mut st = g
            .prepare(
                "SELECT from_drawer_id, to_drawer_id, relation FROM tunnels \
                 WHERE tombstoned = 0 AND relation LIKE 'auto:%' ORDER BY 1,2,3",
            )
            .unwrap();
        let rows = st
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .unwrap();
        rows.map(|r| r.unwrap()).collect()
    }

    fn norm(v: &[f32]) -> Vec<f32> {
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// The load-bearing selectivity guard: an entity links its drawers only when
    /// 2 <= drawer-freq <= K(20). Freq 1 and freq > K produce no coentity edges.
    #[test]
    fn coentity_respects_selectivity_guard() {
        let (s, _t) = store();
        let d: Vec<String> = (0..21)
            .map(|i| drawer(&s, "w", "r", &format!("c{i}")))
            .collect();
        // Shared in 3 drawers -> valid (freq 3) -> C(3,2)=3 edges.
        for (i, di) in d.iter().enumerate().take(3) {
            mention(&s, "Shared", di, &format!("mk-shared-{i}"));
        }
        // Lonely in 1 drawer -> freq 1 -> no edge.
        mention(&s, "Lonely", &d[0], "mk-lonely");
        // Big in all 21 drawers -> freq 21 > K -> no edge.
        for (i, di) in d.iter().enumerate() {
            mention(&s, "Big", di, &format!("mk-big-{i}"));
        }

        let rep = s.generate_auto_tunnels().unwrap();
        let edges = live_auto(&s);

        let co: Vec<_> = edges
            .iter()
            .filter(|(_, _, r)| r.starts_with("auto:coentity:"))
            .collect();
        assert_eq!(co.len(), 3, "only Shared(freq3) yields coentity edges; got {co:?}");
        assert!(
            co.iter().all(|(_, _, r)| r == "auto:coentity:Shared"),
            "no Big/Lonely edges: {co:?}"
        );
        assert_eq!(rep.coentity, 3);
        assert!(
            edges.iter().all(|(_, _, r)| r.starts_with("auto:")),
            "every derived edge is auto:-namespaced"
        );
    }

    /// Re-running over unchanged data is a write-layer no-op: nothing created,
    /// nothing tombstoned, byte-identical edge set. (The idempotency contract.)
    #[test]
    fn regenerate_is_idempotent() {
        let (s, _t) = store();
        let a = drawer(&s, "w", "r", "a");
        let b = drawer(&s, "w", "r", "b");
        let c = drawer(&s, "w", "r", "c");
        for (i, did) in [&a, &b, &c].iter().enumerate() {
            mention(&s, "E", did, &format!("mk{i}"));
        }

        let _r1 = s.generate_auto_tunnels().unwrap();
        let e1 = live_auto(&s);
        let r2 = s.generate_auto_tunnels().unwrap();
        let e2 = live_auto(&s);

        assert_eq!(e1, e2, "edge set stable across runs");
        assert_eq!(r2.made_live, 0, "second run creates nothing new");
        assert_eq!(r2.tombstoned, 0, "second run tombstones nothing");
        assert_eq!(r2.writes, 0, "second run is a no-op at the write layer");
    }

    /// Manual edges (no auto: prefix) are invariant: the generator never reads,
    /// rewrites, or tombstones them.
    #[test]
    fn manual_tunnels_are_invariant() {
        let (s, _t) = store();
        let a = drawer(&s, "w", "r", "a");
        let b = drawer(&s, "w", "r", "b");
        let c = drawer(&s, "w", "r", "c");
        let man = s.create_tunnel(&a, &b, "relates_to", None).unwrap();
        for (i, did) in [&a, &b, &c].iter().enumerate() {
            mention(&s, "E", did, &format!("mk{i}"));
        }

        s.generate_auto_tunnels().unwrap();

        let g = s.conn.lock().unwrap();
        let (rel, tomb): (String, i64) = g
            .query_row(
                "SELECT relation, tombstoned FROM tunnels WHERE id = ?1",
                params![man.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(rel, "relates_to", "manual relation unchanged");
        assert_eq!(tomb, 0, "manual edge never tombstoned by the generator");
        assert!(!rel.starts_with("auto:"), "manual edge never auto:-prefixed");
    }

    /// Re-run after the underlying data changes: an edge whose co-occurrence no
    /// longer holds is tombstoned; the graph converges to current truth.
    #[test]
    fn rerun_reconciles_after_data_change() {
        let (s, _t) = store();
        let a = drawer(&s, "w", "r", "a");
        let b = drawer(&s, "w", "r", "b");
        mention(&s, "E", &a, "mka");
        mention(&s, "E", &b, "mkb");

        s.generate_auto_tunnels().unwrap();
        assert!(
            live_auto(&s).iter().any(|(_, _, r)| r == "auto:coentity:E"),
            "coentity edge present while both drawers live"
        );

        // Remove the co-occurrence: tombstone drawer b. Now E is in 1 live
        // drawer (freq 1) -> below the guard -> the edge must retire.
        s.tombstone(&b).unwrap();
        let r = s.generate_auto_tunnels().unwrap();

        assert!(r.tombstoned >= 1, "stale edge tombstoned; report={r}");
        assert!(
            !live_auto(&s).iter().any(|(_, _, r)| r == "auto:coentity:E"),
            "coentity edge gone after b tombstoned"
        );
    }

    /// No drawer carries more than FANOUT_CAP live auto: edges, even when the
    /// data would derive more. 40 distinct shared entities -> hub saturates at 30.
    #[test]
    fn fanout_cap_bounds_degree() {
        let (s, _t) = store();
        let hub = drawer(&s, "w", "hub", "hub");
        for i in 0..40 {
            // distinct room per partner so same_thread never fires here
            let p = drawer(&s, "w", &format!("p{i}"), &format!("p{i}"));
            mention(&s, &format!("Ent{i}"), &hub, &format!("mkh{i}"));
            mention(&s, &format!("Ent{i}"), &p, &format!("mkp{i}"));
        }

        let rep = s.generate_auto_tunnels().unwrap();
        let edges = live_auto(&s);
        let deg = edges
            .iter()
            .filter(|(f, t, _)| f == &hub || t == &hub)
            .count();

        assert!(deg <= FANOUT_CAP, "hub degree {deg} exceeds cap");
        assert_eq!(deg, FANOUT_CAP, "hub saturates the cap");
        assert!(rep.drawers_capped >= 1, "hub reported as capped");
    }

    /// Similarity keeps exactly the top-N nearest neighbours above tau; a vector
    /// below tau is never linked.
    #[test]
    fn similar_uses_cosine_topn_and_tau() {
        let (s, _t) = store();
        let base = drawer(&s, "w", "r", "base");
        set_embedding(&s, &base, &norm(&[1.0, 0.0, 0.0]));
        for i in 0..6 {
            let d = drawer(&s, "w", &format!("n{i}"), "near");
            set_embedding(&s, &d, &norm(&[1.0, 0.01 * (i as f32 + 1.0), 0.0]));
        }
        let orth = drawer(&s, "w", "o", "orth");
        set_embedding(&s, &orth, &norm(&[0.0, 1.0, 0.0])); // cosine 0 < tau

        s.generate_auto_tunnels().unwrap();
        let sim_from_base: Vec<_> = live_auto(&s)
            .into_iter()
            .filter(|(f, _, r)| f == &base && r == "auto:similar")
            .collect();

        assert_eq!(
            sim_from_base.len(),
            SIM_TOP_N,
            "base keeps exactly top-N similar; got {sim_from_base:?}"
        );
        assert!(
            !sim_from_base.iter().any(|(_, t, _)| t == &orth),
            "below-tau drawer never linked"
        );
    }
}
