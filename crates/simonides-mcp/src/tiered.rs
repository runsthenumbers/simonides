//! Tier-aware store access for the MCP surface.
//!
//! After the membrane split the palace is three separate sqlcipher files, one
//! per tier:
//!
//! | tier | file | key | served on the network? |
//! |---|---|---|---|
//! | L1 open | `palace.db` | machine-bound | **yes — this and only this** |
//! | L2 private | `palace_thread.db` | machine-bound, resident | no |
//! | L3 vault | `palace_private.db` | passphrase + recovery | no |
//!
//! # The property this module exists to preserve
//!
//! > **A locked tier is ABSENT, not filtered.**
//!
//! `Tiered` holds an `Option<Arc<PalaceStore>>` for L3. Locked means `None` —
//! the file is not open, no connection exists, and no query can reach it. Every
//! enumeration here iterates `readable()`, which simply does not yield a store
//! that is locked. There is no predicate to invert and no branch to forget: a
//! locked drawer is indistinguishable from one that never existed, because from
//! this process's point of view it doesn't.
//!
//! That is why this is a fan-out over stores rather than a filter over rows. A
//! filter would have to be remembered at every call site, forever, including
//! ones written next year by someone who has not read this comment.
//!
//! # Why L2 is not gated
//!
//! L2 holds an agent's own continuity - diaries, letters, session state. It is
//! never network-served, but it opens under a resident key so an agent can reach
//! its own history with nobody at the keyboard. Confidentiality against the
//! network is a property of **what the serving layer publishes**, not of which
//! key encrypts a file.
//!
//! # Write routing
//!
//! Writes go to the file the tier policy names for `(wing, room)`, and every store
//! is given that same policy, so the tier a drawer is routed to and the tier
//! stamped on it always agree. A
//! write aimed at a locked tier is refused — never silently redirected to an
//! open one, which would be a downgrade nobody asked for.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use simonides::membrane;
use std::collections::{HashSet, VecDeque};
use simonides::{Drawer, PalaceError, PalaceStore, ResolveOutcome};

/// Mirrors simonides's private `MAX_TRAVERSAL_DEPTH` (64): the cross-tier
/// walk clamps hops to the same ceiling the store's own traverse uses.
const MAX_CROSS_TIER_DEPTH: usize = 64;

/// Result of the cross-tier walk: (reached drawers in discovery order, absent
/// endpoint ids, tiers consulted). Named to keep the signature legible.
pub type CrossTierWalk = (Vec<Drawer>, Vec<String>, Vec<&'static str>);

/// Whether a configured tier is currently open.
///
/// Deliberately two-valued. There is no `Unknown` and no `Empty`: a tier is
/// either open (so it can be measured honestly) or locked (so it must be
/// reported as unmeasured, never as zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierState {
    Open,
    Locked,
}

impl TierState {
    fn from_slot(has_store: bool) -> Self {
        if has_store {
            Self::Open
        } else {
            Self::Locked
        }
    }

    /// Lowercase wire form for status payloads.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Locked => "locked",
        }
    }
}

/// One tier's file, key and (once opened) store.
struct TierSlot {
    tier: &'static str,
    db: PathBuf,
    vault: PathBuf,
    store: Option<Arc<PalaceStore>>,
}

pub struct Tiered {
    /// L1 and L2, always open. Ordered most-open first so `readable()` yields a
    /// deterministic order and point lookups resolve predictably.
    resident: Vec<TierSlot>,
    /// L3. `None` = locked. Behind a lock so unlocking does not need `&mut self`.
    vault_tier: RwLock<TierSlot>,
    /// Decides which tier every write belongs to. Also handed to each store, so the
    /// routing decision and the tag stamped on the drawer can never disagree.
    policy: membrane::TierPolicy,
}

impl Tiered {
    /// Open the resident tiers. L3 is registered but NOT opened — it starts
    /// locked, always, regardless of what is on disk.
    ///
    /// Eight arguments is one over clippy's threshold and it stays that way
    /// deliberately. Each tier contributes db + vault (+ key, where a key is
    /// held), so the arity IS the membrane: three tiers, named individually, no
    /// collection to iterate and therefore no way to add or drop a tier by
    /// accident. Bundling them into a config struct would hide exactly the
    /// thing a reader of this constructor most needs to see — that L3 is passed
    /// WITHOUT a key.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        open_db: &Path,
        open_vault: &Path,
        open_key: &str,
        thread_db: &Path,
        thread_vault: &Path,
        thread_key: &str,
        priv_db: &Path,
        priv_vault: &Path,
        policy: membrane::TierPolicy,
    ) -> Result<Self, String> {
        let l1 = PalaceStore::open(open_db, open_vault, open_key)
            .map_err(|e| format!("open L1 {}: {e}", open_db.display()))?;
        let l2 = PalaceStore::open(thread_db, thread_vault, thread_key)
            .map_err(|e| format!("open L2 {}: {e}", thread_db.display()))?;

        // Distinct keys are load-bearing, not hygiene: two tiers under one key
        // means anything that opens one opens the other, and the separation is
        // cosmetic. Cheap to check, catastrophic to miss.
        if open_key == thread_key {
            return Err("L1 and L2 were given the SAME key — that is not a membrane".into());
        }

        l1.set_tier_policy(policy.clone());
        l2.set_tier_policy(policy.clone());

        Ok(Self {
            resident: vec![
                TierSlot { tier: "L1", db: open_db.into(), vault: open_vault.into(), store: Some(Arc::new(l1)) },
                TierSlot { tier: "L2", db: thread_db.into(), vault: thread_vault.into(), store: Some(Arc::new(l2)) },
            ],
            vault_tier: RwLock::new(TierSlot {
                tier: "L3",
                db: priv_db.into(),
                vault: priv_vault.into(),
                store: None,
            }),
            policy,
        })
    }

    /// Every store readable right now, most-open tier first.
    ///
    /// A locked tier contributes nothing — not an empty result, not an error:
    /// it is simply not in the list. Callers cannot tell a locked tier from an
    /// absent one, and that is the point.
    pub fn readable(&self) -> Vec<(&'static str, Arc<PalaceStore>)> {
        let mut out: Vec<(&'static str, Arc<PalaceStore>)> = self
            .resident
            .iter()
            .filter_map(|s| s.store.clone().map(|st| (s.tier, st)))
            .collect();
        if let Ok(g) = self.vault_tier.read() {
            if let Some(st) = g.store.clone() {
                out.push((g.tier, st));
            }
        }
        out
    }

    /// Every CONFIGURED tier and whether it is open, most-open first.
    ///
    /// This is the deliberate counterpart to `readable()`, and the two must not
    /// be confused. `readable()` answers "what can I touch" and a locked tier is
    /// ABSENT from it — which means a caller holding only `readable()` cannot
    /// count what it skipped, because the skipped ones are not there to count.
    /// That is correct for access and useless for reporting: it is exactly how a
    /// status line ends up saying "0 pending" when the honest answer is "I could
    /// not look."
    ///
    /// So: use `readable()` to decide what to OPERATE on, and this to decide what
    /// to SAY. Never derive one from the other by reaching around the plumbing.
    ///
    /// PROHIBITION — this returns lock state and NOTHING else. Do not extend it
    /// with counts, sizes, drawer totals, timestamps or any other per-tier
    /// measurement for a locked tier. That a locked tier EXISTS is already public
    /// (`membrane_status`, `is_unlocked`) and reveals no drawer. Any *quantity*
    /// about a locked tier is a different claim entirely, and leaking one breaks
    /// structural absence: the guarantee that a session which has not unlocked L3
    /// cannot distinguish an L3 drawer from one that never existed.
    pub fn tier_states(&self) -> Vec<(&'static str, TierState)> {
        let mut out: Vec<(&'static str, TierState)> = self
            .resident
            .iter()
            .map(|s| (s.tier, TierState::from_slot(s.store.is_some())))
            .collect();
        // Read the lock once. A tier that fails to read its own lock is reported
        // Locked, never Open — a poisoned lock must not read as available.
        let vault = match self.vault_tier.read() {
            Ok(g) => (g.tier, TierState::from_slot(g.store.is_some())),
            Err(p) => (p.into_inner().tier, TierState::Locked),
        };
        out.push(vault);
        out
    }

    /// Is the L3 vault open? The ONLY thing that reveals lock state.
    pub fn is_unlocked(&self) -> bool {
        self.vault_tier.read().map(|g| g.store.is_some()).unwrap_or(false)
    }

    /// Unlock L3 with an already-derived raw Data Key.
    ///
    /// The key arrives as bytes because deriving it is somebody else's job: the
    /// passphrase must never travel the MCP transport, since that path can run
    /// through a remote connector and would carry the secret off the machine —
    /// the exact boundary the membrane exists to stop things crossing.
    pub fn unlock_with_dk(&self, dk: &[u8; 32]) -> Result<(), String> {
        let literal = membrane::raw_key_literal(dk);
        let mut g = self.vault_tier.write().map_err(|_| "membrane lock poisoned".to_string())?;
        if g.store.is_some() {
            return Ok(()); // already open; idempotent
        }
        let st = PalaceStore::open(&g.db, &g.vault, literal.as_str())
            .map_err(|e| format!("unlock L3: {e}"))?;
        st.set_tier_policy(self.policy.clone());
        g.store = Some(Arc::new(st));
        Ok(())
    }

    /// Close L3. Dropping the `Arc` closes the sqlcipher connection, so the
    /// rows become absent again rather than merely disallowed.
    pub fn lock(&self) -> Result<(), String> {
        let mut g = self.vault_tier.write().map_err(|_| "membrane lock poisoned".to_string())?;
        g.store = None;
        Ok(())
    }

    /// The tier policy this layer routes by.
    pub fn tier_policy(&self) -> &membrane::TierPolicy {
        &self.policy
    }

    /// The store a write to `(wing, room)` belongs in.
    ///
    /// Refuses rather than redirecting. Silently writing a vault-bound drawer
    /// into an open file because the vault happened to be locked is precisely
    /// the failure this whole design is built to make impossible.
    pub fn writable_for(&self, wing: &str, room: &str) -> Result<Arc<PalaceStore>, String> {
        let t = self.policy.tier(wing, room);
        for (tier, st) in self.readable() {
            if tier == t {
                return Ok(st);
            }
        }
        Err(format!(
            "{wing}/{room} belongs to tier {t}, which is locked, so nothing was saved. \
             This version of Simonides cannot unlock that tier, and the note will not be redirected to another tier."
        ))
    }

    /// Which readable store holds this drawer, if any.
    ///
    /// `None` covers both "no such drawer" and "it lives in a locked tier", and
    /// the caller cannot distinguish them. Deliberate: a NotFound that leaks
    /// "exists but locked" discloses the drawer.
    pub fn owner_of(&self, id: &str) -> Result<Option<(Arc<PalaceStore>, Drawer)>, String> {
        for (_, st) in self.readable() {
            match st.get(id).map_err(|e| e.to_string())? {
                Some(d) => return Ok(Some((st, d))),
                None => continue,
            }
        }
        Ok(None)
    }

    /// Fan out a read over every readable tier and concatenate.
    pub fn fan_out<T, F>(&self, mut f: F) -> Result<Vec<T>, String>
    where
        F: FnMut(&PalaceStore) -> Result<Vec<T>, String>,
    {
        let mut out = Vec::new();
        for (_, st) in self.readable() {
            out.extend(f(&st)?);
        }
        Ok(out)
    }

    /// Fan a per-drawer GRAPH op across readable tiers, TOLERATING a tier that
    /// does not hold the drawer. That tier returns NotFound from resolve_endpoint;
    /// it is not fatal -- it contributes nothing and the tier that DOES hold the
    /// drawer still answers. A fan that aborts on the first tier's NotFound is a
    /// correctness bug even with zero cross-tier edges. Returns the union plus the
    /// tiers consulted, so the caller can state its view is scoped to the tiers it
    /// could read -- never anything about the ones it could not. Non-NotFound
    /// errors still propagate.
    pub fn fan_graph<T, F>(&self, mut f: F) -> Result<(Vec<T>, Vec<&'static str>), String>
    where
        F: FnMut(&PalaceStore) -> Result<Vec<T>, PalaceError>,
    {
        let mut out = Vec::new();
        let mut consulted = Vec::new();
        for (name, st) in self.readable() {
            consulted.push(name);
            match f(&st) {
                Ok(v) => out.extend(v),
                Err(PalaceError::NotFound { .. }) => {}
                Err(e) => return Err(e.to_string()),
            }
        }
        Ok((out, consulted))
    }

    /// Cross-tier breadth-first walk over the WHOLE readable graph.
    ///
    /// Fanning the store's own `traverse` per tier is not enough: each store
    /// sees only edges resident in it and resolves only neighbours resident in
    /// it, so a hop whose edge lives in one tier and whose endpoints live in
    /// another is never followed. This walks all readable tiers as one graph —
    /// a node's edges are gathered from EVERY readable store by exact id (an
    /// edge is stored in its most-restrictive endpoint's tier, which need not be
    /// the node's own), and each neighbour is resolved across tiers via
    /// `owner_of`.
    ///
    /// A neighbour that resolves in no readable tier is neither dropped nor
    /// fatal: it is reported, by id, in the ABSENT set. Absence is uniform — a
    /// removed/not-yet-synced endpoint and one sitting in a tier this session
    /// cannot read are indistinguishable, so the walk never discloses an
    /// unreadable drawer, and empty != absent != could-not-look stays honest. A
    /// start reference that resolves nowhere returns no drawers and lists itself
    /// absent. Bounded by `max_depth` (clamped) plus a `visited` set for cycles.
    /// Returns (reached drawers in discovery order, absent ids, tiers consulted).
    pub fn traverse_cross_tier(
        &self,
        start_ref: &str,
        max_depth: usize,
    ) -> Result<CrossTierWalk, String> {
        let consulted: Vec<&'static str> = self.readable().iter().map(|(n, _)| *n).collect();

        let start_id = match self.resolve_ref(start_ref)? {
            Some(id) => id,
            None => return Ok((Vec::new(), vec![start_ref.to_string()], consulted)),
        };

        let cap = max_depth.min(MAX_CROSS_TIER_DEPTH);
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_id.clone());
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((start_id, 0));
        let mut reached: Vec<Drawer> = Vec::new();
        let mut absent: Vec<String> = Vec::new();

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= cap {
                continue;
            }
            // An edge lives in its most-restrictive endpoint's tier, not
            // necessarily the node's — so gather from every readable store, by
            // EXACT id. A resolving lookup would error in the very store that
            // holds the edge but not the node's drawer.
            let mut neighbours: Vec<String> = Vec::new();
            for (_, st) in self.readable() {
                for edge in st.tunnels_referencing(&node).map_err(|e| e.to_string())? {
                    neighbours.push(if edge.from_drawer_id == node {
                        edge.to_drawer_id
                    } else {
                        edge.from_drawer_id
                    });
                }
            }
            for nb in neighbours {
                if !visited.insert(nb.clone()) {
                    continue;
                }
                match self.owner_of(&nb)? {
                    Some((_, drawer)) => {
                        reached.push(drawer);
                        queue.push_back((nb, depth + 1));
                    }
                    None => absent.push(nb),
                }
            }
        }
        Ok((reached, absent, consulted))
    }

    /// Resolve a drawer reference to a concrete id across readable tiers,
    /// prefix-tolerant, first match wins. `None` = resolvable in no readable
    /// tier (absent, or resident only in a tier this session cannot read).
    fn resolve_ref(&self, reference: &str) -> Result<Option<String>, String> {
        for (_, st) in self.readable() {
            if let ResolveOutcome::Found(d) = st.resolve(reference).map_err(|e| e.to_string())? {
                return Ok(Some(d.id));
            }
        }
        Ok(None)
    }

    /// Fan out and sum. For counts, where merging means addition.
    pub fn fan_sum<F>(&self, mut f: F) -> Result<usize, String>
    where
        F: FnMut(&PalaceStore) -> Result<usize, String>,
    {
        let mut n = 0usize;
        for (_, st) in self.readable() {
            n += f(&st)?;
        }
        Ok(n)
    }

    /// Fan out over string lists, dedup and sort. Wings and rooms exist in more
    /// than one tier — one agent's wing can hold work-log rows in L1 and diary rows in
    /// L2 — so the union is the honest answer and duplicates are an artifact of
    /// how the data is stored, not something the caller should see.
    pub fn fan_names<F>(&self, mut f: F) -> Result<Vec<String>, String>
    where
        F: FnMut(&PalaceStore) -> Result<Vec<String>, String>,
    {
        let mut v = self.fan_out(|s| f(s))?;
        v.sort();
        v.dedup();
        Ok(v)
    }

    // ---------------------------------------------------------------------
    // Merged reads. Each one states how it combines tiers, because "merge" is
    // doing different work in each case and getting it wrong is silent.
    // ---------------------------------------------------------------------

    /// Health, summed across readable tiers.
    ///
    /// Summing is right for counts, and the sum is over what is READABLE — so
    /// with the vault locked, `total` is the total of what this session can
    /// see. That is deliberate: a status line that reported the full corpus
    /// while the vault was closed would disclose how much is behind it.
    pub fn health_merged(&self) -> Result<simonides::HealthReport, String> {
        let mut acc: Option<simonides::HealthReport> = None;
        for (_, st) in self.readable() {
            let h = st.health().map_err(|e| e.to_string())?;
            acc = Some(match acc {
                None => h,
                Some(a) => simonides::HealthReport {
                    total: a.total + h.total,
                    live: a.live + h.live,
                    tombstoned: a.tombstoned + h.tombstoned,
                    reindex_pending: a.reindex_pending + h.reindex_pending,
                    kg_entities: a.kg_entities + h.kg_entities,
                    kg_triples: a.kg_triples + h.kg_triples,
                    closets: a.closets + h.closets,
                    closets_stale: a.closets_stale + h.closets_stale,
                    chunks_total: a.chunks_total + h.chunks_total,
                    chunks_embedded: a.chunks_embedded + h.chunks_embedded,
                    embedding_model_version: a.embedding_model_version,
                },
            });
        }
        acc.ok_or_else(|| "no readable tier".to_string())
    }

    /// Live count for a filter, summed across readable tiers.
    pub fn count_live(&self, wing: Option<&str>, room: Option<&str>) -> Result<usize, String> {
        self.fan_sum(|s| s.count_live(wing, room).map_err(|e| e.to_string()))
    }

    /// Paged listing across tiers, in global id order.
    ///
    /// Each store returns id-ASC, so taking `offset + limit` from every tier and
    /// re-sorting yields exactly the global first `offset + limit` — a standard
    /// merge-paginate. Taking only `limit` per tier would be wrong: a page could
    /// then miss rows that sort earlier in a tier we truncated.
    pub fn list_paged(
        &self,
        wing: Option<&str>,
        room: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Drawer>, String> {
        let need = offset.saturating_add(limit);
        let mut all = self.fan_out(|s| {
            s.list_paged(wing, room, need, 0).map_err(|e| e.to_string())
        })?;
        all.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(all.into_iter().skip(offset).take(limit).collect())
    }

    /// Search across readable tiers.
    ///
    /// **Ranking is per-tier, not global.** Each store computes its own hybrid
    /// BM25+vector score against its own index; those scores are not comparable
    /// across files, so interleaving them by score would invent a ranking that
    /// does not exist. Results are concatenated tier order (open first) and
    /// truncated. Stated here rather than discovered later, because a subtly
    /// wrong ranking is the kind of bug nobody files.
    pub fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<simonides::retrieval::SearchHit>, String> {
        let mut out = Vec::new();
        for (_, st) in self.readable() {
            if out.len() >= limit {
                break;
            }
            if let simonides::retrieval::SearchResult::Hits(hits) =
                st.search(query, limit).map_err(|e| e.to_string())?
            {
                out.extend(hits);
            }
        }
        out.truncate(limit);
        Ok(out)
    }

    /// A status payload safe to return over the network: booleans and counts,
    /// never content, never a drawer id, never a hint about what is inside.
    pub fn membrane_status(&self) -> serde_json::Value {
        let readable: Vec<&str> = self.readable().iter().map(|(t, _)| *t).collect();
        serde_json::json!({
            "tiers_readable": readable,
            "vault_unlocked": self.is_unlocked(),
            "note": "L1 is the only tier a network-facing serving layer publishes. \
                     L2 opens under a resident machine key and is never network-served. L3 opens only on local unlock."
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().expect("tmp")
    }

    /// Neutral tier policy for the tests: one rule per routing case they exercise.
    fn test_policy() -> simonides::membrane::TierPolicy {
        simonides::membrane::TierPolicy::from_toml_str(
            r#"
[[rule]]
room = "vault"
tier = "L3"

[[rule]]
room = "personal-diary"
tier = "L2"

[[rule]]
wing = "private"
tier = "L3"

[[rule]]
wing_prefix = "public"
tier = "L1"

[[rule]]
wing_prefix = "agent_"
tier = "L2"
"#,
        )
        .expect("test policy parses")
    }
    fn build(d: &Path) -> Tiered {
        Tiered::open(
            &d.join("palace.db"), &d.join("v1"), "x'1111111111111111111111111111111111111111111111111111111111111111'",
            &d.join("palace_thread.db"), &d.join("v2"), "x'2222222222222222222222222222222222222222222222222222222222222222'",
            &d.join("palace_private.db"), &d.join("v3"), test_policy(),
        )
        .expect("open tiered")
    }

    #[test]
    fn l3_starts_locked_and_is_absent_not_filtered() {
        let d = tmp();
        let t = build(d.path());
        assert!(!t.is_unlocked(), "L3 must start locked regardless of what is on disk");
        let tiers: Vec<&str> = t.readable().iter().map(|(x, _)| *x).collect();
        assert_eq!(tiers, vec!["L1", "L2"], "a locked tier contributes NOTHING to readable()");
    }

    /// The pair that makes honest status possible.
    ///
    /// `readable()` DROPS a locked tier, so a caller holding only `readable()`
    /// cannot count what it skipped — which is how a status line ends up saying
    /// "0 pending" when the true answer is "I could not look." `tier_states()`
    /// keeps every configured tier so the difference is reportable.
    ///
    /// Asserting both in one test is deliberate: the guarantee is the CONTRAST,
    /// and a test of either one alone would pass while the pair was broken.
    #[test]
    fn tier_states_keeps_the_locked_tier_that_readable_drops() {
        let d = tmp();
        let t = build(d.path());

        let readable: Vec<&str> = t.readable().iter().map(|(x, _)| *x).collect();
        assert_eq!(readable, vec!["L1", "L2"], "readable() must drop the locked tier");

        let states = t.tier_states();
        assert_eq!(
            states,
            vec![("L1", TierState::Open), ("L2", TierState::Open), ("L3", TierState::Locked)],
            "tier_states() must keep every CONFIGURED tier so 'could not look' is sayable"
        );

        // The gap between the two IS the thing status must report.
        let unreachable: Vec<&str> = states
            .iter()
            .filter(|(_, s)| *s == TierState::Locked)
            .map(|(x, _)| *x)
            .collect();
        assert_eq!(unreachable, vec!["L3"]);
        assert_eq!(readable.len() + unreachable.len(), states.len());
    }

    #[test]
    fn tier_states_follows_the_lock_and_is_not_a_static_map() {
        let d = tmp();
        let t = build(d.path());
        assert_eq!(t.tier_states()[2], ("L3", TierState::Locked));

        t.unlock_with_dk(&[7u8; 32]).expect("unlock");
        assert_eq!(
            t.tier_states()[2],
            ("L3", TierState::Open),
            "must report live lock state, not the configuration it was built with"
        );

        t.lock().expect("relock");
        assert_eq!(t.tier_states()[2], ("L3", TierState::Locked), "and must go back");
    }

    #[test]
    fn writes_route_by_tier_and_a_locked_tier_refuses_rather_than_redirects() {
        let d = tmp();
        let t = build(d.path());

        // L1 and L2 are open, so these resolve.
        t.writable_for("public_docs", "work-log").expect("L1 write");
        t.writable_for("agent_ada", "personal-diary").expect("L2 write");

        // L3 is locked. The write must FAIL, not quietly land in an open tier.
        let e = match t.writable_for("private", "health") {
            Err(e) => e,
            Ok(_) => panic!("a locked tier must refuse the write, not resolve it"),
        };
        assert!(e.contains("L3"), "must name the tier: {e}");
        assert!(e.contains("not be redirected"), "must promise no redirect: {e}");

        // Same for an opt-in vault drawer from any wing.
        assert!(t.writable_for("agent_ada", "vault").is_err());
    }

    #[test]
    fn a_drawer_in_a_locked_tier_is_indistinguishable_from_one_that_does_not_exist() {
        let d = tmp();
        let t = build(d.path());
        // Nothing was ever written, and L3 is locked. Both cases must look the
        // same to a caller: None.
        assert!(t.owner_of("019f-nonexistent").expect("lookup").is_none());
    }

    #[test]
    fn status_reveals_lock_state_and_nothing_else() {
        let d = tmp();
        let t = build(d.path());
        let s = t.membrane_status();
        assert_eq!(s["vault_unlocked"], serde_json::json!(false));
        let text = s.to_string();
        // No paths, no keys, no ids. A status tool that leaks the vault's
        // filename has told a network caller where to aim.
        assert!(!text.contains("palace_private"), "status must not disclose paths: {text}");
        assert!(!text.contains('x'.to_string().repeat(8).as_str()), "no key material: {text}");
    }

    #[test]
    fn identical_keys_are_rejected_because_that_is_not_a_membrane() {
        let d = tmp();
        let same = "x'3333333333333333333333333333333333333333333333333333333333333333'";
        let r = Tiered::open(
            &d.path().join("a.db"), &d.path().join("va"), same,
            &d.path().join("b.db"), &d.path().join("vb"), same,
            &d.path().join("c.db"), &d.path().join("vc"), test_policy(),
        );
        let e = match r {
            Err(e) => e,
            Ok(_) => panic!("two tiers under one key must be rejected"),
        };
        assert!(e.contains("SAME key"), "must say why: {e}");
    }

    #[test]
    fn unlock_then_lock_round_trips_and_lock_makes_rows_absent_again() {
        let d = tmp();
        let t = build(d.path());
        let dk = [7u8; 32];

        t.unlock_with_dk(&dk).expect("unlock");
        assert!(t.is_unlocked());
        let tiers: Vec<&str> = t.readable().iter().map(|(x, _)| *x).collect();
        assert_eq!(tiers, vec!["L1", "L2", "L3"]);
        // Now writable.
        t.writable_for("private", "health").expect("L3 write while unlocked");

        // Idempotent.
        t.unlock_with_dk(&dk).expect("second unlock is a no-op");

        t.lock().expect("lock");
        assert!(!t.is_unlocked());
        assert_eq!(t.readable().len(), 2, "locking must remove the tier, not just flag it");
        assert!(t.writable_for("private", "health").is_err());
    }
}
