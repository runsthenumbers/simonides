//! MCP server for a simonides store.
//!
//! A thin mapping over the `PalaceStore` API: each tool wraps one store method.
//! The tool surface is modelled on MemPalace's, so agents that know that tool set
//! will find the same operations here. A few tools (get_aaak_spec, graph_stats,
//! get_taxonomy, memories_filed_away, hook_settings, reconnect, check_duplicate)
//! are helpers with no single store method behind them: they are composed from
//! real store calls, or are a no-op where there is no equivalent, and say so
//! in-line.

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use std::path::Path;
use simonides::retrieval::SearchResult;
use simonides_core::memory::KgDirection;
use simonides::{Drawer, HealthReport, PalaceStore, Tunnel};

pub mod config;
pub mod protocol;
pub mod tiered;
pub use tiered::{CrossTierWalk, Tiered};

pub struct McpServer {
    /// The tier the server writes to by default and the one every legacy
    /// single-store path still uses. After the membrane split this is L1.
    store: std::sync::Arc<PalaceStore>,
    /// Present once the palace is split. `None` = pre-split single store, in
    /// which case every tool behaves exactly as it did before.
    ///
    /// Kept as an `Option` rather than forcing the split everywhere so the
    /// server runs against either shape. The migration is a deployment step,
    /// not a rebuild.
    tiers: Option<std::sync::Arc<Tiered>>,
    rebuild: std::sync::Arc<std::sync::Mutex<RebuildState>>,
    /// Start a background rebuild after every write, so new drawers become
    /// searchable by meaning without anyone calling `rebuild`. Off by default
    /// (tests, embedding callers); the server binary turns it on.
    auto_index: std::sync::atomic::AtomicBool,
}

#[derive(Default)]
struct RebuildState {
    running: bool,
    /// Set when a rebuild is requested while one runs: the running one repeats.
    dirty: bool,
    phase: String,
    error: Option<String>,
}

/// The rebuild state, even if a panicking thread poisoned the lock.
fn lock_state(s: &std::sync::Mutex<RebuildState>) -> std::sync::MutexGuard<'_, RebuildState> {
    s.lock().unwrap_or_else(|p| p.into_inner())
}

impl McpServer {
    /// Pre-split: one store, no tiering. Unchanged behaviour.
    pub fn open(db_path: &Path, vault_root: &Path, key: &str) -> Result<Self, String> {
        let store = PalaceStore::open(db_path, vault_root, key).map_err(|e| e.to_string())?;
        // Tiers are off: label drawers `tier=single`, not the fail-safe `L3`,
        // which would read as "in the locked vault".
        store.set_tier_policy(simonides::membrane::TierPolicy::untiered());
        Ok(Self {
            store: std::sync::Arc::new(store),
            tiers: None,
            rebuild: std::sync::Arc::new(std::sync::Mutex::new(RebuildState::default())),
            auto_index: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Post-split: three files, membrane-aware.
    ///
    /// L3 always starts LOCKED, whatever is on disk — an unattended restart must
    /// never come up with the vault open.
    #[allow(clippy::too_many_arguments)]
    pub fn open_tiered(
        open_db: &Path,
        open_vault: &Path,
        open_key: &str,
        thread_db: &Path,
        thread_vault: &Path,
        thread_key: &str,
        priv_db: &Path,
        priv_vault: &Path,
        policy: simonides::membrane::TierPolicy,
    ) -> Result<Self, String> {
        let tiers = Tiered::open(
            open_db, open_vault, open_key,
            thread_db, thread_vault, thread_key,
            priv_db, priv_vault, policy,
        )?;
        // The default write/legacy store is L1. Anything tier-sensitive goes
        // through `self.tiers`.
        //
        // **This fallback is safe for READS and WRITES and UNSAFE for
        // MAINTENANCE, and the distinction cost us a live bug (2026-07-29).**
        //
        // For a read or a write, `self.store` being L1 means an un-tier-aware
        // path can only ever touch published data — it fails closed, which is
        // what you want.
        //
        // For MAINTENANCE it inverts. `t_rebuild` used `self.store` and
        // therefore reindexed L1 and silently skipped L2, so every diary written to
        // L2 was stored but never embedded — invisible to search,
        // while `rebuild_status` reported success because the L1 rebuild really
        // had finished. Failing "safe" there meant failing to maintain private
        // data, quietly.
        //
        // So: any NEW code that maintains, migrates, reindexes, backs up or
        // audits must enumerate `tiers.readable()` — never `self.store`. See
        // `rebuild_targets()`. A maintenance path that silently covers a subset
        // is worse than one that refuses.
        let store = std::sync::Arc::new(
            PalaceStore::open(open_db, open_vault, open_key).map_err(|e| e.to_string())?,
        );
        Ok(Self {
            store,
            tiers: Some(std::sync::Arc::new(tiers)),
            rebuild: std::sync::Arc::new(std::sync::Mutex::new(RebuildState::default())),
            auto_index: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// The tiered view, when the palace is split.
    fn tiers(&self) -> Option<&Tiered> {
        self.tiers.as_deref()
    }

    /// Route a write by `(wing, room)`. Pre-split this is the single store;
    /// post-split it is the file that tier names, and a locked tier refuses.
    fn write_store(&self, wing: &str, room: &str) -> Result<std::sync::Arc<PalaceStore>, String> {
        match self.tiers() {
            Some(t) => t.writable_for(wing, room),
            None => Ok(self.store.clone()),
        }
    }

    /// Find the store holding `id`, with its drawer. `None` means "not visible
    /// from here" and deliberately does not distinguish absent from locked.
    fn find_owner(&self, id: &str) -> Result<Option<(std::sync::Arc<PalaceStore>, Drawer)>, String> {
        match self.tiers() {
            Some(t) => t.owner_of(id),
            None => Ok(self
                .store
                .get(id)
                .map_err(|e| e.to_string())?
                .map(|d| (self.store.clone(), d))),
        }
    }

    /// The `tools` array for `tools/list`: name, description and JSON Schema per tool.
    pub fn tool_schemas() -> Value {
        protocol::tool_definitions()
    }

    pub fn tool_names() -> Vec<String> {
        Self::tool_schemas()
            .as_array().unwrap().iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect()
    }

    /// Turn background indexing after writes on or off. See `auto_index`.
    pub fn set_auto_index(&self, on: bool) {
        self.auto_index.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn handle_tool(&self, name: &str, args: &Value) -> Result<Value, String> {
        let out = self.dispatch(name, args);
        if out.is_ok()
            && matches!(name, "add_drawer" | "update_drawer" | "diary_write")
            && self.auto_index.load(std::sync::atomic::Ordering::Relaxed)
        {
            // Best effort: if a rebuild is already running it will not see this
            // drawer, and the next write or an explicit `rebuild` picks it up.
            let _ = self.t_rebuild(&json!({}));
        }
        out
    }

    fn dispatch(&self, name: &str, args: &Value) -> Result<Value, String> {
        match name {
            "status" => self.t_status(),
            "list_wings" => self.t_list_wings(),
            "list_rooms" => self.t_list_rooms(args),
            "list_drawers" => self.t_list_drawers(args),
            "get_drawer" => self.t_get_drawer(args),
            "add_drawer" => self.t_add_drawer(args),
            "update_drawer" => self.t_update_drawer(args),
            "delete_drawer" => self.t_delete_drawer(args),
            "check_duplicate" => self.t_check_duplicate(args),
            "search" => self.t_search(args),
            "diary_read" => self.t_diary_read(args),
            "diary_write" => self.t_diary_write(args),
            "create_tunnel" => self.t_create_tunnel(args),
            "list_tunnels" => self.t_list_tunnels(args),
            "find_tunnels" => self.t_find_tunnels(args),
            "follow_tunnels" => self.t_follow_tunnels(args),
            "delete_tunnel" => self.t_delete_tunnel(args),
            "traverse" => self.t_traverse(args),
            "kg_add" => self.t_kg_add_triple(args),
            "kg_invalidate" => self.t_kg_invalidate(args),
            "kg_query" => self.t_kg_query_entity(args),
            "kg_stats" => self.t_kg_stats(),
            "kg_timeline" => self.t_kg_timeline(args),
            "graph_stats" => self.t_graph_stats(),
            "get_taxonomy" => self.t_get_taxonomy(),
            "get_aaak_spec" => Ok(aaak_spec()),
            "memories_filed_away" => self.t_memories_filed_away(args),
            "hook_settings" => Ok(json!({"hooks": [], "note": "simonides has no hook layer; no-op in v1"})),
            "reconnect" => self.t_reconnect(),
            "rebuild" => self.t_rebuild(args),
            "rebuild_status" => self.t_rebuild_status(),
            // Read-only, argument-free, and the ONLY tool that reveals lock
            // state. There is deliberately no `membrane_unlock` tool: the MCP
            // transport may run through a remote connector, so a passphrase passed
            // as a tool argument would leave the machine — the precise boundary
            // the membrane exists to stop things crossing. Unlocking is a local
            // action, out of band from this channel.
            "membrane_status" => self.t_membrane_status(),
            other => Err(format!("unknown tool: {other}")),
        }
    }

    fn t_add_drawer(&self, a: &Value) -> Result<Value, String> {
        let wing = str_arg(a, "wing")?;
        let room = str_arg(a, "room")?;
        let content = str_arg(a, "content")?;
        let provenance = a.get("provenance").and_then(|v| v.as_str()).unwrap_or("mcp");
        // Routed by tier. A drawer bound for a locked tier is refused here,
        // never written to an open file instead.
        let store = self.write_store(wing, room)?;
        let d = match a.get("name").and_then(|v| v.as_str()) {
            Some(n) => store.insert_named(wing, room, content, provenance, n),
            None => store.insert(wing, room, content, provenance),
        }
        .map_err(|e| e.to_string())?;
        Ok(drawer_json(&d))
    }

    fn t_get_drawer(&self, a: &Value) -> Result<Value, String> {
        let id = str_arg(a, "id")?;
        // `found: false` covers both "no such drawer" and "it lives in a tier
        // this session cannot open", and MUST NOT distinguish them — a reply
        // that says "exists but locked" has disclosed the drawer.
        match self.find_owner(id)? {
            Some((_, d)) => Ok(drawer_json(&d)),
            None => Ok(json!({"found": false, "id": id})),
        }
    }

    fn t_update_drawer(&self, a: &Value) -> Result<Value, String> {
        let id = str_arg(a, "id")?;
        let content = str_arg(a, "content")?;
        let (store, _) = self
            .find_owner(id)?
            .ok_or_else(|| format!("no drawer {id}"))?;
        let d = store.update_drawer(id, content).map_err(|e| e.to_string())?;
        Ok(drawer_json(&d))
    }

    fn t_delete_drawer(&self, a: &Value) -> Result<Value, String> {
        let id = str_arg(a, "id")?;
        let (store, _) = self
            .find_owner(id)?
            .ok_or_else(|| format!("no drawer {id}"))?;
        store.tombstone(id).map_err(|e| e.to_string())?;
        Ok(json!({"ok": true, "tombstoned": id}))
    }

    fn t_check_duplicate(&self, a: &Value) -> Result<Value, String> {
        let content = str_arg(a, "content")?;
        let limit = a.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let h = self.health()?;
        let degraded = h.chunks_embedded < h.chunks_total;
        let mut matches: Vec<Value> = Vec::new();
        {
            for hit in self.search_hits(content, limit)? {
                if hit.drawer.content == content {
                    matches.push(drawer_json(&hit.drawer));
                }
            }
        }
        Ok(json!({"duplicate": !matches.is_empty(), "matches": matches, "exact_only": true, "degraded_vector_recall": degraded}))
    }

    fn t_list_drawers(&self, a: &Value) -> Result<Value, String> {
        let wing = a.get("wing").and_then(|v| v.as_str());
        let room = a.get("room").and_then(|v| v.as_str());
        let limit = a.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let offset = a.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        // `total` = true count for this filter. Clients that paginate newest-first
        // read it (probe with limit=1 for the count, then windows). Emitting it is
        // the fix for the silent-empty bug: a nonexistent wing returns 0, a populated
        // wing its real count. Drawers come back oldest-first (id ASC) -- the native
        // order such a client's reversal expects.
        let (total, ds) = match self.tiers() {
            Some(t) => (t.count_live(wing, room)?, t.list_paged(wing, room, limit, offset)?),
            None => (
                self.store.count_live(wing, room).map_err(|e| e.to_string())?,
                self.store.list_paged(wing, room, limit, offset).map_err(|e| e.to_string())?,
            ),
        };
        Ok(json!({
            "drawers": ds.iter().map(drawer_json).collect::<Vec<_>>(),
            "total": total,
            "returned": ds.len(),
            "offset": offset,
            "whole_corpus": false
        }))
    }

    fn t_list_wings(&self) -> Result<Value, String> {
        // Union across readable tiers. A wing usually spans them —
        // one agent's wing can hold work-log rows in L1 and diary rows in L2 — so the
        // union is the honest answer and duplicates are a storage artifact.
        let wings = match self.tiers() {
            Some(t) => t.fan_names(|s| s.list_wings().map_err(|e| e.to_string()))?,
            None => self.store.list_wings().map_err(|e| e.to_string())?,
        };
        Ok(json!({"wings": wings, "count": wings.len()}))
    }

    fn t_list_rooms(&self, a: &Value) -> Result<Value, String> {
        let wing = str_arg(a, "wing")?;
        let rooms = match self.tiers() {
            Some(t) => t.fan_names(|s| s.list_rooms(wing).map_err(|e| e.to_string()))?,
            None => self.store.list_rooms(wing).map_err(|e| e.to_string())?,
        };
        Ok(json!({"wing": wing, "rooms": rooms, "count": rooms.len()}))
    }

    fn t_status(&self) -> Result<Value, String> {
        let h = self.health()?;
        let mut v = status_json(&h);
        // Lock state is surfaced HERE and nowhere else. Counts above are the
        // total of what this session can READ — reporting the full corpus while
        // the vault is closed would disclose how much sits behind it.
        if let Some(t) = self.tiers() {
            v["membrane"] = t.membrane_status();
        }
        Ok(v)
    }

    fn t_membrane_status(&self) -> Result<Value, String> {
        match self.tiers() {
            Some(t) => Ok(t.membrane_status()),
            None => Ok(json!({
                "tiers_readable": ["L1"],
                "vault_unlocked": false,
                "note": "this palace is not split; there is no membrane to report"
            })),
        }
    }

    /// Health across readable tiers, or the single store pre-split.
    fn health(&self) -> Result<HealthReport, String> {
        match self.tiers() {
            Some(t) => t.health_merged(),
            None => self.store.health().map_err(|e| e.to_string()),
        }
    }

    // ---- fan-out helpers -------------------------------------------------
    //
    // Each mirrors the same shape: post-split, ask every READABLE tier and
    // concatenate; pre-split, ask the one store. Written out per return type
    // rather than made generic because the closure has to name the store
    // method, and three short helpers read better than one clever one.

    fn fan_drawers<F>(&self, mut f: F) -> Result<Vec<Drawer>, String>
    where
        F: FnMut(&PalaceStore) -> Result<Vec<Drawer>, String>,
    {
        match self.tiers() {
            Some(t) => t.fan_out(f),
            None => f(&self.store),
        }
    }

    fn fan_graph<T, F>(&self, mut f: F) -> Result<(Vec<T>, Vec<&'static str>), String>
    where
        F: FnMut(&PalaceStore) -> Result<Vec<T>, simonides::PalaceError>,
    {
        match self.tiers() {
            Some(t) => t.fan_graph(f),
            None => match f(&self.store) {
                Ok(v) => Ok((v, vec!["single"])),
                Err(simonides::PalaceError::NotFound { .. }) => Ok((vec![], vec!["single"])),
                Err(e) => Err(e.to_string()),
            },
        }
    }

    /// Server-level cross-tier walk. Tiered deployments walk the whole readable
    /// graph (edges gathered across tiers, endpoints resolved across tiers,
    /// unresolvable ones reported absent). A single-store deployment has no
    /// cross-tier surface, so it degrades to the store's own traverse with an
    /// empty absent set.
    fn traverse_cross_tier(
        &self,
        start: &str,
        depth: usize,
    ) -> Result<CrossTierWalk, String> {
        match self.tiers() {
            Some(t) => t.traverse_cross_tier(start, depth),
            None => {
                let ds = match self.store.traverse(start, depth) {
                    Ok(v) => v,
                    Err(simonides::PalaceError::NotFound { .. }) => vec![],
                    Err(e) => return Err(e.to_string()),
                };
                Ok((ds, vec![], vec!["single"]))
            }
        }
    }

    fn fan_triples<F, T>(&self, mut f: F) -> Result<Vec<T>, String>
    where
        F: FnMut(&PalaceStore) -> Result<Vec<T>, String>,
    {
        match self.tiers() {
            Some(t) => t.fan_out(f),
            None => f(&self.store),
        }
    }

    fn search_hits(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<simonides::retrieval::SearchHit>, String> {
        match self.tiers() {
            Some(t) => t.search(query, limit),
            None => match self.store.search(query, limit).map_err(|e| e.to_string())? {
                SearchResult::Hits(h) => Ok(h),
                SearchResult::Empty { .. } => Ok(Vec::new()),
            },
        }
    }

    /// A tunnel belongs to the MOST RESTRICTIVE tier of its two endpoints, so
    /// no file ever names a drawer id that lives in a stricter file. Same rule
    /// the splitter applies; stated once here so the live path cannot drift
    /// from the migration path.
    fn store_for_tunnel(&self, from: &str, to: &str) -> Result<std::sync::Arc<PalaceStore>, String> {
        let Some(t) = self.tiers() else {
            return Ok(self.store.clone());
        };
        // Resolve both endpoints; whichever tier is found LAST in readable()
        // order (most-open first) is the most restrictive of the two.
        let mut chosen: Option<std::sync::Arc<PalaceStore>> = None;
        for id in [from, to] {
            match t.owner_of(id)? {
                Some((st, _)) => chosen = Some(st),
                None => {
                    return Err(format!(
                        "tunnel endpoint {id} is not visible from this session — \
                         refusing to create a tunnel that would name it"
                    ))
                }
            }
        }
        chosen.ok_or_else(|| "no readable tier".to_string())
    }

    /// The store holding a tunnel. Tunnels are not drawers, so `owner_of` does
    /// not apply; ask each readable tier whether it has the row.
    fn tunnel_owner(&self, id: &str) -> Result<std::sync::Arc<PalaceStore>, String> {
        let Some(t) = self.tiers() else {
            return Ok(self.store.clone());
        };
        for (_, st) in t.readable() {
            if st
                .find_tunnels("")
                .map(|ts| ts.iter().any(|x| x.id == id))
                .unwrap_or(false)
            {
                return Ok(st);
            }
        }
        Err(format!("no tunnel {id}"))
    }

    fn t_search(&self, a: &Value) -> Result<Value, String> {
        let query = str_arg(a, "query")?;
        let limit = a.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let h = self.health()?;
        let degraded = h.chunks_embedded < h.chunks_total;
        let hits = match self.tiers() {
            // Searches only what this session can open. A locked tier is not
            // "no results" — it is not searched at all, so nothing about it
            // reaches the caller, not even a count.
            Some(t) => t.search(query, limit)?,
            None => match self.store.search(query, limit).map_err(|e| e.to_string())? {
                SearchResult::Hits(hits) => hits,
                SearchResult::Empty { .. } => Vec::new(),
            },
        };
        if hits.is_empty() {
            return Ok(json!({"hits": [], "query": query, "degraded_vector_recall": degraded}));
        }
        Ok(json!({
            "hits": hits.iter().map(|hit| json!({
                "drawer": drawer_json(&hit.drawer),
                "closet_context": hit.closet_context,
            })).collect::<Vec<_>>(),
            "degraded_vector_recall": degraded
        }))
    }

    fn t_diary_read(&self, a: &Value) -> Result<Value, String> {
        let agent = str_arg_any(a, &["agent", "agent_name"])?;
        let last_n = a.get("last_n").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        // Diaries are L2, so this resolves without a passphrase. That is the
        // whole point of L2 — "we always have our diaries" — and it is why
        // diary_read must never be gated behind an unlock.
        let ds = match self.tiers() {
            Some(t) => {
                let mut v = t.fan_out(|s| s.diary_read(agent, last_n).map_err(|e| e.to_string()))?;
                v.sort_by(|a, b| b.id.cmp(&a.id));
                v.truncate(last_n);
                v
            }
            None => self.store.diary_read(agent, last_n).map_err(|e| e.to_string())?,
        };
        Ok(json!({"agent": agent, "entries": ds.iter().map(drawer_json).collect::<Vec<_>>()}))
    }

    fn t_diary_write(&self, a: &Value) -> Result<Value, String> {
        let agent = str_arg_any(a, &["agent", "agent_name"])?;
        let entry = str_arg(a, "entry")?;
        let topic = a.get("topic").and_then(|v| v.as_str());
        // A diary is `wing_<agent>/personal-diary`. Honor an explicit `wing=` if the
        // caller passed one; otherwise derive `wing_<agent>`. Normalise to lowercase
        // either way so `wing_Ada` and `wing_ada` cannot split into two wings. The
        // tier is the policy's decision, from wing and room, not this function's.
        let wing = a
            .get("wing")
            .and_then(|v| v.as_str())
            .map(|w| w.to_ascii_lowercase())
            .unwrap_or_else(|| format!("wing_{}", agent.to_ascii_lowercase()));
        let store = self.write_store(&wing, "personal-diary")?;
        let d = store.diary_write(agent, entry, topic).map_err(|e| e.to_string())?;
        Ok(drawer_json(&d))
    }

    fn t_create_tunnel(&self, a: &Value) -> Result<Value, String> {
        let from = str_arg(a, "from")?;
        let to = str_arg(a, "to")?;
        let relation = str_arg(a, "relation")?;
        let confidence = a.get("confidence").and_then(|v| v.as_f64()).map(|f| f as f32);
        let store = self.store_for_tunnel(from, to)?;
        let t = store.create_tunnel(from, to, relation, confidence).map_err(|e| e.to_string())?;
        Ok(tunnel_json(&t))
    }

    fn t_list_tunnels(&self, a: &Value) -> Result<Value, String> {
        let drawer = str_arg(a, "drawer")?;
        let (ts, tiers) = self.fan_graph(|s| s.list_tunnels(drawer))?;
        Ok(json!({"tunnels": ts.iter().map(tunnel_json).collect::<Vec<_>>(), "tiers_consulted": tiers}))
    }

    fn t_find_tunnels(&self, a: &Value) -> Result<Value, String> {
        let relation = str_arg(a, "relation")?;
        let (ts, tiers) = self.fan_graph(|s| s.find_tunnels(relation))?;
        Ok(json!({"tunnels": ts.iter().map(tunnel_json).collect::<Vec<_>>(), "tiers_consulted": tiers}))
    }

    fn t_follow_tunnels(&self, a: &Value) -> Result<Value, String> {
        let drawer = str_arg(a, "drawer")?;
        let relation = str_arg(a, "relation")?;
        let depth = a.get("depth").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let (ds, tiers) = self.fan_graph(|s| s.follow_tunnels(drawer, relation, depth))?;
        Ok(json!({"drawers": ds.iter().map(drawer_json).collect::<Vec<_>>(), "tiers_consulted": tiers}))
    }

    fn t_delete_tunnel(&self, a: &Value) -> Result<Value, String> {
        let id = str_arg(a, "id")?;
        self.tunnel_owner(id)?.tombstone_tunnel(id).map_err(|e| e.to_string())?;
        Ok(json!({"ok": true, "tombstoned_tunnel": id}))
    }

    fn t_traverse(&self, a: &Value) -> Result<Value, String> {
        let drawer = str_arg(a, "drawer")?;
        let depth = a.get("depth").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let (ds, absent, tiers) = self.traverse_cross_tier(drawer, depth)?;
        Ok(json!({
            "drawers": ds.iter().map(drawer_json).collect::<Vec<_>>(),
            "absent": absent.iter().map(|id| json!({"id": id, "status": "unresolved"})).collect::<Vec<_>>(),
            "tiers_consulted": tiers,
        }))
    }

    fn t_kg_add_triple(&self, a: &Value) -> Result<Value, String> {
        let subject = str_arg(a, "subject")?;
        let predicate = str_arg(a, "predicate")?;
        let object = str_arg(a, "object")?;
        let valid_from = opt_dt(a, "valid_from")?;
        let valid_to = opt_dt(a, "valid_to")?;
        let confidence = a.get("confidence").and_then(|v| v.as_f64()).map(|f| f as f32);
        // A hand-asserted triple carries NO tier information: no wing, no room,
        // no source drawer. Rather than guess, this writes to the OPEN tier and
        // says so in the reply. Two reasons it is the right default: the
        // subject/predicate/object arrived as plain strings over the MCP wire,
        // which is the L1-facing channel, so the content has already crossed to
        // that boundary; and a fact placed too openly is visible and
        // correctable, whereas one silently buried in a private file is neither.
        //
        // Extracted triples are different — they inherit their drawer's tier via
        // rebuild_kg(), which runs per file. This path is only for manual asserts.
        let t = self
            .store
            .kg_add_triple(subject, predicate, object, valid_from, valid_to, confidence)
            .map_err(|e| e.to_string())?;
        let mut v = serde_json::to_value(&t).map_err(|e| e.to_string())?;
        if self.tiers().is_some() {
            v["tier"] = json!("L1");
            v["note"] = json!(
                "manually asserted triples have no tier of their own and are written to the \
                 open tier. Extracted triples inherit their drawer's tier via rebuild_kg()."
            );
        }
        Ok(v)
    }

    fn t_kg_invalidate(&self, a: &Value) -> Result<Value, String> {
        let subject = str_arg(a, "subject")?;
        let predicate = str_arg(a, "predicate")?;
        let object = str_arg(a, "object")?;
        let ended = opt_dt(a, "ended")?.ok_or_else(|| "kg_invalidate requires 'ended' (rfc3339)".to_string())?;
        // Invalidation must reach the fact wherever it lives, so it runs against
        // every readable tier. A triple that exists in a locked tier is not
        // invalidated and not reported — consistent with it being invisible.
        let mut hit = false;
        match self.tiers() {
            Some(t) => {
                for (_, st) in t.readable() {
                    if st.kg_invalidate(subject, predicate, object, ended).is_ok() {
                        hit = true;
                    }
                }
            }
            None => {
                self.store
                    .kg_invalidate(subject, predicate, object, ended)
                    .map_err(|e| e.to_string())?;
                hit = true;
            }
        }
        Ok(json!({"ok": hit}))
    }

    fn t_kg_query_entity(&self, a: &Value) -> Result<Value, String> {
        let entity = str_arg(a, "entity")?;
        let as_of = opt_dt(a, "as_of")?;
        let direction = match a.get("direction").and_then(|v| v.as_str()) {
            Some("outgoing") => KgDirection::Outgoing,
            Some("incoming") => KgDirection::Incoming,
            Some("both") | None => KgDirection::Both,
            Some(other) => return Err(format!("bad direction: {other} (want outgoing|incoming|both)")),
        };
        let ts = self.fan_triples(|s| s.kg_query_entity(entity, as_of, direction).map_err(|e| e.to_string()))?;
        Ok(json!({"triples": ts.iter().map(|t| serde_json::to_value(t).unwrap_or(Value::Null)).collect::<Vec<_>>()}))
    }

    fn t_kg_stats(&self) -> Result<Value, String> {
        let h = self.health()?;
        Ok(json!({"entities": h.kg_entities, "triples": h.kg_triples}))
    }

    fn t_kg_timeline(&self, a: &Value) -> Result<Value, String> {
        let entity = str_arg(a, "entity")?;
        let ts = self.fan_triples(|s| s.kg_timeline(entity).map_err(|e| e.to_string()))?;
        Ok(json!({"entity": entity, "timeline": ts.iter().map(|t| serde_json::to_value(t).unwrap_or(Value::Null)).collect::<Vec<_>>()}))
    }

    fn t_graph_stats(&self) -> Result<Value, String> {
        let h = self.health()?;
        Ok(json!({"kg_entities": h.kg_entities, "kg_triples": h.kg_triples, "closets": h.closets, "drawers": h.total}))
    }

    fn t_get_taxonomy(&self) -> Result<Value, String> {
        let wings = match self.tiers() {
            Some(t) => t.fan_names(|s| s.list_wings().map_err(|e| e.to_string()))?,
            None => self.store.list_wings().map_err(|e| e.to_string())?,
        };
        let mut tax = serde_json::Map::new();
        for w in &wings {
            let rooms = match self.tiers() {
                Some(t) => t.fan_names(|s| s.list_rooms(w).map_err(|e| e.to_string()))?,
                None => self.store.list_rooms(w).map_err(|e| e.to_string())?,
            };
            tax.insert(w.clone(), json!(rooms));
        }
        Ok(json!({"taxonomy": Value::Object(tax), "wing_count": wings.len()}))
    }

    fn t_memories_filed_away(&self, a: &Value) -> Result<Value, String> {
        let limit = a.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let h = self.health()?;
        // Recent across readable tiers, newest first. A locked tier contributes
        // nothing, so "recent" means recent among what this session can see.
        let mut recent = self.fan_drawers(|s| s.list(limit).map_err(|e| e.to_string()))?;
        recent.sort_by(|a, b| b.id.cmp(&a.id));
        recent.truncate(limit);
        Ok(json!({"total": h.total, "recent": recent.iter().map(drawer_json).collect::<Vec<_>>()}))
    }

    fn t_reconnect(&self) -> Result<Value, String> {
        let h = self.health()?;
        Ok(json!({"ok": true, "reconnected": true, "total": h.total}))
    }

    /// Which stores a rebuild must visit.
    ///
    /// Extracted so it can be asserted directly. The bug this replaced was
    /// entirely a question of *which stores* got rebuilt — it maintained L1 and
    /// silently skipped L2 — so that is the thing a test has to name. A test
    /// that drove a rebuild and then checked the store it already knew about
    /// would have agreed with the bug.
    pub(crate) fn rebuild_targets(&self) -> Vec<(&'static str, std::sync::Arc<PalaceStore>)> {
        match self.tiers() {
            Some(t) => t.readable(),
            None => vec![("single", self.store.clone())],
        }
    }

    fn t_rebuild(&self, a: &Value) -> Result<Value, String> {
        // Default is VECTORS-ONLY: the KG is now maintained inline on every insert
        // (store::insert_impl), so the routine "clear degraded_vector_recall after
        // saving" path only needs to embed the few pending chunks -- incremental,
        // worker-bounded, seconds, phone-viable. `full_kg=true` also re-runs the
        // full-corpus KG extract, for repair/migration.
        let full_kg = a.get("full_kg").and_then(|v| v.as_bool()).unwrap_or(false);
        {
            let mut g = lock_state(&self.rebuild);
            if g.running {
                // Ask the running pass to go round once more, so this write is indexed too.
                g.dirty = true;
                return Ok(json!({"started": false, "reason": "already running; will re-run", "phase": g.phase}));
            }
            // Claimed here, under the lock, so two calls cannot both start one.
            g.running = true;
            g.phase = "starting".to_string();
            g.error = None;
        }
        // EVERY readable tier, not just L1.
        //
        // This was `self.store.clone()`, which post-split is the OPEN tier and
        // only ever the open tier. Rebuild therefore maintained L1 and silently
        // skipped L2 — so every diary, letter and `state` drawer written to L2
        // landed with `reindex_pending = true` and was never embedded.
        // Stored, but invisible to search. `rebuild_status` reported
        // `phase=done, error=null` throughout, because the L1 rebuild genuinely
        // did complete. Found from a stuck
        // `reindex_pending` count; confirmed by reading this function.
        //
        // The design error is worth stating, because the reasoning that
        // produced it sounded right. `self.store` was left as L1 so that any
        // path not yet made tier-aware would "fail safe." That holds for a
        // WRITE path — failing safe means it cannot reach private data. It is
        // exactly inverted for a MAINTENANCE path, where the same default means
        // it silently does not maintain private data. Same fallback, opposite
        // consequence. A maintenance path must fail LOUD, never quiet.
        let stores = self.rebuild_targets();
        // Names captured BEFORE the move, for the reply.
        let tiers: Vec<&'static str> = stores.iter().map(|(t, _)| *t).collect();
        let state = self.rebuild.clone();
        std::thread::spawn(move || {
            // Passes repeat while writes arrived during one (`dirty`), so no
            // drawer is left unindexed until the next write. A panic must not
            // leave `running` stuck at true, so the work is caught and reported.
            //
            // A locked tier is not in `stores` and is therefore not rebuilt:
            // unreachable by construction, and `membrane_status` reports it.
            let mut kg = full_kg;
            loop {
                lock_state(&state).dirty = false;
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
                    for (tier, store) in &stores {
                        if kg {
                            lock_state(&state).phase = format!("{tier}:kg");
                            store.rebuild_kg().map_err(|e| format!("{tier} rebuild_kg: {e}"))?;
                        }
                        lock_state(&state).phase = format!("{tier}:vectors");
                        store.rebuild_vectors().map_err(|e| format!("{tier} rebuild_vectors: {e}"))?;
                    }
                    Ok(())
                }));
                kg = false;
                let mut g = lock_state(&state);
                match outcome {
                    Ok(Ok(())) if g.dirty => continue,
                    Ok(Ok(())) => {
                        g.phase = "done".to_string();
                    }
                    Ok(Err(e)) => {
                        g.phase = "error".to_string();
                        g.error = Some(e);
                    }
                    Err(_) => {
                        g.phase = "error".to_string();
                        g.error = Some("rebuild panicked; see the server log".to_string());
                    }
                }
                g.running = false;
                break;
            }
        });
        Ok(json!({
            "started": true,
            "full_kg": full_kg,
            "tiers": tiers,
            "note": "background reindex across every READABLE tier; poll rebuild_status. \
                     A locked tier is not rebuilt — unlock it and re-run. default=vectors-only \
                     (KG maintained inline on insert); pass full_kg=true for a full-corpus KG rebuild"
        }))
    }

    fn t_rebuild_status(&self) -> Result<Value, String> {
        let h = self.health()?;
        let g = lock_state(&self.rebuild);
        let mut out = json!({
            "running": g.running,
            "phase": g.phase,
            "error": g.error,
            "chunks_total": h.chunks_total,
            "chunks_embedded": h.chunks_embedded,
            "reindex_pending": h.reindex_pending,
            "kg_triples": h.kg_triples
        });
        drop(g);
        // `reindex_pending` above is merged over READABLE tiers only. A locked
        // tier contributes nothing to that sum and would otherwise vanish from
        // the report, leaving "0 pending" indistinguishable from "could not
        // look." So name every configured tier with its lock state: an open one
        // is measured, a locked one reads as "locked" and is never counted as
        // zero. Uses tier_states() (lock state ONLY, no per-tier quantity), so
        // structural absence for a locked L3 is preserved.
        if let Some(t) = self.tiers() {
            let tiers: Vec<Value> = t
                .tier_states()
                .into_iter()
                .map(|(tier, state)| json!({ "tier": tier, "state": state.as_str() }))
                .collect();
            out["tiers"] = json!(tiers);
            out["reindex_pending_scope"] = json!(
                "merged over open tiers only; a locked tier is listed under \"tiers\" as \"locked\" and never counted as 0"
            );
        }
        Ok(out)
    }
}

fn aaak_spec() -> Value {
    json!({
        "aaak": "AAAK is a compressed memory dialect readable by humans and LLMs without decoding.",
        "entities": "3-letter uppercase codes",
        "emotions": "*action markers* before/during text",
        "structure": "pipe-separated fields; FAM/PROJ/warnings",
        "dates": "ISO (2026-03-31)",
        "importance": "star 1-5",
        "note": "static spec surfaced for wake-up parity with the legacy server"
    })
}

fn str_arg<'a>(a: &'a Value, key: &str) -> Result<&'a str, String> {
    a.get(key).and_then(|v| v.as_str()).ok_or_else(|| format!("missing required string arg: {key}"))
}

/// First of several spellings, or a message naming all of them.
///
/// Clients disagree about argument names: one sends `agent_name`, another
/// `agent`. A tool that accepts only one spelling fails silently for the rest.
///
/// Handled here rather than in each client because the server is the thing every
/// caller reaches, and because a tool surface should be **forgiving and hard to
/// misuse** — the same reason the palace is built to be driven by a small local
/// model that will not reason its way out of an argument-name mismatch. The
/// error, when it does fire, names every accepted spelling instead of only the
/// one this function happened to try first.
fn str_arg_any<'a>(a: &'a Value, keys: &[&str]) -> Result<&'a str, String> {
    for k in keys {
        if let Some(s) = a.get(*k).and_then(|v| v.as_str()) {
            return Ok(s);
        }
    }
    Err(format!("missing required string arg: one of {}", keys.join(", ")))
}

fn opt_dt(a: &Value, key: &str) -> Result<Option<DateTime<Utc>>, String> {
    match a.get(key).and_then(|v| v.as_str()) {
        Some(s) => Ok(Some(DateTime::parse_from_rfc3339(s).map_err(|e| format!("{key}: bad rfc3339: {e}"))?.with_timezone(&Utc))),
        None => Ok(None),
    }
}

fn drawer_json(d: &Drawer) -> Value {
    json!({"id": d.id, "wing": d.wing, "room": d.room, "content": d.content, "provenance": d.provenance, "state": format!("{:?}", d.state), "reindex_pending": d.reindex_pending, "created_at": d.created_at.to_rfc3339(), "seq": d.seq})
}

fn tunnel_json(t: &Tunnel) -> Value {
    json!({"id": t.id, "from": t.from_drawer_id, "to": t.to_drawer_id, "relation": t.relation, "confidence": t.confidence, "tombstoned": t.tombstoned})
}

fn status_json(h: &HealthReport) -> Value {
    json!({"total": h.total, "live": h.live, "tombstoned": h.tombstoned, "reindex_pending": h.reindex_pending, "kg_entities": h.kg_entities, "kg_triples": h.kg_triples, "closets": h.closets, "closets_stale": h.closets_stale, "chunks_total": h.chunks_total, "chunks_embedded": h.chunks_embedded, "embedding_model_version": h.embedding_model_version, "vector_recall_partial": h.chunks_embedded < h.chunks_total})
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const KEY: &str = "mig1-test-key";

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
    fn srv() -> (McpServer, TempDir) {
        let tmp = TempDir::new().unwrap();
        let db = tmp.path().join("palace.db");
        let vault = tmp.path().join("vault");
        let s = McpServer::open(&db, &vault, KEY).expect("open");
        (s, tmp)
    }

    // The MemPalace-compatible tool surface. This list is the drop-in contract.
    const COMPAT_TOOLS: [&str; 29] = [
        "status","list_wings","list_rooms","list_drawers",
        "get_drawer","add_drawer","update_drawer","delete_drawer",
        "check_duplicate","search","diary_read","diary_write",
        "create_tunnel","list_tunnels","find_tunnels","follow_tunnels",
        "delete_tunnel","traverse","kg_add","kg_invalidate",
        "kg_query","kg_stats","kg_timeline","graph_stats",
        "get_taxonomy","get_aaak_spec","memories_filed_away",
        "hook_settings","reconnect",
    ];

    #[test]
    fn every_compat_tool_is_registered() {
        let names = McpServer::tool_names();
        for t in COMPAT_TOOLS {
            assert!(names.contains(&t.to_string()), "compatible tool missing from surface: {t}");
        }
        assert!(names.contains(&"rebuild".to_string()), "rebuild tool present");
        assert!(names.contains(&"rebuild_status".to_string()), "rebuild_status present");
        assert!(
            names.contains(&"membrane_status".to_string()),
            "membrane_status present — the only tool that reveals lock state"
        );
        assert_eq!(
            names.len(),
            COMPAT_TOOLS.len() + 3,
            "surface = 29 compatible tools + 2 maintenance + 1 membrane"
        );
    }

    /// A caller cannot label a drawer less restrictively than the policy that
    /// routed it. `provenance` comes straight from the MCP client, so a
    /// `tier=` segment in it must be replaced by the policy's tier, not kept.
    #[test]
    fn a_caller_supplied_tier_label_is_overridden_by_the_policy() {
        // Single store: tiers are off, so every drawer is labelled `single`.
        let (s, _tmp) = srv();
        let v = s
            .handle_tool("add_drawer", &json!({"wing":"w","room":"r","content":"c","provenance":"x|tier=L1"}))
            .expect("add");
        assert_eq!(v["provenance"], "x|tier=single", "{v}");

        // Tiered: an agent_ wing routes to L2; the caller claims L1.
        let d = tempfile::tempdir().expect("tmp");
        let p = d.path();
        let t = McpServer::open_tiered(
            &p.join("palace.db"), &p.join("v1"),
            "x'1111111111111111111111111111111111111111111111111111111111111111'",
            &p.join("palace_thread.db"), &p.join("v2"),
            "x'2222222222222222222222222222222222222222222222222222222222222222'",
            &p.join("palace_private.db"), &p.join("v3"), test_policy(),
        )
        .expect("tiered");
        let v = t
            .handle_tool("add_drawer", &json!({"wing":"agent_ada","room":"notes","content":"c","provenance":"tier=L1|src=y"}))
            .expect("add");
        assert_eq!(v["provenance"], "src=y|tier=L2", "{v}");
    }

    /// **The regression gate for the silently-unindexed-L2 bug.**
    ///
    /// `t_rebuild` used `self.store`, which post-split is the OPEN tier and only
    /// ever the open tier. So rebuild maintained L1 and skipped L2 — every diary,
    /// letter and `state` drawer written to L2 landed with
    /// `reindex_pending = true` and was never embedded. Stored, invisible to
    /// search, and `rebuild_status` said `phase=done, error=null` throughout,
    /// because the L1 rebuild genuinely completed. Found from a stuck
    /// `reindex_pending` count.
    ///
    /// The reasoning that produced it: `self.store` was deliberately left as L1
    /// so an un-tier-aware path would "fail safe." True for a WRITE path —
    /// failing safe means it cannot reach private data. Inverted for a
    /// MAINTENANCE path, where the same default means it silently does not
    /// maintain private data. Same fallback, opposite consequence.
    #[test]
    fn rebuild_visits_every_readable_tier_not_just_the_open_one() {
        let d = tempfile::tempdir().expect("tmp");
        let p = d.path();
        let s = McpServer::open_tiered(
            &p.join("palace.db"), &p.join("v1"),
            "x'1111111111111111111111111111111111111111111111111111111111111111'",
            &p.join("palace_thread.db"), &p.join("v2"),
            "x'2222222222222222222222222222222222222222222222222222222222222222'",
            &p.join("palace_private.db"), &p.join("v3"), test_policy(),
        )
        .expect("tiered");

        let tiers: Vec<&str> = s.rebuild_targets().iter().map(|(t, _)| *t).collect();
        assert!(tiers.contains(&"L2"), "rebuild MUST reach the L2 tier — diaries live there");
        assert!(tiers.contains(&"L1"), "rebuild must still reach the open tier");
        assert_eq!(tiers, vec!["L1", "L2"], "exactly the readable tiers, in order");
        // L3 is locked, so it is absent rather than skipped. Unlocking is what
        // makes it a rebuild target; nothing here should special-case it.
        assert!(!tiers.contains(&"L3"), "a locked tier is not reachable, so not a target");
    }

    /// The status wire surface must SAY which tiers its pending-count covers.
    ///
    /// `reindex_pending` is summed over readable tiers only. With L3 locked, a
    /// naive report drops L3 entirely, and "reindex_pending: 0" then reads as
    /// "all clean" when the truth is "L1/L2 clean, L3 not looked at." The fix
    /// carries tier_states() into the payload so a locked tier appears as
    /// "locked": present, unmeasured, impossible to mistake for a zero.
    ///
    /// The assertion is the CONTRAST: L1/L2 present AND open, L3 present AND
    /// locked, with no numeric count attached to the locked tier.
    #[test]
    fn rebuild_status_names_the_locked_tier_it_could_not_count() {
        let d = tempfile::tempdir().expect("tmp");
        let p = d.path();
        let s = McpServer::open_tiered(
            &p.join("palace.db"), &p.join("v1"),
            "x'1111111111111111111111111111111111111111111111111111111111111111'",
            &p.join("palace_thread.db"), &p.join("v2"),
            "x'2222222222222222222222222222222222222222222222222222222222222222'",
            &p.join("palace_private.db"), &p.join("v3"), test_policy(),
        )
        .expect("tiered");

        let v = s.t_rebuild_status().expect("rebuild status");
        let tiers = v["tiers"].as_array().expect("tiers array present");
        let state_of = |name: &str| -> Option<&str> {
            tiers
                .iter()
                .find(|e| e["tier"] == name)
                .and_then(|e| e["state"].as_str())
        };
        assert_eq!(state_of("L1"), Some("open"), "L1 is readable, report it open");
        assert_eq!(state_of("L2"), Some("open"), "L2 is readable, report it open");
        assert_eq!(
            state_of("L3"),
            Some("locked"),
            "a locked tier must be PRESENT as locked, not dropped the way readable() drops it"
        );

        // Structural absence: the locked tier carries lock state and no quantity.
        let l3 = tiers.iter().find(|e| e["tier"] == "L3").expect("L3 present");
        assert!(l3.get("reindex_pending").is_none(), "no per-tier count may leak for a locked tier");
        assert!(l3.get("chunks_total").is_none(), "no per-tier count may leak for a locked tier");

        // The scope note must say the pending-count is open-tiers-only, and the
        // merged count itself is still reported as before.
        assert!(
            v["reindex_pending_scope"].as_str().unwrap().contains("open tiers"),
            "the report must state its pending-count covers open tiers only"
        );
        assert!(v["reindex_pending"].is_number(), "merged pending count still reported");
    }

    /// Pre-split, rebuild still targets exactly the one store. The tiered fix
    /// must not change single-store behaviour.
    #[test]
    fn rebuild_targets_the_single_store_when_not_split() {
        let (s, _t) = srv();
        let tiers: Vec<&str> = s.rebuild_targets().iter().map(|(t, _)| *t).collect();
        assert_eq!(tiers, vec!["single"]);
    }

    /// There must be **no unlock tool on the MCP surface**, ever.
    ///
    /// This server may be reached through a remote connector, so a passphrase
    /// passed as a tool argument would leave the machine — the exact boundary
    /// the membrane exists to stop things crossing — and would land in
    /// conversation transcripts on disk. Unlocking is a local action, out of
    /// band from this channel. A future maintainer adding `membrane_unlock`
    /// for convenience breaks this test, which is the point.
    #[test]
    fn there_is_no_unlock_tool_on_the_network_surface() {
        for n in McpServer::tool_names() {
            assert!(
                !n.contains("unlock"),
                "{n} would carry a passphrase over the MCP transport"
            );
            assert!(
                !n.contains("passphrase") && !n.contains("_key"),
                "{n} names key material on a network-reachable surface"
            );
        }
    }

    #[test]
    fn add_get_status_roundtrip() {
        let (s, _t) = srv();
        let added = s.handle_tool("add_drawer", &json!({"wing":"w","room":"r","content":"verbatim payload","provenance":"test"})).expect("add");
        let id = added["id"].as_str().expect("id").to_string();
        let got = s.handle_tool("get_drawer", &json!({"id": id})).expect("get");
        assert_eq!(got["content"], "verbatim payload");
        let st = s.handle_tool("status", &json!({})).expect("status");
        assert_eq!(st["total"], 1);
    }

    #[test]
    fn list_wings_and_rooms_enumerate() {
        let (s, _t) = srv();
        s.handle_tool("add_drawer", &json!({"wing":"public","room":"recognition","content":"c","provenance":"t"})).unwrap();
        let w = s.handle_tool("list_wings", &json!({})).expect("wings");
        assert!(w["wings"].as_array().unwrap().iter().any(|x| x=="public"), "list_wings must return the wing: {w}");
        let r = s.handle_tool("list_rooms", &json!({"wing":"public"})).expect("rooms");
        assert!(r["rooms"].as_array().unwrap().iter().any(|x| x=="recognition"), "list_rooms must return the room: {r}");
        let tax = s.handle_tool("get_taxonomy", &json!({})).expect("tax");
        assert!(tax["taxonomy"]["public"].as_array().unwrap().iter().any(|x| x=="recognition"), "taxonomy composes wings+rooms: {tax}");
    }

    #[test]
    fn diary_roundtrips() {
        let (s, _t) = srv();
        s.handle_tool("diary_write", &json!({"agent":"ada","entry":"kept thick"})).expect("write");
        let r = s.handle_tool("diary_read", &json!({"agent":"ada","last_n":5})).expect("read");
        assert_eq!(r["entries"].as_array().unwrap().len(), 1, "diary write/read through the wrap: {r}");
    }

    #[test]
    fn tunnels_and_traverse() {
        let (s, _t) = srv();
        let a = s.handle_tool("add_drawer", &json!({"wing":"w","room":"r","content":"A","provenance":"t"})).unwrap();
        let b = s.handle_tool("add_drawer", &json!({"wing":"w","room":"r","content":"B","provenance":"t"})).unwrap();
        let aid = a["id"].as_str().unwrap().to_string();
        let bid = b["id"].as_str().unwrap().to_string();
        s.handle_tool("create_tunnel", &json!({"from":aid,"to":bid,"relation":"references"})).expect("tunnel");
        let f = s.handle_tool("find_tunnels", &json!({"relation":"references"})).expect("find");
        assert_eq!(f["tunnels"].as_array().unwrap().len(), 1, "find_tunnels through the wrap: {f}");
        let tv = s.handle_tool("traverse", &json!({"drawer":aid,"depth":2})).expect("traverse");
        assert!(!tv["drawers"].as_array().unwrap().is_empty(), "traverse returns reachable set: {tv}");
    }

    /// Defect 1 abort-fix: a graph read must NOT die because one readable tier
    /// lacks the drawer. Before the fix, list_tunnels/traverse fanned with a `?`
    /// that aborted on the first tier whose resolve_endpoint returned NotFound --
    /// so a drawer get_drawer resolves reported "not found" from the graph tools.
    /// And the envelope must state which tiers it consulted, or it is a confident
    /// partial answer in the tool meant to cure exactly that.
    #[test]
    fn graph_reads_tolerate_a_tier_that_lacks_the_drawer_and_state_their_scope() {
        let d = tempfile::tempdir().expect("tmp");
        let p = d.path();
        let s = McpServer::open_tiered(
            &p.join("palace.db"), &p.join("v1"),
            "x'1111111111111111111111111111111111111111111111111111111111111111'",
            &p.join("palace_thread.db"), &p.join("v2"),
            "x'2222222222222222222222222222222222222222222222222222222222222222'",
            &p.join("palace_private.db"), &p.join("v3"), test_policy(),
        )
        .expect("tiered");

        // A drawer in `public` lands in L1 only; L2 does not hold it.
        let added = s
            .handle_tool("add_drawer", &json!({"wing":"public","room":"r","content":"x","provenance":"t"}))
            .expect("add");
        let id = added["id"].as_str().expect("id").to_string();

        // Before the fix these Err'd "not found" because the L2 store aborted the
        // fan. The .expect IS the regression; the envelope assertion is the scope.
        let lt = s
            .handle_tool("list_tunnels", &json!({"drawer": id}))
            .expect("list_tunnels must not abort when a tier lacks the drawer");
        assert!(lt["tiers_consulted"].as_array().unwrap().len() >= 2, "must state consulted tiers: {lt}");

        let tv = s
            .handle_tool("traverse", &json!({"drawer": id, "depth": 2}))
            .expect("traverse must not abort when a tier lacks the drawer");
        assert!(tv["tiers_consulted"].as_array().unwrap().len() >= 2, "traverse must state its scope: {tv}");
    }

    /// The tier-aware walk is transitive and bounded, and it gathers each node's
    /// edges across EVERY readable store: A->B->C reached from A at depth 2,
    /// deduped, terminating. (A true cross-STORE edge — one stored in a
    /// different tier than its node — cannot be built through create_tunnel
    /// today; see the store_for_tunnel finding. The multi-store gather loop is
    /// still exercised, and tunnels_referencing's own test pins the exact-id
    /// edge gather that the walk relies on.)
    #[test]
    fn traverse_multi_hop_is_transitive_and_bounded() {
        let d = tempfile::tempdir().expect("tmp");
        let p = d.path();
        let s = McpServer::open_tiered(
            &p.join("palace.db"), &p.join("v1"),
            "x'1111111111111111111111111111111111111111111111111111111111111111'",
            &p.join("palace_thread.db"), &p.join("v2"),
            "x'2222222222222222222222222222222222222222222222222222222222222222'",
            &p.join("palace_private.db"), &p.join("v3"), test_policy(),
        )
        .expect("tiered");

        let add = |c: &str| {
            s.handle_tool("add_drawer", &json!({"wing":"agent_ada","room":"personal-diary","content":c,"provenance":"t"}))
                .expect("add")["id"].as_str().unwrap().to_string()
        };
        let a = add("A");
        let b = add("B");
        let c = add("C");
        s.handle_tool("create_tunnel", &json!({"from":a,"to":b,"relation":"next"})).expect("a->b");
        s.handle_tool("create_tunnel", &json!({"from":b,"to":c,"relation":"next"})).expect("b->c");

        let tv = s.handle_tool("traverse", &json!({"drawer": a, "depth": 2})).expect("traverse");
        let reached: Vec<String> = tv["drawers"].as_array().unwrap().iter().map(|x| x["id"].as_str().unwrap().to_string()).collect();
        assert!(reached.contains(&b) && reached.contains(&c), "depth-2 walk reaches B and C transitively: {tv}");
        assert!(tv["absent"].as_array().unwrap().is_empty(), "all endpoints resolve; nothing absent: {tv}");
        assert!(tv["tiers_consulted"].as_array().unwrap().len() >= 2, "states its scope across readable tiers: {tv}");
    }

    /// Honesty boundary: a start that resolves in no readable tier is reported
    /// ABSENT in-band ({id, status}), never a silent empty and never an error.
    #[test]
    fn traverse_reports_unresolvable_start_as_absent_not_error() {
        let d = tempfile::tempdir().expect("tmp");
        let p = d.path();
        let s = McpServer::open_tiered(
            &p.join("palace.db"), &p.join("v1"),
            "x'1111111111111111111111111111111111111111111111111111111111111111'",
            &p.join("palace_thread.db"), &p.join("v2"),
            "x'2222222222222222222222222222222222222222222222222222222222222222'",
            &p.join("palace_private.db"), &p.join("v3"), test_policy(),
        )
        .expect("tiered");
        let tv = s
            .handle_tool("traverse", &json!({"drawer": "ffffffff-ffff-ffff-ffff-ffffffffffff", "depth": 2}))
            .expect("an unresolvable start must not error");
        assert!(tv["drawers"].as_array().unwrap().is_empty(), "no drawers for an unresolvable start: {tv}");
        let absent = tv["absent"].as_array().unwrap();
        assert_eq!(absent.len(), 1, "the unresolvable start is reported absent: {tv}");
        assert_eq!(absent[0]["status"], "unresolved", "absent entries carry a uniform status: {tv}");
        assert!(!tv["tiers_consulted"].as_array().unwrap().is_empty(), "still states the tiers it consulted: {tv}");
    }

    #[test]
    fn kg_bitemporal_and_stats() {
        let (s, _t) = srv();
        s.handle_tool("kg_add", &json!({"subject":"Mira","predicate":"lives_in","object":"Dili","valid_from":"2024-01-01T00:00:00Z"})).expect("add");
        s.handle_tool("kg_invalidate", &json!({"subject":"Mira","predicate":"lives_in","object":"Dili","ended":"2025-01-01T00:00:00Z"})).expect("inv");
        let inside = s.handle_tool("kg_query", &json!({"entity":"Mira","as_of":"2024-06-01T00:00:00Z"})).expect("q");
        assert_eq!(inside["triples"].as_array().unwrap().len(), 1);
        let now = s.handle_tool("kg_query", &json!({"entity":"Mira"})).expect("q now");
        assert_eq!(now["triples"].as_array().unwrap().len(), 0);
        let ks = s.handle_tool("kg_stats", &json!({})).expect("kg_stats");
        assert!(ks.get("triples").is_some(), "kg_stats present: {ks}");
    }

    #[test]
    fn meta_tools_never_panic() {
        let (s, _t) = srv();
        for t in ["get_aaak_spec","graph_stats","hook_settings","reconnect","memories_filed_away"] {
            assert!(s.handle_tool(t, &json!({})).is_ok(), "meta tool must return, not panic: {t}");
        }
    }

    #[test]
    fn unknown_tool_and_missing_arg_error() {
        let (s, _t) = srv();
        assert!(s.handle_tool("delete_everything", &json!({})).is_err());
        assert!(s.handle_tool("get_drawer", &json!({})).is_err());
    }

    #[test]
    fn tool_schema_has_no_key_field() {
        let schemas = McpServer::tool_schemas().to_string().to_lowercase();
        for banned in ["\"key\"", "password", "secret", "token"] {
            assert!(!schemas.contains(banned), "tool contract must not expose {banned}");
        }
    }

    #[test]
    fn list_drawers_filters_and_reports_total() {
        let (s, _t) = srv();
        s.handle_tool("add_drawer", &json!({"wing":"agent_ada","room":"work-log","content":"A","provenance":"t"})).unwrap();
        s.handle_tool("add_drawer", &json!({"wing":"agent_ada","room":"work-log","content":"B","provenance":"t"})).unwrap();
        s.handle_tool("add_drawer", &json!({"wing":"public","room":"canon","content":"C","provenance":"t"})).unwrap();

        let all = s.handle_tool("list_drawers", &json!({"limit":50})).expect("all");
        assert_eq!(all["total"], 3, "unfiltered total counts every live drawer: {all}");
        assert_eq!(all["drawers"].as_array().unwrap().len(), 3);

        let ww = s.handle_tool("list_drawers", &json!({"wing":"agent_ada"})).expect("wing");
        assert_eq!(ww["total"], 2, "populated wing returns its TRUE count, not a silent 0: {ww}");

        let wr = s.handle_tool("list_drawers", &json!({"wing":"agent_ada","room":"work-log"})).expect("wing+room");
        assert_eq!(wr["total"], 2, "wing+room filter honored: {wr}");

        let none = s.handle_tool("list_drawers", &json!({"wing":"zzz_nope"})).expect("none");
        assert_eq!(none["total"], 0, "absent wing is an HONEST 0, distinguishable from a populated wing: {none}");

        // Exact regression: a client probing with limit=1 to read `total`. Pre-fix
        // that returned no `total` key -> 0 -> empty list for EVERY query.
        let probe = s.handle_tool("list_drawers", &json!({"wing":"agent_ada","limit":1,"offset":0})).expect("probe");
        assert_eq!(probe["total"], 2, "limit=1 probe still reports the FULL total: {probe}");
        assert_eq!(probe["drawers"].as_array().unwrap().len(), 1, "limit=1 returns one row though total=2");
    }
}
