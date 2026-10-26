//! P4 — closets: a **disposable, locally-derived projection** over drawers.
//!
//! ## The load-bearing design call (brief §3): closets DO NOT sync
//!
//! A closet is a *grouping* of drawers, never a record in its own right. It is
//! derived — entirely — from drawer content that has already synced. Therefore
//! closet generation is deliberately **NOT a CRDT write**:
//!
//! - it writes no `crdt_docs` entry and folds nothing into an Automerge doc;
//! - it stamps no HLC and contributes nothing to `produce_delta`;
//! - each device regenerates its own closets from the drawers it already has.
//!
//! This is the exact inverse of P3's KG writes (which *are* CRDT ops) and it is
//! the correct application of the locked principle: drawers sync, their derived
//! groupings do not. The property that makes non-syncing safe is that a closet
//! can never hold the sole copy of anything — losing the `closets` table is one
//! `rebuild_closets` away from whole, never data loss. The gate
//! `closet_generation_writes_no_crdt_doc` is what keeps that true.
//!
//! **Corollary — authored content is a DRAWER.** the predecessor store exposes a
//! `create_closet_with_content` (authored closet body). Palace deliberately does
//! not: a closet only ever *points* at drawers. Anything a user wants to keep is
//! a drawer, which syncs. Making closets authorable would turn them into a
//! system-of-record and pull them into the sync surface — a decision that is not
//! this layer's to make.
//!
//! ## Path A only
//!
//! Generation is Path A of the four-path chain: regex capitalized-n-gram
//! extraction plus a stoplist. No LLM, no cloud key, works at install on every
//! device (the "design for the local model" lock). Paths B (local Phi-4-mini),
//! C (BYOK cloud), and D (hierarchical chunking) are out of scope here.
//!
//! ## Algorithm
//!
//! 1. Tokenize on whitespace/punctuation, keeping word-internal `'` and `-`.
//! 2. Take runs of 1–3 consecutive capitalized words as entity candidates
//!    (sub-spans emitted too, so "Sarah's mother Linda" yields both names).
//! 3. Drop single-word candidates that are on the STOPLIST (sentence-starters,
//!    weekdays, months — capitalized but not proper nouns).
//! 4. Bucket drawers by entity; each entity seen in >= `min_drawers` drawers
//!    becomes one closet whose content is a pointer-doc of drawer ids.
//!
//! The pointer-doc is capped at [`DEFAULT_CLOSET_CHAR_LIMIT`] chars. The cap is
//! **explicit, never silent**: when it bites, the closet records how many of the
//! bucket's drawers it included versus how many exist, and `rebuild_closets`
//! returns those truncations in its report.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use rusqlite::params;
use sha2::{Digest, Sha256};
use simonides_core::memory::Closet;
use uuid::Uuid;

use crate::{PalaceError, PalaceStore};

type Result<T> = std::result::Result<T, PalaceError>;

/// Closet projection schema. Lives in the sqlcipher DB alongside drawers, so it
/// inherits M2 encryption-at-rest for free — there is deliberately no sidecar
/// file, no plaintext cache, and no separate index on disk.
///
/// Every table here is disposable: dropping all three and calling
/// `rebuild_closets` for each room reproduces them exactly.
pub(crate) const CLOSETS_SCHEMA: &str = r#"
-- A closet: one entity, pointing at the drawers that mention it.
-- `content` is a pointer-doc (`<entity>||→<id>,<id>,...`), never authored prose.
CREATE TABLE IF NOT EXISTS closets (
    id              TEXT PRIMARY KEY,
    wing            TEXT NOT NULL,
    room            TEXT NOT NULL,
    label           TEXT NOT NULL,
    summary         TEXT,
    content         TEXT NOT NULL,
    source_file     TEXT,
    -- Explicit-cap bookkeeping (§4, no silent cap): how many of the bucket's
    -- drawers the pointer-doc actually lists, versus how many were found.
    included_count  INTEGER NOT NULL DEFAULT 0,
    total_count     INTEGER NOT NULL DEFAULT 0,
    content_hash    TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    UNIQUE(wing, room, label)
);
CREATE INDEX IF NOT EXISTS closets_room_idx   ON closets(wing, room);
CREATE INDEX IF NOT EXISTS closets_source_idx ON closets(source_file)
    WHERE source_file IS NOT NULL;

-- Closet → drawer membership. A projection edge, not a CRDT edge (contrast
-- `tunnels`, which are HLC-stamped and DO sync). Cascades so a purged closet
-- takes its membership with it.
CREATE TABLE IF NOT EXISTS closet_drawers (
    closet_id  TEXT NOT NULL REFERENCES closets(id) ON DELETE CASCADE,
    drawer_id  TEXT NOT NULL REFERENCES drawers(id) ON DELETE CASCADE,
    PRIMARY KEY (closet_id, drawer_id)
);
CREATE INDEX IF NOT EXISTS closet_drawers_drawer_idx ON closet_drawers(drawer_id);

-- Per-room staleness ledger. `drawer_fingerprint` is the hash of the live
-- drawers the projection was built from. If the room's drawers have changed
-- since, the fingerprint no longer matches and the projection reports STALE —
-- loudly, never served as authoritative (§4).
CREATE TABLE IF NOT EXISTS closet_projection (
    wing               TEXT NOT NULL,
    room               TEXT NOT NULL,
    drawer_fingerprint TEXT NOT NULL,
    generated_at       TEXT NOT NULL,
    PRIMARY KEY (wing, room)
);
"#;

/// Pointer-doc char cap (locked default). Explicit, not silent: see
/// [`ClosetTruncation`].
pub const DEFAULT_CLOSET_CHAR_LIMIT: usize = 1500;

/// An entity must be mentioned by at least this many drawers to earn a closet.
/// One drawer is not a grouping.
pub const DEFAULT_MIN_DRAWERS_PER_CLOSET: usize = 2;

/// Longest capitalized run considered as one entity candidate.
const MAX_NGRAM: usize = 3;

/// Words that appear capitalized at sentence boundaries but are not proper
/// nouns. Conservative and biased toward inclusion: a false positive is a closet
/// that may still cluster meaningfully; a false negative vanishes entirely.
const STOPLIST: &[&str] = &[
    // Sentence-starter pronouns + determiners
    "The", "This", "That", "These", "Those", "There", "Their", "It", "Its", "He", "She", "They",
    "We", "Our", "His", "Her", "My", "Your", "A", "An", "And", "But", "Or", "If", "When", "Where",
    "Why", "How", "What", "Who", "Whom", "Whose", "Which", "While", "After", "Before", "Until",
    "Although", "Because", "However", "Therefore", "Thus", "So",
    // Weekdays + months — capitalized, rarely meaningful as entity labels
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January",
    "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
    // Common sentence-initial verbs
    "Is", "Was", "Will", "Would", "Could", "Should", "Can", "May", "Might", "Are", "Have", "Has",
    "Had", "Do", "Does", "Did", "Yes", "No", "Maybe", "Perhaps", "Probably",
    // Discourse particles
    "Hi", "Hey", "Hello", "Thanks", "Okay", "Right", "Well", "Now", "Then",
];

// ── Path A extraction ─────────────────────────────────────────────────────────

/// Split on whitespace and punctuation, keeping word-internal `'` and `-` so
/// "Sarah's" and "Al-Wakrah" survive as single tokens.
fn tokenize(text: &str) -> Vec<&str> {
    text.split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\'' && c != '-'))
        .filter(|s| !s.is_empty())
        .collect()
}

fn is_capitalized(word: &str) -> bool {
    word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
}

/// Extract capitalized entity candidates, deduped, in order of first appearance.
/// Deterministic: the same text always yields the same list in the same order.
pub fn extract_capitalized_entities(text: &str) -> Vec<String> {
    let tokens = tokenize(text);
    let mut entities: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let n = tokens.len();
    let mut i = 0;
    while i < n {
        if !is_capitalized(tokens[i]) {
            i += 1;
            continue;
        }
        // Longest run of consecutive capitalized words starting at i.
        let mut j = i + 1;
        while j < n && j - i < MAX_NGRAM && is_capitalized(tokens[j]) {
            j += 1;
        }
        let run = &tokens[i..j];
        // Emit every sub-span, longest first (most specific signal first). The
        // single-word stoplist applies only to length-1 candidates, so a
        // compound like "The Cedar Pod" still yields "Cedar Pod" while bare
        // "The" is dropped.
        for length in (1..=run.len()).rev() {
            for start in 0..=(run.len() - length) {
                let span = run[start..start + length].join(" ");
                if span.is_empty() {
                    continue;
                }
                if length == 1 && STOPLIST.contains(&span.as_str()) {
                    continue;
                }
                if seen.insert(span.clone()) {
                    entities.push(span);
                }
            }
        }
        i = j.max(i + 1);
    }
    entities
}

/// One drawer's extracted entities — the input row to clustering.
#[derive(Debug, Clone)]
pub struct DrawerEntities {
    pub drawer_id: String,
    pub entities: Vec<String>,
}

/// A closet proposed by Path A, before persistence.
#[derive(Debug, Clone, PartialEq)]
pub struct ProposedCloset {
    /// The entity that bucketed these drawers. Becomes the closet label.
    pub entity: String,
    /// Drawer ids actually listed in `packed_content` (post-cap), sorted.
    pub drawer_ids: Vec<String>,
    /// How many drawers mentioned this entity in total. Equals
    /// `drawer_ids.len()` unless the char cap truncated the pointer-doc.
    pub total_drawers: usize,
    /// Pointer-doc body: `<entity>||→<drawer_id>,<drawer_id>,...`
    pub packed_content: String,
}

impl ProposedCloset {
    /// `true` when the char cap dropped drawers from the pointer-doc.
    pub fn is_truncated(&self) -> bool {
        self.drawer_ids.len() < self.total_drawers
    }
}

/// An explicit report that a closet's pointer-doc hit the char cap. Surfaced by
/// [`ClosetRebuildReport`] — the cap is never applied silently (§4).
#[derive(Debug, Clone, PartialEq)]
pub struct ClosetTruncation {
    pub label: String,
    /// Drawers listed in the pointer-doc.
    pub included: usize,
    /// Drawers that mention the entity.
    pub total: usize,
}

impl std::fmt::Display for ClosetTruncation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "closet '{}' truncated at the {}-char cap: {} of {} drawers listed",
            self.label, DEFAULT_CLOSET_CHAR_LIMIT, self.included, self.total
        )
    }
}

/// What a `rebuild_closets` pass did. Loud by construction: the caller can see
/// how many drawers were scanned, how many closets were written, and exactly
/// which ones lost drawers to the char cap.
#[derive(Debug, Clone)]
pub struct ClosetRebuildReport {
    pub wing: String,
    pub room: String,
    pub drawers_scanned: usize,
    pub closets_written: usize,
    /// Non-empty when at least one pointer-doc hit the char cap.
    pub truncations: Vec<ClosetTruncation>,
}

impl std::fmt::Display for ClosetRebuildReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}: scanned {} drawers, wrote {} closets",
            self.wing, self.room, self.drawers_scanned, self.closets_written
        )?;
        if self.truncations.is_empty() {
            write!(f, ", no truncation")
        } else {
            write!(f, ", {} truncated:", self.truncations.len())?;
            for t in &self.truncations {
                write!(f, " [{t}]")?;
            }
            Ok(())
        }
    }
}

/// Bucket drawers by entity; every entity mentioned by >= `min_drawers` drawers
/// becomes one proposed closet.
///
/// `closet_char_limit` caps the pointer-doc. When an entity is referenced by
/// more drawers than fit, the lowest-sorting ids are kept (drawer ids are UUID
/// v7, so that is oldest-first and stable) and the remainder are reported as a
/// truncation rather than dropped in silence.
///
/// Deterministic: same input rows → identical output, in identical order.
pub fn cluster_into_closets(
    indices: &[DrawerEntities],
    min_drawers: usize,
    closet_char_limit: usize,
) -> Vec<ProposedCloset> {
    // Possessive variants cluster as one entity ("Linda" == "Linda's"); the
    // extracted surface form is preserved, only the bucket key is normalized.
    fn normalize_key(entity: &str) -> &str {
        entity
            .strip_suffix("'s")
            .or_else(|| entity.strip_suffix("'S"))
            .or_else(|| entity.strip_suffix("\u{2019}s"))
            .unwrap_or(entity)
    }

    let mut bucket: HashMap<&str, Vec<&str>> = HashMap::new();
    for di in indices {
        for entity in &di.entities {
            bucket
                .entry(normalize_key(entity))
                .or_default()
                .push(di.drawer_id.as_str());
        }
    }

    let mut closets: Vec<ProposedCloset> = Vec::new();
    for (entity, mut drawer_ids) in bucket {
        // A drawer that mentions "Linda" and "Linda's" must count once.
        drawer_ids.sort_unstable();
        drawer_ids.dedup();
        let total_drawers = drawer_ids.len();
        if total_drawers < min_drawers {
            continue;
        }

        let mut packed = format!("{entity}||→");
        let mut kept: Vec<String> = Vec::new();
        for did in &drawer_ids {
            let piece = if kept.is_empty() {
                (*did).to_string()
            } else {
                format!(",{did}")
            };
            if packed.len() + piece.len() > closet_char_limit {
                break;
            }
            packed.push_str(&piece);
            kept.push((*did).to_string());
        }
        // A cap so tight it can't hold the minimum membership yields no closet
        // at all rather than a misleading one-drawer "grouping".
        if kept.len() < min_drawers {
            continue;
        }

        closets.push(ProposedCloset {
            entity: entity.to_string(),
            drawer_ids: kept,
            total_drawers,
            packed_content: packed,
        });
    }

    // Stable, deterministic order: biggest bucket first, alphabetical to break ties.
    closets.sort_by(|a, b| {
        b.total_drawers
            .cmp(&a.total_drawers)
            .then_with(|| a.entity.cmp(&b.entity))
    });
    closets
}

/// Deterministic change-detection hash. Not cryptographic — it only answers
/// "are these the same bytes as last time".
fn content_hash(content: &str) -> String {
    hex::encode(Sha256::digest(content.as_bytes()))
}

// ── Store surface ─────────────────────────────────────────────────────────────

impl PalaceStore {
    /// Run Path A over `drawers` and return the proposed closets. Pure: nothing
    /// is persisted and nothing is synced.
    pub fn generate_closets_path_a(
        &self,
        drawers: &[crate::Drawer],
        min_drawers_per_closet: usize,
        closet_char_limit: usize,
    ) -> Vec<ProposedCloset> {
        let indices: Vec<DrawerEntities> = drawers
            .iter()
            .map(|d| DrawerEntities {
                drawer_id: d.id.clone(),
                entities: extract_capitalized_entities(&d.content),
            })
            .collect();
        cluster_into_closets(&indices, min_drawers_per_closet, closet_char_limit)
    }

    /// The fingerprint of a room's live drawers — what the projection was built
    /// from. Two rooms with the same live drawer content hash the same; any
    /// insert, edit, or tombstone changes it, which is what makes staleness
    /// detectable.
    fn room_drawer_fingerprint(&self, wing: &str, room: &str) -> Result<String> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard
            .prepare(
                "SELECT id, content FROM drawers \
                 WHERE wing = ?1 AND room = ?2 AND state = 'live' ORDER BY id",
            )
            .map_err(|e| PalaceError::Storage(format!("closet fingerprint prepare: {e}")))?;
        let rows = stmt
            .query_map(params![wing, room], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| PalaceError::Storage(format!("closet fingerprint query: {e}")))?;

        let mut hasher = Sha256::new();
        for row in rows {
            let (id, content) = row.map_err(|e| PalaceError::Storage(e.to_string()))?;
            hasher.update(id.as_bytes());
            hasher.update([0u8]);
            hasher.update(content.as_bytes());
            hasher.update([0u8]);
        }
        Ok(hex::encode(hasher.finalize()))
    }

    /// Regenerate the closet projection for one room from its live drawers.
    ///
    /// **Not a CRDT write.** This touches `closets`, `closet_drawers`, and
    /// `closet_projection` only. It writes no `crdt_docs`, stamps no HLC, and
    /// adds nothing to `produce_delta` — closets are derived and stay local
    /// (§3). Deleting all three tables and calling this again reproduces the
    /// identical set, which is the property that makes not-syncing safe.
    ///
    /// Idempotent: running it twice on unchanged drawers is a no-op beyond
    /// rewriting the same rows.
    pub fn rebuild_closets(&self, wing: &str, room: &str) -> Result<ClosetRebuildReport> {
        let drawers: Vec<crate::Drawer> = {
            let guard = self.conn.lock().unwrap();
            let sql = format!(
                "SELECT {cols} FROM drawers \
                 WHERE wing = ?1 AND room = ?2 AND state = 'live' ORDER BY id",
                cols = crate::COLS
            );
            let mut stmt = guard
                .prepare(&sql)
                .map_err(|e| PalaceError::Storage(format!("rebuild_closets prepare: {e}")))?;
            let rows = stmt
                .query_map(params![wing, room], crate::row_to_drawer)
                .map_err(|e| PalaceError::Storage(format!("rebuild_closets query: {e}")))?;
            let mut out = Vec::new();
            for r in rows {
                // `row_to_drawer` returns a nested Result: the outer is rusqlite's,
                // the inner is the decode (unknown state / bad timestamp).
                out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))??);
            }
            out
        };

        let proposed = self.generate_closets_path_a(
            &drawers,
            DEFAULT_MIN_DRAWERS_PER_CLOSET,
            DEFAULT_CLOSET_CHAR_LIMIT,
        );
        let fingerprint = self.room_drawer_fingerprint(wing, room)?;
        let now = Utc::now().to_rfc3339();

        let truncations: Vec<ClosetTruncation> = proposed
            .iter()
            .filter(|p| p.is_truncated())
            .map(|p| ClosetTruncation {
                label: p.entity.clone(),
                included: p.drawer_ids.len(),
                total: p.total_drawers,
            })
            .collect();

        {
            let mut guard = self.conn.lock().unwrap();
            let tx = guard
                .transaction()
                .map_err(|e| PalaceError::Storage(format!("rebuild_closets tx: {e}")))?;

            // Full replace for the room — the projection is disposable, so the
            // cheapest correct rebuild is "throw it away and derive it again".
            tx.execute(
                "DELETE FROM closets WHERE wing = ?1 AND room = ?2",
                params![wing, room],
            )
            .map_err(|e| PalaceError::Storage(format!("rebuild_closets clear: {e}")))?;

            for p in &proposed {
                let id = Uuid::now_v7().to_string();
                tx.execute(
                    "INSERT INTO closets \
                     (id, wing, room, label, summary, content, source_file, \
                      included_count, total_count, content_hash, generated_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, ?7, ?8, ?9, ?10)",
                    params![
                        id,
                        wing,
                        room,
                        p.entity,
                        "Auto-generated via Path A regex extraction",
                        p.packed_content,
                        p.drawer_ids.len() as i64,
                        p.total_drawers as i64,
                        content_hash(&p.packed_content),
                        now,
                    ],
                )
                .map_err(|e| PalaceError::Storage(format!("rebuild_closets insert: {e}")))?;

                for did in &p.drawer_ids {
                    tx.execute(
                        "INSERT OR IGNORE INTO closet_drawers (closet_id, drawer_id) \
                         VALUES (?1, ?2)",
                        params![id, did],
                    )
                    .map_err(|e| {
                        PalaceError::Storage(format!("rebuild_closets membership: {e}"))
                    })?;
                }
            }

            tx.execute(
                "INSERT INTO closet_projection (wing, room, drawer_fingerprint, generated_at) \
                 VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(wing, room) DO UPDATE SET \
                   drawer_fingerprint = excluded.drawer_fingerprint, \
                   generated_at       = excluded.generated_at",
                params![wing, room, fingerprint, now],
            )
            .map_err(|e| PalaceError::Storage(format!("rebuild_closets fingerprint: {e}")))?;

            tx.commit()
                .map_err(|e| PalaceError::Storage(format!("rebuild_closets commit: {e}")))?;
        }

        Ok(ClosetRebuildReport {
            wing: wing.to_string(),
            room: room.to_string(),
            drawers_scanned: drawers.len(),
            closets_written: proposed.len(),
            truncations,
        })
    }

    /// Rooms whose closet projection no longer matches their drawers. Reported
    /// by `health()` as `closets_stale`; each is one `rebuild_closets` from
    /// current. A stale projection is never served as authoritative (§4).
    pub fn stale_closet_rooms(&self) -> Result<Vec<(String, String)>> {
        let recorded: Vec<(String, String, String)> = {
            let guard = self.conn.lock().unwrap();
            let mut stmt = guard
                .prepare("SELECT wing, room, drawer_fingerprint FROM closet_projection")
                .map_err(|e| PalaceError::Storage(format!("stale_closet_rooms prepare: {e}")))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
                })
                .map_err(|e| PalaceError::Storage(format!("stale_closet_rooms query: {e}")))?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
            }
            out
        };

        let mut stale = Vec::new();
        for (wing, room, recorded_fp) in recorded {
            if self.room_drawer_fingerprint(&wing, &room)? != recorded_fp {
                stale.push((wing, room));
            }
        }
        Ok(stale)
    }

    /// Closets in a room, biggest bucket first. Pure read.
    pub fn list_closets_for_room(&self, wing: &str, room: &str) -> Result<Vec<Closet>> {
        self.query_closets(
            "SELECT id, room, label, summary, content, source_file, generated_at \
             FROM closets WHERE wing = ?1 AND room = ?2 \
             ORDER BY total_count DESC, label ASC",
            params![wing, room],
        )
    }

    /// Closets derived from a given source file. Pure read. Path A closets carry
    /// no `source_file`, so this is the surface Paths B–D will populate; it
    /// exists now so the read API matches the predecessor store's.
    pub fn list_closets_for_source(&self, source_file: &str) -> Result<Vec<Closet>> {
        self.query_closets(
            "SELECT id, room, label, summary, content, source_file, generated_at \
             FROM closets WHERE source_file = ?1 ORDER BY total_count DESC, label ASC",
            params![source_file],
        )
    }

    /// Closets that point at `drawer_id` — the reverse membership lookup that
    /// lets `search()` attach context to a hit. Pure read.
    pub fn closets_for_drawer(&self, drawer_id: &str) -> Result<Vec<Closet>> {
        self.query_closets(
            "SELECT c.id, c.room, c.label, c.summary, c.content, c.source_file, c.generated_at \
             FROM closets c \
             JOIN closet_drawers cd ON cd.closet_id = c.id \
             WHERE cd.drawer_id = ?1 \
             ORDER BY c.total_count DESC, c.label ASC",
            params![drawer_id],
        )
    }

    fn query_closets(&self, sql: &str, p: &[&dyn rusqlite::ToSql]) -> Result<Vec<Closet>> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard
            .prepare(sql)
            .map_err(|e| PalaceError::Storage(format!("closet query prepare: {e}")))?;
        let rows = stmt
            .query_map(p, |r| {
                let created: String = r.get(6)?;
                Ok(Closet {
                    id: r.get(0)?,
                    room_id: r.get(1)?,
                    label: r.get(2)?,
                    summary: r.get(3)?,
                    content: r.get(4)?,
                    source_file: r.get(5)?,
                    created_at: chrono::DateTime::parse_from_rfc3339(&created)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            })
            .map_err(|e| PalaceError::Storage(format!("closet query: {e}")))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| PalaceError::Storage(e.to_string()))?);
        }
        Ok(out)
    }

    /// A closet's display context: the most informative of content > summary >
    /// label. `None` when the closet does not exist. Ported from the predecessor store's
    /// `closet_context_for`.
    pub fn closet_context_for(&self, closet_id: &str) -> Result<Option<String>> {
        let guard = self.conn.lock().unwrap();
        let row = guard
            .query_row(
                "SELECT label, summary, content FROM closets WHERE id = ?1",
                params![closet_id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(PalaceError::Storage(format!("closet_context_for: {other}"))),
            })?;

        Ok(row.map(|(label, summary, content)| {
            content
                .filter(|c| !c.is_empty())
                .or(summary.filter(|s| !s.is_empty()))
                .unwrap_or(label)
        }))
    }

    /// Closet context for every closet a drawer belongs to. This is what
    /// `search()` attaches to each hit at assembly time — it changes no score,
    /// it only tells the caller which groupings the hit sits in.
    pub(crate) fn closet_context_for_drawer(&self, drawer_id: &str) -> Result<Vec<String>> {
        Ok(self
            .closets_for_drawer(drawer_id)?
            .into_iter()
            .map(|c| {
                c.content
                    .filter(|s| !s.is_empty())
                    .or(c.summary.filter(|s| !s.is_empty()))
                    .unwrap_or(c.label)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DrawerState, PalaceStore};

    const TEST_KEY: &str = "palace-test-fixed-key-for-all-tests";

    fn store() -> (PalaceStore, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let s = PalaceStore::open_in_memory(tmp.path(), TEST_KEY).expect("open_in_memory");
        (s, tmp)
    }

    /// The corpus used by the clustering gates. Mira+Hanna+Cedar co-occur in the
    /// first two drawers; Linda+Portland in the last two; Sarah appears once.
    fn seed_corpus(s: &PalaceStore) {
        for content in [
            "Mira and Hanna discussed the Cedar project at length.",
            "Hanna flew to Dili to meet Mira about Cedar.",
            "Sarah called from Portland about her mom Linda.",
            "Linda's surgery in Portland went well.",
        ] {
            s.insert("home", "study", content, "test").expect("insert");
        }
    }

    fn labels(closets: &[Closet]) -> Vec<String> {
        closets.iter().map(|c| c.label.clone()).collect()
    }

    // ── Gate 1 (brief §6): Path A extracts and clusters ───────────────────────

    #[test]
    fn closet_path_a_extracts_and_clusters() {
        // Extraction: proper nouns in, stoplist sentence-starters out.
        let entities =
            extract_capitalized_entities("Thursday The meeting with Mira in Dili went well.");
        assert!(entities.contains(&"Mira".to_string()), "got {entities:?}");
        assert!(entities.contains(&"Dili".to_string()), "got {entities:?}");
        assert!(
            !entities.iter().any(|e| e == "Thursday"),
            "weekday must be stoplisted: {entities:?}"
        );
        assert!(
            !entities.iter().any(|e| e == "The"),
            "sentence-starter must be stoplisted: {entities:?}"
        );
        // A compound that merely starts with a stoplist word still clusters.
        let compound = extract_capitalized_entities("Meeting with The Cedar Pod today.");
        assert!(compound.iter().any(|e| e == "Cedar Pod"), "got {compound:?}");
        assert!(!compound.iter().any(|e| e == "The"));

        // Clustering: the known corpus produces exactly the expected closets.
        let (s, _t) = store();
        seed_corpus(&s);
        let report = s.rebuild_closets("home", "study").expect("rebuild");
        assert_eq!(report.drawers_scanned, 4);

        let got = labels(&s.list_closets_for_room("home", "study").expect("list"));
        for expected in ["Mira", "Hanna", "Cedar", "Linda", "Portland"] {
            assert!(
                got.contains(&expected.to_string()),
                "expected a '{expected}' closet, got {got:?}"
            );
        }
        assert!(
            !got.contains(&"Sarah".to_string()),
            "Sarah is in one drawer only — below min_drawers, must not earn a closet: {got:?}"
        );

        // Membership is real: the Mira closet points at both Mira drawers.
        let mira = s
            .list_closets_for_room("home", "study")
            .unwrap()
            .into_iter()
            .find(|c| c.label == "Mira")
            .expect("Mira closet");
        let members: Vec<String> = {
            let g = s.conn.lock().unwrap();
            let mut st = g
                .prepare("SELECT drawer_id FROM closet_drawers WHERE closet_id = ?1")
                .unwrap();
            let rows = st.query_map(params![mira.id], |r| r.get::<_, String>(0)).unwrap();
            rows.map(|r| r.unwrap()).collect()
        };
        assert_eq!(members.len(), 2, "Mira must point at exactly 2 drawers");
        // And the pointer-doc lists those same ids — a closet is a pointer, never a copy.
        let content = mira.content.expect("pointer-doc content");
        for m in &members {
            assert!(content.contains(m), "pointer-doc {content} must list drawer {m}");
        }
    }

    // ── Gate 2 (brief §6): the projection is pure and rebuildable ─────────────

    #[test]
    fn closet_is_pure_projection_rebuildable() {
        let (s, _t) = store();
        seed_corpus(&s);
        s.rebuild_closets("home", "study").expect("first build");

        let before = s.list_closets_for_room("home", "study").expect("list before");
        assert!(!before.is_empty(), "fixture must produce closets");
        let before_shape: Vec<(String, Option<String>)> =
            before.iter().map(|c| (c.label.clone(), c.content.clone())).collect();

        // Nuke the entire projection — every closet table, as if the device had
        // never generated one. This is exactly what a fresh peer sees after sync:
        // it has the drawers, and no closets at all.
        {
            let g = s.conn.lock().unwrap();
            g.execute_batch(
                "DELETE FROM closet_drawers; DELETE FROM closets; DELETE FROM closet_projection;",
            )
            .expect("drop projection");
        }
        assert!(
            s.list_closets_for_room("home", "study").unwrap().is_empty(),
            "projection must actually be gone"
        );

        // One rebuild from the drawers alone reproduces the identical set.
        s.rebuild_closets("home", "study").expect("rebuild");
        let after = s.list_closets_for_room("home", "study").expect("list after");
        let after_shape: Vec<(String, Option<String>)> =
            after.iter().map(|c| (c.label.clone(), c.content.clone())).collect();

        assert_eq!(
            before_shape, after_shape,
            "rebuild must reproduce the identical closet set from the drawers alone — \
             this is the property that makes non-syncing safe"
        );

        // Idempotent: a third pass changes nothing again.
        s.rebuild_closets("home", "study").expect("rebuild again");
        let third: Vec<(String, Option<String>)> = s
            .list_closets_for_room("home", "study")
            .unwrap()
            .iter()
            .map(|c| (c.label.clone(), c.content.clone()))
            .collect();
        assert_eq!(after_shape, third, "rebuild must be idempotent");
    }

    // ── Gate 3 (brief §6, THE load-bearing one): closets are NOT a CRDT write ──

    #[test]
    fn closet_generation_writes_no_crdt_doc() {
        use simonides_core::sync::{Hlc, PeerClock, SyncableStore};

        let (s, _t) = store();
        seed_corpus(&s);

        let snapshot = |s: &PalaceStore| -> (Vec<(String, Vec<u8>)>, i64, i64, i64) {
            let g = s.conn.lock().unwrap();
            let mut st = g
                .prepare("SELECT drawer_id, doc_bytes FROM crdt_docs ORDER BY drawer_id")
                .unwrap();
            let docs: Vec<(String, Vec<u8>)> = st
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))
                .unwrap()
                .map(|r| r.unwrap())
                .collect();
            // The HLC high-water mark across every table that carries one. If
            // closet generation stamped an HLC anywhere, this moves.
            let hlc_tunnels: i64 = g
                .query_row("SELECT COALESCE(MAX(hlc_wall), 0) FROM tunnels", [], |r| r.get(0))
                .unwrap();
            let hlc_triples: i64 = g
                .query_row("SELECT COALESCE(MAX(hlc_wall), 0) FROM triples", [], |r| r.get(0))
                .unwrap();
            let hlc_entities: i64 = g
                .query_row("SELECT COALESCE(MAX(hlc_wall), 0) FROM entities", [], |r| r.get(0))
                .unwrap();
            (docs, hlc_tunnels, hlc_triples, hlc_entities)
        };

        let clock = PeerClock {
            peer_node_id: [0u8; 16],
            last_seen_hlc: Hlc { wall_time: 0, logical: 0, node_id: [0u8; 16] },
        };

        let before = snapshot(&s);
        let delta_before = s.produce_delta(&clock).expect("produce_delta before");

        let report = s.rebuild_closets("home", "study").expect("generate closets");
        assert!(report.closets_written > 0, "the fixture must actually write closets");

        let after = snapshot(&s);
        let delta_after = s.produce_delta(&clock).expect("produce_delta after");

        // 1. Not one byte of crdt_docs moved.
        assert_eq!(
            before.0, after.0,
            "closet generation must write NO crdt_doc — it is a local projection, not a CRDT op"
        );
        // 2. No HLC anywhere advanced.
        assert_eq!(
            (before.1, before.2, before.3),
            (after.1, after.2, after.3),
            "closet generation must stamp NO HLC (tunnels/triples/entities high-water unchanged)"
        );
        // 3. Closets contribute nothing to the sync payload — they do not travel.
        assert_eq!(
            delta_before.payload, delta_after.payload,
            "the sync delta must be byte-identical before and after generating closets — \
             closets DO NOT sync (brief §3, the inverse of P3's KG CRDT write)"
        );

        // 4. And the end-to-end consequence: a fresh peer that merges the delta
        //    gets the drawers and ZERO closets — then rebuilds them itself.
        let (peer, _tp) = store();
        peer.apply_remote_delta(delta_after).expect("merge onto fresh peer");
        assert!(
            peer.list_closets_for_room("home", "study").unwrap().is_empty(),
            "closets must NOT arrive over sync"
        );
        peer.rebuild_closets("home", "study").expect("peer rebuilds locally");
        assert_eq!(
            labels(&peer.list_closets_for_room("home", "study").unwrap()),
            labels(&s.list_closets_for_room("home", "study").unwrap()),
            "the peer must derive the identical closet set from the synced drawers alone"
        );
    }

    // ── Gate 4 (brief §6): the char cap is explicit, never silent ─────────────

    #[test]
    fn closet_content_char_limit_reports_truncation() {
        // 40 drawers all mentioning Hub; a 200-char cap cannot list them all.
        let indices: Vec<DrawerEntities> = (0..40)
            .map(|i| DrawerEntities {
                drawer_id: format!("0199a1b2-c3d4-7e5f-8a9b-{i:012}"),
                entities: vec!["Hub".into()],
            })
            .collect();

        let closets = cluster_into_closets(&indices, DEFAULT_MIN_DRAWERS_PER_CLOSET, 200);
        assert_eq!(closets.len(), 1);
        let hub = &closets[0];

        assert!(hub.packed_content.len() <= 200, "the cap must actually bind");
        assert!(hub.is_truncated(), "40 drawers cannot fit a 200-char pointer-doc");
        // The counts are the report: included vs total, never a silent drop.
        assert_eq!(hub.total_drawers, 40, "total must be the true bucket size");
        assert!(
            hub.drawer_ids.len() < 40,
            "included must be less than total when truncated"
        );
        assert_eq!(
            hub.drawer_ids.len(),
            hub.packed_content.matches('-').count() / 4,
            "every listed id must be counted as included"
        );

        // Under the real cap the same bucket fits — truncation is a cap effect,
        // not a lossy default.
        let full = cluster_into_closets(
            &indices,
            DEFAULT_MIN_DRAWERS_PER_CLOSET,
            DEFAULT_CLOSET_CHAR_LIMIT,
        );
        assert!(!full[0].is_truncated(), "40 short ids fit inside 1500 chars");
        assert_eq!(full[0].drawer_ids.len(), 40);

        // And the store-level report surfaces it against the REAL 1500-char cap.
        // A drawer id is a 36-char UUID plus a separator, so ~40 ids is all a
        // 1500-char pointer-doc holds; 80 drawers overflows it decisively.
        const OVERFLOW_DRAWERS: usize = 80;
        let (s, _t) = store();
        for _ in 0..OVERFLOW_DRAWERS {
            s.insert("home", "big", "Hub coordination notes.", "test")
                .expect("insert");
        }
        let report = s.rebuild_closets("home", "big").expect("rebuild");
        assert!(
            !report.truncations.is_empty(),
            "{OVERFLOW_DRAWERS} UUID-sized ids exceed the {DEFAULT_CLOSET_CHAR_LIMIT}-char cap; \
             the report must say so rather than dropping drawers in silence"
        );
        let t = &report.truncations[0];
        assert_eq!(t.label, "Hub");
        assert_eq!(t.total, OVERFLOW_DRAWERS, "report must state the true bucket size");
        assert!(t.included < t.total, "report must state how many were included");
        assert!(t.included > 0, "the cap must still list what it can");
        assert!(
            report.to_string().contains("truncated"),
            "the report's Display must be loud about it: {report}"
        );

        // The persisted row carries the same counts — the cap is durable
        // bookkeeping, not a transient log line a caller can miss.
        let (included, total): (i64, i64) = {
            let g = s.conn.lock().unwrap();
            g.query_row(
                "SELECT included_count, total_count FROM closets WHERE label = 'Hub'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .expect("read counts")
        };
        assert_eq!(total, OVERFLOW_DRAWERS as i64);
        assert_eq!(included, t.included as i64);
        assert!(included < total);
    }

    // ── Gate 5 (brief §6): search attaches closet context ─────────────────────

    #[test]
    fn search_attaches_closet_context_when_present() {
        let (s, _t) = store();
        seed_corpus(&s);

        // Before generation there is no projection, so hits carry no context —
        // and search still works (the closet leg is additive, never required).
        match s.search("Cedar project", 10).expect("search pre-build") {
            crate::retrieval::SearchResult::Hits(hits) => {
                assert!(!hits.is_empty(), "the corpus must match");
                assert!(
                    hits.iter().all(|h| h.closet_context.is_empty()),
                    "no closets generated yet — context must be empty, not fabricated"
                );
            }
            crate::retrieval::SearchResult::Empty { query } => {
                panic!("expected hits for {query}")
            }
        }

        s.rebuild_closets("home", "study").expect("rebuild");

        match s.search("Cedar project", 10).expect("search post-build") {
            crate::retrieval::SearchResult::Hits(hits) => {
                let top = hits.first().expect("at least one hit");
                assert_eq!(top.state, DrawerState::Live, "Deref to Drawer still works");
                assert!(
                    !top.closet_context.is_empty(),
                    "a drawer in a closet must carry its closet context at assembly time"
                );
                // The context is the closet's pointer-doc, and it names the hit.
                assert!(
                    top.closet_context.iter().any(|c| c.contains(&top.drawer.id)),
                    "attached context must be a closet that actually points at this drawer: {:?}",
                    top.closet_context
                );
                assert!(
                    top.closet_context.iter().any(|c| c.starts_with("Cedar")
                        || c.starts_with("Mira")
                        || c.starts_with("Hanna")),
                    "context must come from the entity closets this drawer sits in: {:?}",
                    top.closet_context
                );
            }
            crate::retrieval::SearchResult::Empty { query } => {
                panic!("expected hits for {query}")
            }
        }

        // closet_context_for resolves a closet by id; an unknown id is None,
        // never a fabricated string.
        let mira = s
            .list_closets_for_room("home", "study")
            .unwrap()
            .into_iter()
            .find(|c| c.label == "Mira")
            .expect("Mira closet");
        let ctx = s.closet_context_for(&mira.id).expect("context").expect("some");
        assert!(ctx.starts_with("Mira||→"), "got {ctx}");
        assert!(
            s.closet_context_for("no-such-closet").expect("miss").is_none(),
            "an unknown closet id must be None"
        );
    }

    // ── Reliability: staleness is loud (brief §4) ─────────────────────────────

    #[test]
    fn closet_projection_staleness_is_loud() {
        let (s, _t) = store();
        seed_corpus(&s);

        // Fresh build → not stale.
        s.rebuild_closets("home", "study").expect("rebuild");
        assert!(s.stale_closet_rooms().unwrap().is_empty());
        assert_eq!(s.health().unwrap().closets_stale, 0);
        assert!(s.health().unwrap().closets > 0, "health must count closets");

        // A new drawer changes the room's drawers → the projection is now stale,
        // and health says so rather than serving it as authoritative.
        s.insert("home", "study", "Mira met Hanna in Dili again.", "test")
            .expect("insert");
        assert_eq!(
            s.stale_closet_rooms().unwrap(),
            vec![("home".to_string(), "study".to_string())],
            "an inserted drawer must make the room's closet projection stale"
        );
        assert_eq!(s.health().unwrap().closets_stale, 1);
        assert!(
            s.health().unwrap().to_string().contains("closets_stale=1"),
            "health Display must surface staleness: {}",
            s.health().unwrap()
        );

        // Rebuild clears it.
        s.rebuild_closets("home", "study").expect("rebuild");
        assert!(s.stale_closet_rooms().unwrap().is_empty());
        assert_eq!(s.health().unwrap().closets_stale, 0);

        // An edit to an existing drawer is caught too (content is fingerprinted,
        // not just the row count).
        let d = s.list(1).unwrap().into_iter().next().expect("a drawer");
        s.update_content(&d.id, "Entirely different content about Wren.")
            .expect("edit");
        assert_eq!(s.health().unwrap().closets_stale, 1, "an edit must be caught");
    }

    #[test]
    fn tombstoned_drawers_leave_the_projection_on_rebuild() {
        let (s, _t) = store();
        seed_corpus(&s);
        s.rebuild_closets("home", "study").expect("rebuild");
        assert!(labels(&s.list_closets_for_room("home", "study").unwrap())
            .contains(&"Portland".to_string()));

        // Retire one of the two Portland drawers; Portland drops below
        // min_drawers and its closet disappears on the next rebuild.
        let portland = s
            .list(10)
            .unwrap()
            .into_iter()
            .find(|d| d.content.contains("Linda's surgery"))
            .expect("the surgery drawer");
        s.tombstone(&portland.id).expect("tombstone");
        s.rebuild_closets("home", "study").expect("rebuild");

        let got = labels(&s.list_closets_for_room("home", "study").unwrap());
        assert!(
            !got.contains(&"Portland".to_string()),
            "a tombstoned drawer must leave the projection: {got:?}"
        );
    }

    #[test]
    fn list_closets_for_source_is_empty_for_path_a() {
        // Path A closets are entity-derived, not file-derived: they carry no
        // source_file, so the by-source read is correctly empty (never a
        // mislabelled match).
        let (s, _t) = store();
        seed_corpus(&s);
        s.rebuild_closets("home", "study").expect("rebuild");
        assert!(s.list_closets_for_source("notes.md").unwrap().is_empty());
    }

    #[test]
    fn rebuild_is_scoped_to_one_room() {
        let (s, _t) = store();
        seed_corpus(&s);
        s.insert("home", "other", "Mira and Hanna again.", "test").unwrap();
        s.insert("home", "other", "Hanna and Mira once more.", "test").unwrap();

        s.rebuild_closets("home", "study").expect("rebuild study");
        assert!(
            s.list_closets_for_room("home", "other").unwrap().is_empty(),
            "rebuilding one room must not build another"
        );
        s.rebuild_closets("home", "other").expect("rebuild other");
        assert!(!s.list_closets_for_room("home", "other").unwrap().is_empty());
        // And rebuilding `other` did not disturb `study`.
        assert!(labels(&s.list_closets_for_room("home", "study").unwrap())
            .contains(&"Mira".to_string()));
    }

    #[test]
    fn empty_room_rebuild_is_a_clean_no_op() {
        let (s, _t) = store();
        let report = s.rebuild_closets("home", "nothing-here").expect("rebuild empty");
        assert_eq!(report.drawers_scanned, 0);
        assert_eq!(report.closets_written, 0);
        assert!(report.truncations.is_empty());
        assert!(s.list_closets_for_room("home", "nothing-here").unwrap().is_empty());
    }

    #[test]
    fn extraction_is_deterministic() {
        let text = "Mira met Hanna in Dili to discuss Cedar with Wren.";
        assert_eq!(extract_capitalized_entities(text), extract_capitalized_entities(text));
        assert!(extract_capitalized_entities("").is_empty());
        assert!(extract_capitalized_entities("nothing here is capitalized.").is_empty());
    }

    #[test]
    fn possessive_variants_cluster_as_one_entity() {
        let indices = vec![
            DrawerEntities { drawer_id: "d1".into(), entities: vec!["Linda".into()] },
            DrawerEntities { drawer_id: "d2".into(), entities: vec!["Linda's".into()] },
        ];
        let closets = cluster_into_closets(&indices, 2, DEFAULT_CLOSET_CHAR_LIMIT);
        assert_eq!(closets.len(), 1, "Linda and Linda's are one entity");
        assert_eq!(closets[0].entity, "Linda");
        assert_eq!(closets[0].total_drawers, 2);
    }

    #[test]
    fn a_drawer_naming_an_entity_twice_counts_once() {
        // The bucket must be a set of drawers, not a bag of mentions — otherwise
        // one chatty drawer could mint a closet on its own.
        let indices = vec![DrawerEntities {
            drawer_id: "d1".into(),
            entities: vec!["Mira".into(), "Mira's".into()],
        }];
        assert!(
            cluster_into_closets(&indices, 2, DEFAULT_CLOSET_CHAR_LIMIT).is_empty(),
            "one drawer mentioning an entity twice is still one drawer"
        );
    }
}
