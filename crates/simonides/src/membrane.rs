//! The membrane — structural L1/L2 separation via two keyed databases.
//!
//! # The property
//!
//! > **An outer-tier session is CRYPTOGRAPHICALLY INCAPABLE of reading an
//! > inner-tier drawer.**
//!
//! Incapable, not forbidden. That distinction is the entire module. A forbidden
//! read is a check that ran and said no — one inverted boolean, one refactored
//! query, one forgotten call site away from returning the row. An incapable
//! read is a query against data that is **not present in the connection** and
//! could not be decrypted by the key in hand even if it were.
//!
//! Concretely: L2 drawers live in a *separate sqlcipher file* under a
//! *separate key*. When the membrane is closed that file is not `ATTACH`ed.
//! The rows are **absent**, not filtered. `NotFound` is structural.
//!
//! # Why not an envelope over L2 content in one database
//!
//! Rejected deliberately. Sealing only the *content* of L2 rows
//! leaves the rows themselves in the open database: `drawer_id`, `wing`,
//! `room`, timestamps and ciphertext length all readable by an outer-tier
//! session. That discloses the existence, volume, timing and wing-distribution
//! of private material — an attacker learns a private wing holds N drawers written
//! on specific dates without decrypting a byte.
//!
//! It also makes every enumeration path a permanent audit surface — `list`,
//! counts, search, KG traversal, each must remember to filter, forever. That is
//! policy wearing a cryptography costume.
//!
//! # Session semantics — a vault, not a turnstile
//!
//! The passphrase is entered **once** at
//! `membrane_open`; the private database stays attached for the working
//! session; there is **no re-authentication per drawer**. It closes on an
//! explicit `membrane_close` **or** after an idle timeout — both, 60 minutes.
//!
//! Both paths converge on the same teardown (`DETACH`, then zeroize), so there
//! is one code path to get right and one to test. They differ only in what
//! fires them.
//!
//! # Key custody
//!
//! `K_OPEN` is machine-bound and resident — the open tier must come up
//! unattended after a reboot. `K_PRIV` is **never resident**: a random Data Key
//! wrapped under an Argon2id passphrase-derived Master Key, with a second
//! independent keyslot wrapping the *same* DK under a recovery code. Either
//! opens the vault; neither reveals the other.
//!
//! A second keyslot buys **availability, never strength** — the vault is as
//! strong as its weakest opener. The recovery code is machine-generated at 128
//! bits, so the human-memorised passphrase is the weak slot by construction.
//! That is why the Argon2id cost parameters on the passphrase slot are
//! load-bearing rather than ceremonial.
//!
//! # Custody
//!
//! Keep the passphrase and the recovery code **off this machine**, somewhere a
//! person you trust can reach them. A vault that dies with its only keyholder
//! fails the one job a memory store has. That trades a little confidentiality
//! for availability, deliberately: **a vault nobody can ever open is a shredder
//! with extra steps.**

use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use simonides_crypto::key_primitives::{aead_unwrap, aead_wrap, argon2id_derive, Argon2Params};
use zeroize::Zeroizing;

use crate::{PalaceError, Result};

/// Schema name the private database is attached under.
pub const PRIV_SCHEMA: &str = "priv";

/// AAD for the passphrase keyslot wrap. Domain-separates the two slots so a
/// blob from one can never be unwrapped as the other.
const AAD_PASS: &[u8] = b"simonides-membrane-v1-keyslot-passphrase";
/// AAD for the recovery-code keyslot wrap.
const AAD_RECOVERY: &[u8] = b"simonides-membrane-v1-keyslot-recovery";

// ---------------------------------------------------------------------------
// Clock — injectable so the idle-timeout test does not sleep for an hour
// ---------------------------------------------------------------------------

/// Time source for membrane session expiry.
///
/// Injectable because the locked idle timeout is **60 minutes**. A test that
/// genuinely waits an hour is a test nobody runs, and a security property
/// nobody exercises is a security property nobody has. `TestClock` advances
/// manually so expiry is asserted in microseconds.
pub trait Clock: Send + Sync {
    /// Monotonic-ish "now" in whole seconds since an arbitrary epoch.
    fn now_secs(&self) -> u64;
}

/// Wall-clock source used in production.
#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_secs(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Manually-advanced clock for tests.
#[derive(Debug, Default)]
pub struct TestClock {
    secs: std::sync::atomic::AtomicU64,
}

impl TestClock {
    pub fn new(start: u64) -> Self {
        Self { secs: std::sync::atomic::AtomicU64::new(start) }
    }
    /// Jump forward. Used to cross the idle threshold without sleeping.
    pub fn advance(&self, d: Duration) {
        self.secs.fetch_add(d.as_secs(), std::sync::atomic::Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn now_secs(&self) -> u64 {
        self.secs.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Membrane session policy.
///
/// Defaults: explicit close **and**
/// idle auto-lock at 60 minutes, no hard ceiling. Tunable so a policy change is
/// a setting rather than a rebuild.
#[derive(Debug, Clone)]
pub struct MembraneConfig {
    /// Auto-lock after this much inactivity. `None` = explicit close only.
    pub idle_timeout: Option<Duration>,
    /// Ceiling regardless of activity. `None` = a continuously busy session
    /// never re-locks. Deliberate: that is correct vault behaviour, and the
    /// knob exists so a ceiling can be added later without a rebuild.
    pub max_session: Option<Duration>,
    /// Does activity reset the idle clock?
    pub renew_on_activity: bool,
}

impl Default for MembraneConfig {
    fn default() -> Self {
        Self {
            idle_timeout: Some(Duration::from_secs(60 * 60)), // 60 minutes
            max_session: None,
            renew_on_activity: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Keyslots
// ---------------------------------------------------------------------------

/// On-disk keyslot record, stored beside the private database.
///
/// Holds **only ciphertext and public parameters**. Both wrapped blobs are
/// inert without their opener, which is what makes it safe to keep them
/// together. Neither the passphrase nor the recovery code is stored anywhere,
/// by design — there is no escrow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembraneKeyslots {
    /// Salt for the passphrase slot's Argon2id derivation.
    pub salt_pass: Vec<u8>,
    /// DK sealed under the passphrase-derived MK.
    pub wrapped_dk_pass: Vec<u8>,
    /// Salt for the recovery-code slot.
    pub salt_recovery: Vec<u8>,
    /// The same DK sealed under the recovery-code-derived MK.
    pub wrapped_dk_recovery: Vec<u8>,
    /// Argon2id parameters actually used. Read from the record rather than a
    /// global constant so a re-enrollment can raise cost without breaking
    /// existing vaults.
    pub params: Argon2Params,
}

impl MembraneKeyslots {
    /// Seal one Data Key under two independent openers.
    ///
    /// **One DK wrapped twice — not two DKs.** Either slot unwraps the same
    /// key, so both open the same data and neither reveals the other. Same
    /// shape as a LUKS keyslot or a password-manager recovery kit.
    pub fn seal(
        dk: &[u8; 32],
        passphrase: &[u8],
        recovery_code: &[u8],
        params: &Argon2Params,
    ) -> Result<Self> {
        let salt_pass = random_salt();
        let salt_recovery = random_salt();

        let mk_pass = argon2id_derive(passphrase, &salt_pass, params).map_err(kerr)?;
        let mk_recovery = argon2id_derive(recovery_code, &salt_recovery, params).map_err(kerr)?;

        let wrapped_dk_pass = aead_wrap(&mk_pass, dk, AAD_PASS).map_err(kerr)?;
        let wrapped_dk_recovery = aead_wrap(&mk_recovery, dk, AAD_RECOVERY).map_err(kerr)?;

        Ok(Self {
            salt_pass,
            wrapped_dk_pass,
            salt_recovery,
            wrapped_dk_recovery,
            params: params.clone(),
        })
    }

    /// Recover the DK from the passphrase slot. Wrong passphrase → `Err`, and
    /// crucially **no attach is attempted** by the caller on `Err`.
    pub fn unwrap_with_passphrase(&self, passphrase: &[u8]) -> Result<Zeroizing<[u8; 32]>> {
        let mk = argon2id_derive(passphrase, &self.salt_pass, &self.params).map_err(kerr)?;
        aead_unwrap(&mk, &self.wrapped_dk_pass, AAD_PASS).map_err(kerr)
    }

    /// Recover the DK from the recovery-code slot.
    pub fn unwrap_with_recovery(&self, recovery_code: &[u8]) -> Result<Zeroizing<[u8; 32]>> {
        let mk = argon2id_derive(recovery_code, &self.salt_recovery, &self.params).map_err(kerr)?;
        aead_unwrap(&mk, &self.wrapped_dk_recovery, AAD_RECOVERY).map_err(kerr)
    }

    pub fn write_to(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| PalaceError::Storage(format!("keyslots encode: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| PalaceError::Storage(format!("keyslots write {}: {e}", path.display())))
    }

    pub fn read_from(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| PalaceError::Storage(format!("keyslots read {}: {e}", path.display())))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| PalaceError::Storage(format!("keyslots decode: {e}")))
    }
}

fn random_salt() -> Vec<u8> {
    use rand::RngCore;
    let mut s = vec![0u8; 16];
    rand::thread_rng().fill_bytes(&mut s);
    s
}

fn kerr(e: simonides_core::SimonidesError) -> PalaceError {
    PalaceError::Storage(format!("membrane key: {e}"))
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/// Live membrane session. Absent (`None`) means the membrane is closed and the
/// private database is not attached.
#[derive(Debug, Clone)]
pub struct MembraneSession {
    pub opened_at_secs: u64,
    pub last_activity_secs: u64,
    pub priv_path: PathBuf,
}

impl MembraneSession {
    /// Has this session expired under `cfg` as of `now`?
    pub fn is_expired(&self, cfg: &MembraneConfig, now: u64) -> bool {
        if let Some(max) = cfg.max_session {
            if now.saturating_sub(self.opened_at_secs) >= max.as_secs() {
                return true;
            }
        }
        if let Some(idle) = cfg.idle_timeout {
            let since = if cfg.renew_on_activity {
                now.saturating_sub(self.last_activity_secs)
            } else {
                now.saturating_sub(self.opened_at_secs)
            };
            if since >= idle.as_secs() {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Attach / detach
// ---------------------------------------------------------------------------

/// Render a 32-byte key in sqlcipher's raw-key form (`x'<hex>'`) so sqlcipher
/// uses the bytes directly instead of running its own KDF over them. The DK is
/// already a uniformly-random 256-bit key; re-deriving would be pointless work
/// and would make the two slots disagree about what the key *is*.
pub fn raw_key_literal(dk: &[u8; 32]) -> Zeroizing<String> {
    Zeroizing::new(format!("x'{}'", hex::encode(dk)))
}

/// `ATTACH` the private database under `PRIV_SCHEMA` using `dk`.
///
/// On a wrong key this returns `Err` and **nothing is attached** — the failure
/// is at the encryption boundary, not deferred to a later confusing read.
pub fn attach_private(conn: &Connection, priv_path: &Path, dk: &[u8; 32]) -> Result<()> {
    let key = raw_key_literal(dk);
    conn.execute(
        &format!("ATTACH DATABASE ?1 AS {PRIV_SCHEMA} KEY ?2"),
        rusqlite::params![priv_path.to_string_lossy().as_ref(), key.as_str()],
    )
    .map_err(|e| PalaceError::Storage(format!("membrane attach: {e}")))?;

    // Force a real read through the attached schema. sqlcipher does not verify
    // the key at ATTACH time — it verifies on first page decrypt. Without this
    // probe a wrong key would appear to succeed here and fail later somewhere
    // confusing, which is the exact silent-deferral shape the eager-validation
    // doctrine prohibits.
    let probe: std::result::Result<i64, _> = conn.query_row(
        &format!("SELECT COUNT(*) FROM {PRIV_SCHEMA}.sqlite_master"),
        [],
        |r| r.get(0),
    );
    if let Err(e) = probe {
        let _ = conn.execute_batch(&format!("DETACH DATABASE {PRIV_SCHEMA}"));
        return Err(PalaceError::Storage(format!(
            "membrane attach: key rejected at first decrypt: {e}"
        )));
    }
    Ok(())
}

/// `DETACH` the private database. Idempotent-ish: detaching when not attached
/// is reported by sqlite as an error, which callers may ignore.
pub fn detach_private(conn: &Connection) -> Result<()> {
    conn.execute_batch(&format!("DETACH DATABASE {PRIV_SCHEMA}"))
        .map_err(|e| PalaceError::Storage(format!("membrane detach: {e}")))
}

/// Is the private schema currently attached to this connection?
///
/// Asks sqlite's own `pragma_database_list` rather than trusting a bool we
/// maintain — the point of the design is that attachment is ground truth.
pub fn is_attached(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT COUNT(*) FROM pragma_database_list WHERE name = ?1",
        [PRIV_SCHEMA],
        |r| r.get::<_, i64>(0),
    )
    .map(|n| n > 0)
    .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tiering — which (wing, room) belongs on which side of the membrane
// ---------------------------------------------------------------------------

/// The label on drawers in a palace with tiers turned off. See [`TierPolicy::untiered`].
pub const UNTIERED: &str = "single";

/// Restrictiveness ordering: `L1` < `L2` < `L3`.
///
/// Exists so that "is this a downgrade?" is a comparison rather than a pair of
/// hardcoded string equalities. With two tiers `stored == "L2" && computed ==
/// "L1"` was adequate; with three it silently misses `L3 -> L1` and `L3 -> L2`,
/// which are the *worst* downgrades. Any tool that reconciles tags must compare
/// ranks, never specific pairs.
///
/// Unknown input ranks highest — an unrecognised tier is treated as maximally
/// restrictive rather than waved through.
pub fn tier_rank(t: &str) -> u8 {
    match t {
        "L1" => 1,
        "L2" => 2,
        "L3" => 3,
        _ => u8::MAX,
    }
}

/// Whether a tier may be published by a network-facing serving layer.
///
/// **L1 only.** This is the whole of the network-exposure rule, in one place,
/// so no serving-layer code has to re-derive it. L2 and L3 differ in local key
/// custody, not in network reachability: neither is ever served.
pub fn is_network_servable(t: &str) -> bool {
    t == "L1"
}

/// Marker appended to provenance so a drawer carries its tier with it.
pub const TIER_TAG_PREFIX: &str = "tier=";

/// Read the `|tier=Lx` marker out of a provenance string.
///
/// `None` means **absent**, and callers must treat that as a halt rather than a
/// default. An untagged drawer is one whose tier nobody decided; guessing is
/// precisely the decision that must not be made silently.
pub fn tier_from_provenance(provenance: &str) -> Option<&str> {
    provenance.rsplit('|').find_map(|seg| seg.strip_prefix(TIER_TAG_PREFIX))
}


// ---------------------------------------------------------------------------
// Tier policy - user-configurable routing of (wing, room) to a tier
// ---------------------------------------------------------------------------

fn parse_tier(s: &str) -> std::result::Result<&'static str, String> {
    match s.trim().to_ascii_uppercase().as_str() {
        "L1" => Ok("L1"),
        "L2" => Ok("L2"),
        "L3" => Ok("L3"),
        _ => Err(format!("unknown tier '{s}' (expected L1, L2 or L3)")),
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    fn into_lower_vec(self) -> Vec<String> {
        match self {
            OneOrMany::One(s) => vec![s.to_ascii_lowercase()],
            OneOrMany::Many(v) => v.into_iter().map(|s| s.to_ascii_lowercase()).collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleFile {
    wing: Option<OneOrMany>,
    wing_prefix: Option<String>,
    room: Option<OneOrMany>,
    room_prefix: Option<String>,
    tier: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyFile {
    default: Option<String>,
    #[serde(default, rename = "rule")]
    rules: Vec<RuleFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TierRule {
    wings: Option<Vec<String>>,
    wing_prefix: Option<String>,
    rooms: Option<Vec<String>>,
    room_prefix: Option<String>,
    tier: &'static str,
}

impl TierRule {
    /// Every matcher that is set must match; unset matchers are ignored.
    fn matches(&self, wing: &str, room: &str) -> bool {
        self.wings.as_ref().is_none_or(|ws| ws.iter().any(|w| w == wing))
            && self.wing_prefix.as_ref().is_none_or(|p| wing.starts_with(p.as_str()))
            && self.rooms.as_ref().is_none_or(|rs| rs.iter().any(|r| r == room))
            && self.room_prefix.as_ref().is_none_or(|p| room.starts_with(p.as_str()))
    }
}

/// Which tier each `(wing, room)` belongs to.
///
/// An ordered list of rules plus a default. Matching is case-insensitive and the
/// **first matching rule wins**, so put the most specific rules first (a `vault`
/// room rule above any wing rule, for example). Anything no rule matches gets the
/// default, which is `L3` unless the policy says otherwise: a misfiled private
/// drawer in an open tier has already been exposed, while an over-sealed one is
/// only a nuisance.
///
/// Policy file format (TOML):
///
/// ```toml
/// default = "L3"                 # optional; L3 if omitted
///
/// [[rule]]
/// room = "vault"                 # a `vault` room in any wing is always sealed
/// tier = "L3"
///
/// [[rule]]
/// room = ["diary", "letters"]    # a string or a list
/// tier = "L2"
///
/// [[rule]]
/// wing_prefix = "public_"        # also: wing, room_prefix
/// tier = "L1"
/// ```
///
/// Loading is strict: unknown keys, unknown tiers, empty lists or prefixes, and
/// rules with no matcher are all rejected, because each of those would otherwise
/// route data somewhere the author did not intend without saying so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierPolicy {
    rules: Vec<TierRule>,
    default: &'static str,
}

impl Default for TierPolicy {
    fn default() -> Self {
        Self::fail_safe()
    }
}

impl TierPolicy {
    /// No rules; everything is `L3`. The default for a store that was given no policy.
    pub fn fail_safe() -> Self {
        Self { rules: Vec::new(), default: "L3" }
    }

    /// For a palace with tiers turned off: every drawer is labelled
    /// [`UNTIERED`] (`tier=single`).
    ///
    /// Labelling everyday notes `L3` there would tell a reader they had gone
    /// into the locked vault. `single` says what is true: there is one store.
    /// It is not a tier: [`tier_rank`] treats it as maximally restrictive and
    /// [`is_network_servable`] refuses it, so if this palace is ever merged into
    /// a tiered one, nothing is exposed until a policy re-labels it.
    pub fn untiered() -> Self {
        Self { rules: Vec::new(), default: UNTIERED }
    }

    /// Parse a policy from TOML source.
    pub fn from_toml_str(src: &str) -> Result<Self> {
        let bad = |msg: String| PalaceError::InvalidConfig(format!("tier policy: {msg}"));
        let file: PolicyFile = toml::from_str(src).map_err(|e| bad(e.to_string()))?;
        let default = match file.default.as_deref() {
            None => "L3",
            Some(s) => parse_tier(s).map_err(|e| bad(format!("default: {e}")))?,
        };
        let mut rules = Vec::with_capacity(file.rules.len());
        for (i, r) in file.rules.into_iter().enumerate() {
            let n = i + 1;
            if r.wing.is_none() && r.wing_prefix.is_none() && r.room.is_none() && r.room_prefix.is_none() {
                return Err(bad(format!(
                    "rule {n} has no wing, wing_prefix, room or room_prefix, so it would match \
                     everything; use `default` for that"
                )));
            }
            if r.wing_prefix.as_deref() == Some("") || r.room_prefix.as_deref() == Some("") {
                return Err(bad(format!("rule {n}: an empty prefix matches everything")));
            }
            let tier = parse_tier(&r.tier).map_err(|e| bad(format!("rule {n}: {e}")))?;
            let wings = r.wing.map(OneOrMany::into_lower_vec);
            let rooms = r.room.map(OneOrMany::into_lower_vec);
            if wings.as_ref().is_some_and(|v| v.is_empty()) || rooms.as_ref().is_some_and(|v| v.is_empty()) {
                return Err(bad(format!("rule {n}: an empty list matches nothing")));
            }
            rules.push(TierRule {
                wings,
                wing_prefix: r.wing_prefix.map(|p| p.to_ascii_lowercase()),
                rooms,
                room_prefix: r.room_prefix.map(|p| p.to_ascii_lowercase()),
                tier,
            });
        }
        Ok(Self { rules, default })
    }

    /// Read and parse a policy file.
    pub fn load(path: &Path) -> Result<Self> {
        let src = std::fs::read_to_string(path).map_err(|e| {
            PalaceError::InvalidConfig(format!("tier policy {}: {e}", path.display()))
        })?;
        Self::from_toml_str(&src)
    }

    /// The tier for a destination `(wing, room)`. Case-insensitive.
    pub fn tier(&self, wing: &str, room: &str) -> &'static str {
        let wing = wing.to_ascii_lowercase();
        let room = room.to_ascii_lowercase();
        self.rules.iter().find(|r| r.matches(&wing, &room)).map_or(self.default, |r| r.tier)
    }

    /// Stamp `provenance` with the tier this policy assigns to `(wing, room)`.
    ///
    /// Any `tier=` segment already present is removed first. Provenance can come
    /// from a caller (an MCP client, a sync peer), and a caller must not be able
    /// to label a drawer less restrictively than the policy that routed it: the
    /// label would then lie to anything that trusts it later (export, sync, a
    /// serving layer). Idempotent: re-stamping under the same policy is a no-op.
    pub fn tag_provenance(&self, provenance: &str, wing: &str, room: &str) -> String {
        let kept: Vec<&str> = provenance
            .split('|')
            .filter(|seg| !seg.trim_start().to_ascii_lowercase().starts_with(TIER_TAG_PREFIX))
            .collect();
        format!("{}|{TIER_TAG_PREFIX}{}", kept.join("|"), self.tier(wing, room))
    }
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::TempDir;

    /// Cheap Argon2 params. Production reads params from the keyslot record;
    /// tests must not pay 19 MiB × N.
    fn fast_params() -> Argon2Params {
        Argon2Params { m_cost: 8, t_cost: 1, p_cost: 1 }
    }

    fn open_encrypted(path: &Path, key: &str) -> Connection {
        let c = Connection::open(path).expect("open");
        c.pragma_update(None, "key", key).expect("key");
        c
    }

    /// Build an open-tier db and a private-tier db with a known L2 drawer.
    /// Returns (tempdir, open_path, priv_path, dk).
    fn two_tier_fixture() -> (TempDir, PathBuf, PathBuf, [u8; 32]) {
        let td = TempDir::new().unwrap();
        let open_path = td.path().join("palace.db");
        let priv_path = td.path().join("palace_private.db");

        // Open tier: one L1 drawer.
        let oc = open_encrypted(&open_path, "open-tier-key");
        oc.execute_batch(
            "CREATE TABLE drawers (id TEXT PRIMARY KEY, wing TEXT, room TEXT, content TEXT);
             INSERT INTO drawers VALUES ('L1-001','public','notes','open tier content');",
        )
        .unwrap();
        drop(oc);

        // Private tier: one L2 drawer, under its own raw DK.
        let mut dk = [0u8; 32];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut dk);
        let pc = open_encrypted(&priv_path, raw_key_literal(&dk).as_str());
        pc.execute_batch(
            "CREATE TABLE drawers (id TEXT PRIMARY KEY, wing TEXT, room TEXT, content TEXT);
             INSERT INTO drawers VALUES ('L2-SECRET','private','notes','PRIVATE MATERIAL');",
        )
        .unwrap();
        drop(pc);

        (td, open_path, priv_path, dk)
    }

    // -----------------------------------------------------------------------
    // TEST 7 — the one that distinguishes a membrane from a policy.
    // -----------------------------------------------------------------------

    /// **Test 7: a deliberately absent tier check still cannot
    /// read L2.**
    ///
    /// Every other test could pass on a system that merely *filters* L2 rows.
    /// This one cannot. It runs the most permissive query the schema allows —
    /// no tier predicate, no wing predicate, no policy layer of any kind, the
    /// exact query a bug or a malicious refactor would produce — and asserts it
    /// still returns nothing, because the private database was never attached.
    ///
    /// If this passes, the property is structural. If it fails, we built an
    /// envelope and mislabelled it a membrane.
    #[test]
    fn test7_inverted_tier_check_still_cannot_read_private() {
        let (_td, open_path, _priv_path, _dk) = two_tier_fixture();

        // Outer-tier session: open tier key only, membrane CLOSED.
        let conn = open_encrypted(&open_path, "open-tier-key");
        assert!(!is_attached(&conn), "membrane must start closed");

        // 1. The unfiltered query — no tier check whatsoever.
        let found: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM drawers WHERE id = 'L2-SECRET'",
                [],
                |r| r.get(0),
            )
            .expect("query itself must succeed");
        assert_eq!(found, 0, "L2 row must be ABSENT from the open database, not filtered");

        // 2. Reaching for the private schema by name must be an error, not an
        //    empty result — there is no such schema in this connection.
        let cross: std::result::Result<i64, _> = conn.query_row(
            &format!("SELECT COUNT(*) FROM {PRIV_SCHEMA}.drawers"),
            [],
            |r| r.get(0),
        );
        assert!(cross.is_err(), "private schema must not resolve while closed");

        // 3. And the open tier is genuinely serving — this is not a dead
        //    connection producing zeros for everything.
        let l1: i64 = conn
            .query_row("SELECT COUNT(*) FROM drawers WHERE id = 'L1-001'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(l1, 1, "open tier must still work — otherwise the test proves nothing");
    }

    /// Attaching with the right DK makes L2 readable; detaching makes it absent
    /// again. The positive control for Test 7.
    #[test]
    fn membrane_open_then_close_round_trip() {
        let (_td, open_path, priv_path, dk) = two_tier_fixture();
        let conn = open_encrypted(&open_path, "open-tier-key");

        attach_private(&conn, &priv_path, &dk).expect("attach with correct DK");
        assert!(is_attached(&conn));

        let content: String = conn
            .query_row(
                &format!("SELECT content FROM {PRIV_SCHEMA}.drawers WHERE id='L2-SECRET'"),
                [],
                |r| r.get(0),
            )
            .expect("L2 readable while open");
        assert_eq!(content, "PRIVATE MATERIAL");

        detach_private(&conn).expect("detach");
        assert!(!is_attached(&conn), "closed means detached");

        let after: std::result::Result<i64, _> = conn.query_row(
            &format!("SELECT COUNT(*) FROM {PRIV_SCHEMA}.drawers"),
            [],
            |r| r.get(0),
        );
        assert!(after.is_err(), "after close the private schema is gone again");
    }

    /// **Wrong key → no attach.** Not a rejected query afterwards: the
    /// attachment itself must not survive, so the connection is left exactly as
    /// it was.
    #[test]
    fn wrong_key_leaves_nothing_attached() {
        let (_td, open_path, priv_path, _dk) = two_tier_fixture();
        let conn = open_encrypted(&open_path, "open-tier-key");

        let wrong = [0x11u8; 32];
        let r = attach_private(&conn, &priv_path, &wrong);
        assert!(r.is_err(), "wrong DK must fail at the encryption boundary");
        assert!(!is_attached(&conn), "failed open must leave NOTHING attached");
    }

    /// **Test 8: the recovery code opens exactly what the passphrase opens.**
    /// Proves the two keyslots address one DK rather than two divergent copies.
    #[test]
    fn test8_recovery_code_opens_what_passphrase_opens() {
        let (_td, open_path, priv_path, dk) = two_tier_fixture();
        let slots = MembraneKeyslots::seal(&dk, b"correct horse", b"RECOVERY-CODE-128", &fast_params())
            .expect("seal");

        let via_pass = slots.unwrap_with_passphrase(b"correct horse").expect("pass slot");
        let via_rec = slots.unwrap_with_recovery(b"RECOVERY-CODE-128").expect("recovery slot");
        assert_eq!(via_pass.as_slice(), via_rec.as_slice(), "both slots must yield the SAME DK");
        assert_eq!(via_pass.as_slice(), &dk[..]);

        // And it actually opens the database.
        let conn = open_encrypted(&open_path, "open-tier-key");
        attach_private(&conn, &priv_path, &via_rec).expect("recovery code opens the vault");
        let n: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {PRIV_SCHEMA}.drawers"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    /// **Test 9: neither wrapped blob leaks the other, and neither opens
    /// without its opener.** This is what makes it safe to store both blobs
    /// beside the database.
    #[test]
    fn test9_keyslots_are_independent() {
        // Distinctive sentinels. An earlier version of this test used b"pass"
        // and b"REC", which collide with the record's own field names
        // (`salt_pass`, `wrapped_dk_pass`) — the leak assertion matched schema
        // rather than secret and failed for the wrong reason. A leak test whose
        // needle appears in the haystack by construction proves nothing.
        const PASS: &[u8] = b"XYZZY-passphrase-sentinel-9f3a";
        const RECOVERY: &[u8] = b"PLUGH-recovery-sentinel-b71c";

        let (_td, _o, _p, dk) = two_tier_fixture();
        let slots = MembraneKeyslots::seal(&dk, PASS, RECOVERY, &fast_params()).unwrap();

        assert!(slots.unwrap_with_passphrase(b"wrong").is_err(), "wrong passphrase must fail");
        assert!(slots.unwrap_with_recovery(b"wrong").is_err(), "wrong recovery code must fail");

        // Cross-slot: the passphrase must not open the recovery blob. Distinct
        // AAD is what enforces this even if a salt were ever reused.
        let mk = argon2id_derive(PASS, &slots.salt_pass, &slots.params).unwrap();
        assert!(
            aead_unwrap(&mk, &slots.wrapped_dk_recovery, AAD_PASS).is_err(),
            "a slot's key must not unwrap the other slot's blob"
        );
        // ...and symmetrically.
        let mkr = argon2id_derive(RECOVERY, &slots.salt_recovery, &slots.params).unwrap();
        assert!(
            aead_unwrap(&mkr, &slots.wrapped_dk_pass, AAD_RECOVERY).is_err(),
            "the recovery key must not unwrap the passphrase slot's blob"
        );

        // The stored record carries ciphertext and public parameters only.
        // Neither opener, nor the DK itself, is recoverable from it.
        let blob = serde_json::to_vec(&slots).unwrap();
        let hay = String::from_utf8_lossy(&blob);
        let pass_s = String::from_utf8_lossy(PASS);
        let rec_s = String::from_utf8_lossy(RECOVERY);
        assert!(!hay.contains(pass_s.as_ref()), "passphrase must not appear in the record");
        assert!(!hay.contains(rec_s.as_ref()), "recovery code must not appear in the record");
        assert!(
            !hay.contains(&hex::encode(dk)),
            "the Data Key must never appear in the on-disk record"
        );
    }

    /// **Test 5 (reshaped): idle expiry fires without sleeping for an hour.**
    /// The 60-minute default is real, so the clock is injected and advanced.
    #[test]
    fn test5_idle_expiry_via_injected_clock() {
        let clock = TestClock::new(1_000);
        let cfg = MembraneConfig::default(); // 60 min idle, no ceiling
        let s = MembraneSession {
            opened_at_secs: clock.now_secs(),
            last_activity_secs: clock.now_secs(),
            priv_path: PathBuf::from("unused"),
        };

        clock.advance(Duration::from_secs(59 * 60));
        assert!(!s.is_expired(&cfg, clock.now_secs()), "59 min idle: still open");

        clock.advance(Duration::from_secs(2 * 60)); // now 61 min
        assert!(s.is_expired(&cfg, clock.now_secs()), "61 min idle: must auto-lock");
    }

    /// Activity renews the idle clock — a busy session does not lock under the
    /// configured policy. Named so the behaviour is deliberate, not accidental.
    #[test]
    fn busy_session_does_not_idle_out() {
        let cfg = MembraneConfig::default();
        let mut s = MembraneSession {
            opened_at_secs: 0,
            last_activity_secs: 0,
            priv_path: PathBuf::from("unused"),
        };
        // Touch every 30 minutes for 10 hours.
        for i in 1..=20 {
            let now = i * 30 * 60;
            assert!(!s.is_expired(&cfg, now), "busy session must stay open at {now}s");
            s.last_activity_secs = now;
        }
    }

    /// With a ceiling configured, even a busy session eventually re-locks.
    /// Proves `max_session` is wired, so enabling it later is a config change.
    #[test]
    fn max_session_ceiling_locks_a_busy_session() {
        let cfg = MembraneConfig {
            max_session: Some(Duration::from_secs(2 * 60 * 60)),
            ..MembraneConfig::default()
        };
        let s = MembraneSession {
            opened_at_secs: 0,
            last_activity_secs: 2 * 60 * 60, // active right up to the ceiling
            priv_path: PathBuf::from("unused"),
        };
        assert!(s.is_expired(&cfg, 2 * 60 * 60), "ceiling must fire regardless of activity");
    }

    /// The defaults, asserted so a silent change to the policy breaks
    /// a test rather than shipping quietly.
    #[test]
    fn defaults_are_sixty_minutes_idle_no_ceiling() {
        let c = MembraneConfig::default();
        assert_eq!(c.idle_timeout, Some(Duration::from_secs(3600)));
        assert_eq!(c.max_session, None);
        assert!(c.renew_on_activity);
    }

    // -- tier policy --------------------------------------------------------

    const SAMPLE_POLICY: &str = r#"
default = "L3"

[[rule]]
room = "vault"
tier = "L3"

[[rule]]
room = ["diary", "letters"]
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
"#;

    fn sample() -> TierPolicy {
        TierPolicy::from_toml_str(SAMPLE_POLICY).expect("sample policy parses")
    }

    #[test]
    fn first_matching_rule_wins() {
        let p = sample();
        // `vault` is listed first, so it beats every wing rule below it.
        assert_eq!(p.tier("public", "vault"), "L3");
        assert_eq!(p.tier("agent_ada", "vault"), "L3");
        // Room rules listed before wing rules win over them.
        assert_eq!(p.tier("private", "diary"), "L2");
        assert_eq!(p.tier("public_docs", "letters"), "L2");
        // Otherwise the wing rules decide.
        assert_eq!(p.tier("public_docs", "readme"), "L1");
        assert_eq!(p.tier("agent_ada", "work-log"), "L2");
        assert_eq!(p.tier("private", "health"), "L3");
    }

    #[test]
    fn order_is_the_only_precedence() {
        // The same two rules in the opposite order: the wing rule now wins.
        let p = TierPolicy::from_toml_str(
            "[[rule]]\nwing = \"public\"\ntier = \"L1\"\n\n[[rule]]\nroom = \"vault\"\ntier = \"L3\"\n",
        )
        .unwrap();
        assert_eq!(p.tier("public", "vault"), "L1");
    }

    #[test]
    fn unmatched_input_fails_safe_to_the_default() {
        let p = sample();
        assert_eq!(p.tier("something_unmapped", "who-knows"), "L3");
        assert_eq!(p.tier("", ""), "L3");
        // No policy at all behaves like an empty one: everything sealed.
        assert_eq!(TierPolicy::fail_safe().tier("public", "readme"), "L3");
        assert_eq!(TierPolicy::default(), TierPolicy::fail_safe());
        assert_eq!(TierPolicy::from_toml_str("").unwrap(), TierPolicy::fail_safe());
        // An explicit default is honoured.
        let p2 = TierPolicy::from_toml_str("default = \"L2\"\n").unwrap();
        assert_eq!(p2.tier("anything", "at-all"), "L2");
    }

    #[test]
    fn matching_is_case_insensitive_so_case_cannot_downgrade() {
        let p = sample();
        assert_eq!(p.tier("PRIVATE", "notes"), "L3");
        assert_eq!(p.tier("Private", "notes"), "L3");
        assert_eq!(p.tier("agent_ada", "VAULT"), "L3");
        assert_eq!(p.tier("Public_Docs", "readme"), "L1");
        // Patterns and tier names in the file are case-insensitive too.
        let q = TierPolicy::from_toml_str("default = \"L1\"\n\n[[rule]]\nwing = \"Private\"\ntier = \"l3\"\n").unwrap();
        assert_eq!(q.tier("private", "x"), "L3");
        assert_eq!(q.tier("PRIVATE", "x"), "L3");
    }

    #[test]
    fn a_rule_with_several_matchers_needs_all_of_them() {
        let p = TierPolicy::from_toml_str(
            "default = \"L1\"\n\n[[rule]]\nwing = \"agent_ada\"\nroom_prefix = \"secret\"\ntier = \"L3\"\n",
        )
        .unwrap();
        assert_eq!(p.tier("agent_ada", "secret-plans"), "L3");
        assert_eq!(p.tier("agent_ada", "plans"), "L1");
        assert_eq!(p.tier("agent_ben", "secret-plans"), "L1");
    }

    #[test]
    fn strict_loading_rejects_anything_that_would_misroute_silently() {
        let cases = [
            ("[[rule]]\nwing_prefx = \"public\"\ntier = \"L1\"\n", "misspelled key must not be ignored"),
            ("bogus = 1\n", "unknown top-level key"),
            ("[[rule]]\ntier = \"L1\"\n", "rule with no matcher would match everything"),
            ("[[rule]]\nwing_prefix = \"\"\ntier = \"L1\"\n", "empty prefix matches everything"),
            ("[[rule]]\nroom = []\ntier = \"L1\"\n", "empty list matches nothing"),
            ("[[rule]]\nroom = \"x\"\ntier = \"L4\"\n", "unknown tier"),
            ("default = \"open\"\n", "unknown default tier"),
            ("[[rule]]\nroom = \"x\"\n", "missing tier"),
            ("not toml at all [", "not TOML"),
        ];
        for (src, why) in cases {
            match TierPolicy::from_toml_str(src) {
                Err(PalaceError::InvalidConfig(_)) => {}
                other => panic!("{why}: expected InvalidConfig, got {other:?}"),
            }
        }
    }

    #[test]
    fn load_reads_a_policy_file_and_names_a_missing_one() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("tiers.toml");
        std::fs::write(&path, SAMPLE_POLICY).unwrap();
        assert_eq!(TierPolicy::load(&path).unwrap(), sample());
        match TierPolicy::load(&td.path().join("missing.toml")) {
            Err(PalaceError::InvalidConfig(msg)) => assert!(msg.contains("missing.toml"), "{msg}"),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn the_shipped_example_policy_loads_and_routes_as_its_comments_say() {
        let p = TierPolicy::from_toml_str(include_str!("../../../examples/tier-policy.toml"))
            .expect("examples/tier-policy.toml must stay valid");
        assert_eq!(p.tier("public_notes", "vault"), "L3", "a vault room is sealed even in a public wing");
        assert_eq!(p.tier("wing_ada", "personal-diary"), "L2", "diary_write's room is covered");
        assert_eq!(p.tier("wing_ada", "diary"), "L2");
        assert_eq!(p.tier("anything", "Personal"), "L2", "matching ignores case");
        assert_eq!(p.tier("public_notes", "ideas"), "L1");
        assert_eq!(p.tier("work", "ideas"), "L3", "unmatched falls to the default");
    }

    #[test]
    fn only_l1_is_ever_network_served() {
        assert!(is_network_servable("L1"));
        assert!(!is_network_servable("L2"));
        assert!(!is_network_servable("L3"));
        assert!(!is_network_servable("nonsense"));
        let p = sample();
        assert!(is_network_servable(p.tier("public", "readme")));
        assert!(!is_network_servable(p.tier("agent_ada", "diary")));
        assert!(!is_network_servable(p.tier("private", "health")));
    }

    #[test]
    fn an_untiered_palace_labels_drawers_single_and_never_l3() {
        let p = TierPolicy::untiered();
        assert_eq!(p.tag_provenance("agent", "work", "notes"), "agent|tier=single");
        assert_eq!(p.tag_provenance("x|tier=L1", "work", "notes"), "x|tier=single", "a caller label never survives");
        assert_eq!(tier_rank(UNTIERED), u8::MAX, "single is not a tier: it ranks as most restrictive");
        assert!(!is_network_servable(UNTIERED), "single is never network-served");
    }

    #[test]
    fn tier_rank_orders_restrictiveness_and_unknown_ranks_highest() {
        assert!(tier_rank("L1") < tier_rank("L2"));
        assert!(tier_rank("L2") < tier_rank("L3"));
        assert!(tier_rank("L3") < tier_rank("something-new"));
    }

    #[test]
    fn tag_provenance_is_idempotent_and_absence_is_none() {
        let p = sample();
        assert_eq!(tier_from_provenance("src=x|key=y"), None, "absent must be None, never a default");
        let once = p.tag_provenance("src=x", "private", "health");
        assert_eq!(tier_from_provenance(&once), Some("L3"));
        let twice = p.tag_provenance(&once, "private", "health");
        assert_eq!(once, twice, "re-tagging must be a no-op");
        // A caller-supplied tag never survives: the policy's tier always wins,
        // whether the caller tried to loosen it or tighten it.
        assert_eq!(p.tag_provenance("src=x|tier=L1", "private", "health"), "src=x|tier=L3");
        assert_eq!(p.tag_provenance("src=x|TIER=L1|tier=L2", "public", "readme"), "src=x|tier=L1");
        assert_eq!(p.tag_provenance("tier=L1", "private", "health"), "|tier=L3");
        assert_eq!(p.tag_provenance("agent-write", "public", "readme"), "agent-write|tier=L1");
    }
}