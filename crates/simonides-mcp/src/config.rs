//! Startup configuration: where Simonides keeps its files and how it gets its keys.
//!
//! Zero configuration is the default. On first run Simonides creates its home
//! directory, generates a random 256-bit database key into a file that only the
//! current user can read, and opens an encrypted store there. Every setting can
//! be overridden with an environment variable; see [`USAGE`].

use std::io::Write;
use std::path::{Path, PathBuf};

use zeroize::Zeroizing;

/// Printed by `simonides --help`.
pub const USAGE: &str = "\
simonides - an encrypted, local-first memory palace for AI agents (MCP over stdio)

USAGE:
    simonides            run the MCP server on stdin/stdout (what your agent client launches)
    simonides --setup    create the store and key if needed, and print a client config block
    simonides --help     show this help
    simonides --version  show the version

With no configuration, everything lives in one folder:
    Windows  %APPDATA%\\simonides
    macOS    ~/Library/Application Support/simonides
    Linux    ~/.local/share/simonides

ENVIRONMENT (all optional):
    SIMONIDES_HOME         folder for the store, key and model cache
    SIMONIDES_KEY          database key; if unset, read from (or created in) <home>/key
    SIMONIDES_DB           database file           (default <home>/palace.db)
    SIMONIDES_VAULT        blob folder             (default <home>/vault)
    SIMONIDES_TIER_POLICY  path to a tier policy file; setting it turns on the three-tier
                           membrane. Then also, optionally:
        SIMONIDES_L2_KEY / SIMONIDES_L2_DB / SIMONIDES_L2_VAULT
        SIMONIDES_L3_DB / SIMONIDES_L3_VAULT
                           (L3 never takes a key from the environment)
    SIMONIDES_EMBED_WORKER program used to embed in a child process (default: this binary)
    SIMONIDES_EMBED_BATCH  chunks embedded per child process (default 2000)
    FASTEMBED_CACHE_DIR    embedding model cache   (default <home>/models)

Back up your key file. Without it the store cannot be opened, by you or anyone.
";

/// An environment variable, treating unset and blank the same.
pub fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Simonides' home folder: `SIMONIDES_HOME`, else the platform data directory.
pub fn home_dir() -> Result<PathBuf, String> {
    if let Some(h) = env_nonempty("SIMONIDES_HOME") {
        return Ok(PathBuf::from(h));
    }
    dirs::data_dir()
        .map(|d| d.join("simonides"))
        .ok_or_else(|| "could not determine a data directory; set SIMONIDES_HOME".to_string())
}

/// A path from `env_name` if set, else `default`.
pub fn path_or(env_name: &str, default: PathBuf) -> PathBuf {
    env_nonempty(env_name).map(PathBuf::from).unwrap_or(default)
}

/// The key for a store: from `env_name` if set, else from `file` (created on first use).
pub fn resolve_key(env_name: &str, file: &Path) -> Result<Zeroizing<String>, String> {
    if let Some(k) = env_nonempty(env_name) {
        return Ok(Zeroizing::new(k));
    }
    load_or_create_key(file)
}

/// Read a key file, or create one holding a fresh random 256-bit key.
///
/// An existing file is never overwritten. A file that is not a Simonides key is
/// refused rather than used as a passphrase, because "the key changed" is a
/// mistake you only want to make loudly.
pub fn load_or_create_key(path: &Path) -> Result<Zeroizing<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(s) => {
            let key = Zeroizing::new(s.trim().to_string());
            if !is_raw_key_literal(&key) {
                return Err(format!(
                    "{} is not a Simonides key file (expected x'<64 hex digits>'); refusing to use it",
                    path.display()
                ));
            }
            if let Some(w) = permission_warning(path) {
                eprintln!("simonides: warning: {w}");
            }
            Ok(key)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => create_key_file(path),
        Err(e) => Err(format!("read {}: {e}", path.display())),
    }
}

fn is_raw_key_literal(s: &str) -> bool {
    s.len() == 67
        && s.starts_with("x'")
        && s.ends_with('\'')
        && s[2..66].bytes().all(|b| b.is_ascii_hexdigit())
}

fn create_key_file(path: &Path) -> Result<Zeroizing<String>, String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let mut dk = Zeroizing::new([0u8; 32]);
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, dk.as_mut());
    let key = simonides::membrane::raw_key_literal(&dk);

    // create_new: if two processes race, exactly one creates the key and the other
    // fails here instead of silently replacing it.
    let f = open_new_private(path)?;
    drop(f);
    // Restrict the file while it is still EMPTY, then write the key. If the
    // lock-down fails, nothing secret was ever on disk.
    if let Err(e) = lock_down(path) {
        let _ = std::fs::remove_file(path);
        return Err(format!(
            "could not restrict {} to the current user ({e}); removed it. \
             Set SIMONIDES_KEY instead, or fix the folder's permissions and retry.",
            path.display()
        ));
    }
    let written = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|mut f| {
            f.write_all(key.as_bytes())?;
            f.write_all(b"\n")?;
            f.sync_all()
        });
    if let Err(e) = written {
        let _ = std::fs::remove_file(path);
        return Err(format!("write {}: {e}", path.display()));
    }
    eprintln!(
        "simonides: created a new database key at {}. Back it up: without it the store cannot be opened.",
        path.display()
    );
    Ok(key)
}

#[cfg(unix)]
fn open_new_private(path: &Path) -> Result<std::fs::File, String> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))
}

#[cfg(not(unix))]
fn open_new_private(path: &Path) -> Result<std::fs::File, String> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))
}

/// The current user as icacls should name them: by SID where `whoami` can
/// report it (works for local, domain and Entra ID accounts alike), else by
/// `DOMAIN\user`.
#[cfg(windows)]
fn current_principal() -> Result<String, String> {
    if let Ok(out) = std::process::Command::new("whoami").args(["/user", "/fo", "csv", "/nh"]).output() {
        let text = String::from_utf8_lossy(&out.stdout);
        if let Some(sid) = text.split(',').map(|f| f.trim().trim_matches('"')).find(|f| f.starts_with("S-1-")) {
            return Ok(format!("*{sid}"));
        }
    }
    match (std::env::var("USERDOMAIN"), std::env::var("USERNAME")) {
        (Ok(d), Ok(u)) if !d.is_empty() && !u.is_empty() => Ok(format!("{d}\\{u}")),
        (_, Ok(u)) if !u.is_empty() => Ok(u),
        _ => Err("cannot tell who the current user is".to_string()),
    }
}

/// Windows: drop inherited permissions and grant the current user sole access.
#[cfg(windows)]
fn lock_down(path: &Path) -> Result<(), String> {
    let user = current_principal()?;
    let out = std::process::Command::new("icacls")
        .arg(path)
        .arg("/inheritance:r")
        .arg("/grant:r")
        .arg(format!("{user}:(F)"))
        .output()
        .map_err(|e| format!("run icacls: {e}"))?;
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stdout).trim().to_string() + String::from_utf8_lossy(&out.stderr).trim();
        return Err(format!("icacls failed: {msg}"));
    }
    Ok(())
}

/// Unix: the file was created with mode 0600, so there is nothing more to do.
#[cfg(not(windows))]
fn lock_down(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn permission_warning(path: &Path) -> Option<String> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path).ok()?.permissions().mode();
    (mode & 0o077 != 0).then(|| {
        format!("{} can be read by other users (mode {:o}); run: chmod 600 {}", path.display(), mode & 0o777, path.display())
    })
}

/// Windows: warn if a broad group can read an existing key file.
#[cfg(windows)]
fn permission_warning(path: &Path) -> Option<String> {
    let out = std::process::Command::new("icacls").arg(path).output().ok()?;
    let acl = String::from_utf8_lossy(&out.stdout).to_string();
    let broad = ["Everyone", "BUILTIN\\Users", "Authenticated Users"]
        .into_iter()
        .find(|b| acl.contains(b))?;
    Some(format!(
        "{} is readable beyond your account ({broad} in its ACL). Restrict it with: \
         icacls \"{}\" /inheritance:r /grant:r \"%USERNAME%:(F)\"",
        path.display(),
        path.display()
    ))
}

#[cfg(not(any(unix, windows)))]
fn permission_warning(_path: &Path) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_call_creates_a_valid_key_and_later_calls_return_the_same_one() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("nested").join("key");
        let k1 = load_or_create_key(&p).expect("create");
        assert!(is_raw_key_literal(&k1), "generated key has the raw-key form: {}", k1.len());
        let k2 = load_or_create_key(&p).expect("reload");
        assert_eq!(*k1, *k2, "an existing key file is read, never regenerated");
        let k3 = load_or_create_key(&d.path().join("other")).expect("second store");
        assert_ne!(*k1, *k3, "every key file gets its own random key");
    }

    #[test]
    fn a_file_that_is_not_a_key_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("key");
        for bad in ["hunter2", "x'1234'", "x'zz00000000000000000000000000000000000000000000000000000000000000'"] {
            std::fs::write(&p, bad).unwrap();
            assert!(load_or_create_key(&p).is_err(), "must refuse {bad:?}");
        }
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "x'zz00000000000000000000000000000000000000000000000000000000000000'", "a refused file is left untouched");
    }

    #[cfg(unix)]
    #[test]
    fn the_key_file_is_private_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("key");
        load_or_create_key(&p).unwrap();
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode was {mode:o}");
    }

    #[cfg(windows)]
    #[test]
    fn an_existing_key_readable_by_everyone_is_flagged_on_windows() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("key");
        load_or_create_key(&p).unwrap();
        assert!(permission_warning(&p).is_none(), "a key we created must not be flagged");
        let ok = std::process::Command::new("icacls").arg(&p).args(["/grant", "*S-1-1-0:(R)"]).output().unwrap();
        assert!(ok.status.success(), "test setup: grant Everyone read");
        assert!(permission_warning(&p).is_some(), "Everyone in the ACL must be flagged");
    }

    #[cfg(windows)]
    #[test]
    fn the_key_file_is_private_on_windows() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("key");
        load_or_create_key(&p).unwrap();
        let out = std::process::Command::new("icacls").arg(&p).output().expect("icacls");
        let acl = String::from_utf8_lossy(&out.stdout).to_string();
        let user = std::env::var("USERNAME").unwrap();
        assert!(acl.to_lowercase().contains(&user.to_lowercase()), "owner must be granted: {acl}");
        for broad in ["Everyone", "BUILTIN\\Users", "Authenticated Users", "SYSTEM", "Administrators", "(I)"] {
            assert!(!acl.contains(broad), "{broad} must not appear in the key file's ACL: {acl}");
        }
    }
}