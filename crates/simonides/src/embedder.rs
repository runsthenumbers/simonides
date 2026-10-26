//! Thin fastembed wrapper for simonides M3b vector retrieval.
//!
//! Vectors are never written to disk here â€” the caller stores the raw f32
//! bytes as BLOBs inside the sqlcipher DB.  This module is pure inference.

use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::PalaceError;

// hf-hub 0.5.x uses a per-file lock file that protects cross-PROCESS concurrent
// downloads but NOT concurrent threads within the same process.  Multiple tests
// (or multiple PalaceStore instances) starting up in parallel all call try_new()
// simultaneously and race on the incomplete ONNX file.  This process-level mutex
// serialises initialisation so only one thread downloads the model; after the
// first success every subsequent try_new() hits the cache with no contention.
static ONNX_INIT: Mutex<()> = Mutex::new(());

type Result<T> = std::result::Result<T, PalaceError>;

/// HuggingFace model ID recorded in meta.embedding_model_version.
/// Changing this string must trigger a full reindex (all rows reindex_pending=1).
pub(crate) const MODEL_ID: &str = "BAAI/bge-small-en-v1.5";

/// Output dimensionality of BGE-small-en-v1.5.
#[allow(dead_code)] // used only in tests; named constant for self-documenting assertions
pub(crate) const EMBEDDING_DIM: usize = 384;

/// Wraps a `TextEmbedding` session.  One instance per `PalaceStore` (lazy
/// init on first use, held for the lifetime of the store).
pub(crate) struct Embedder {
    inner: TextEmbedding,
}

impl Embedder {
    /// Initialise the ONNX session for BGE-small-en-v1.5.
    /// Downloads model weights on first run; cached to disk by fastembed.
    pub(crate) fn try_new() -> Result<Self> {
        let _guard = ONNX_INIT.lock().unwrap();
        let mut opts = TextInitOptions::new(EmbeddingModel::BGESmallENV15);
        // Pin an ABSOLUTE cache dir. fastembed's default (`.fastembed_cache`) is
        // CWD-relative, so the server and a build resolve DIFFERENT caches and the
        // model silently splits (seen in practice: a 4 MB `.part` on one side, the
        // real 127 MB model on the other). Honour FASTEMBED_CACHE_DIR if set, else
        // an absolute <home>/.fastembed_cache -- never CWD-relative.
        if let Some(dir) = resolved_cache_dir() {
            opts = opts.with_cache_dir(dir);
        }
        let inner = TextEmbedding::try_new(opts)
            .map_err(|e| PalaceError::Storage(format!("embedder init (BGE-small-en-v1.5): {e}")))?;
        Ok(Self { inner })
    }

    /// Embed a batch of texts.  Returns one 384-dim f32 vector per input text,
    /// in the same order as `texts`.
    pub(crate) fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.inner
            .embed(texts, None)
            .map_err(|e| PalaceError::Storage(format!("embed: {e}")))
    }
}

/// Resolve an ABSOLUTE fastembed cache dir, independent of the process CWD.
pub(crate) fn resolved_cache_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("FASTEMBED_CACHE_DIR") {
        if !d.is_empty() {
            return Some(std::path::PathBuf::from(d));
        }
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    Some(std::path::PathBuf::from(home).join(".fastembed_cache"))
}

// -- Out-of-process embed worker -------------------------------------------------
// The fastembed/ONNX runtime retains process-global memory across inferences that
// dropping the session does NOT reclaim (~0.39 MB/chunk, unbounded). The only
// guaranteed-reclaim path is process EXIT. So the vector rebuild embeds each batch
// in a child process that exits afterward: the OS reclaims every byte, peak RSS is
// bounded by one batch, and it is the right shape for a phone. Framing is
// length-prefixed binary so arbitrary chunk text passes through untouched.

/// Parent -> worker stdin. Frame: u32 LE count, then per text u32 LE len + bytes.
pub(crate) fn write_batch<W: std::io::Write>(w: &mut W, texts: &[&str]) -> std::io::Result<()> {
    w.write_all(&(texts.len() as u32).to_le_bytes())?;
    for t in texts {
        let b = t.as_bytes();
        w.write_all(&(b.len() as u32).to_le_bytes())?;
        w.write_all(b)?;
    }
    w.flush()
}

/// Worker side: read the batch written by [`write_batch`].
pub(crate) fn read_batch<R: std::io::Read>(r: &mut R) -> std::io::Result<Vec<String>> {
    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    let n = u32::from_le_bytes(u32buf) as usize;
    let mut texts = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut u32buf)?;
        let len = u32::from_le_bytes(u32buf) as usize;
        let mut b = vec![0u8; len];
        r.read_exact(&mut b)?;
        texts.push(String::from_utf8_lossy(&b).into_owned());
    }
    Ok(texts)
}

/// Worker -> parent stdout. Frame: u32 LE count, then per vec u32 LE dim + dim*f32.
pub(crate) fn write_vecs<W: std::io::Write>(w: &mut W, vecs: &[Vec<f32>]) -> std::io::Result<()> {
    w.write_all(&(vecs.len() as u32).to_le_bytes())?;
    for v in vecs {
        w.write_all(&(v.len() as u32).to_le_bytes())?;
        for f in v {
            w.write_all(&f.to_le_bytes())?;
        }
    }
    w.flush()
}

/// Parent side: read vectors written by [`write_vecs`].
pub(crate) fn read_vecs<R: std::io::Read>(r: &mut R) -> std::io::Result<Vec<Vec<f32>>> {
    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    let n = u32::from_le_bytes(u32buf) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut u32buf)?;
        let dim = u32::from_le_bytes(u32buf) as usize;
        let mut v = Vec::with_capacity(dim);
        let mut fbuf = [0u8; 4];
        for _ in 0..dim {
            r.read_exact(&mut fbuf)?;
            v.push(f32::from_le_bytes(fbuf));
        }
        out.push(v);
    }
    Ok(out)
}

/// Worker entry point (called by the binary on `--embed-batch`): read one batch
/// from stdin, embed it, write vectors to stdout. The caller then EXITS, which is
/// what frees the retained native memory.
pub fn run_embed_worker() -> Result<()> {
    let texts = {
        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        read_batch(&mut lock)
            .map_err(|e| PalaceError::Storage(format!("embed-worker read: {e}")))?
    };
    let mut emb = Embedder::try_new()?;
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let vecs = emb.embed(&refs)?;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    write_vecs(&mut lock, &vecs)
        .map_err(|e| PalaceError::Storage(format!("embed-worker write: {e}")))?;
    Ok(())
}

/// Parent side: embed `texts` in a fresh child process (`<exe> --embed-batch`)
/// that exits afterward so the OS reclaims all retained memory. A writer thread
/// feeds stdin while this thread reads stdout, to avoid pipe deadlock.
pub(crate) fn embed_via_subprocess(exe: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let mut child = std::process::Command::new(exe)
        .arg("--embed-batch")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .map_err(|e| PalaceError::Storage(format!("embed-worker spawn ({exe}): {e}")))?;

    let mut stdin = child.stdin.take().expect("child stdin piped");
    let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        write_batch(&mut stdin, &refs)
        // stdin dropped here -> EOF for the child.
    });

    let mut stdout = child.stdout.take().expect("child stdout piped");
    let vecs = read_vecs(&mut stdout)
        .map_err(|e| PalaceError::Storage(format!("embed-worker read vectors: {e}")))?;

    let _ = writer.join();
    let status = child
        .wait()
        .map_err(|e| PalaceError::Storage(format!("embed-worker wait: {e}")))?;
    if !status.success() {
        return Err(PalaceError::Storage(format!(
            "embed-worker exited with failure: {status}"
        )));
    }
    if vecs.len() != texts.len() {
        return Err(PalaceError::Storage(format!(
            "embed-worker returned {} vectors for {} texts",
            vecs.len(),
            texts.len()
        )));
    }
    Ok(vecs)
}


// â”€â”€ Serialisation helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Pack an f32 slice into little-endian bytes for BLOB storage.
pub(crate) fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Unpack little-endian bytes (stored BLOB) back into an f32 vector.
pub(crate) fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// â”€â”€ Similarity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Cosine similarity in [âˆ’1, 1].  Returns 0.0 for zero-norm vectors.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    #[test]
    fn batch_roundtrip() {
        let texts = ["hello world", "", "unicode: cafe nihongo", "a"];
        let refs: Vec<&str> = texts.to_vec();
        let mut buf = Vec::new();
        write_batch(&mut buf, &refs).unwrap();
        let got = read_batch(&mut std::io::Cursor::new(buf)).unwrap();
        let want: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn vecs_roundtrip() {
        let vecs = vec![vec![1.0f32, -2.5, 3.0], vec![], vec![0.25f32; 384]];
        let mut buf = Vec::new();
        write_vecs(&mut buf, &vecs).unwrap();
        let got = read_vecs(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(got, vecs);
    }
}
