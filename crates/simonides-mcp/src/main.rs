//! simonides: stdio JSON-RPC MCP server over an encrypted Simonides store.
//!
//! Keys and paths come from the environment or from files on this machine,
//! **never from a tool argument**: a key passed as an argument would travel
//! through the client, its logs, and anything relaying it.
//!
//! # Two shapes, chosen by what is configured
//!
//! **Single store (default).** One encrypted database and blob folder in the
//! Simonides home folder. The key is `SIMONIDES_KEY` if set, else the key file
//! `<home>/key`, generated on first run and readable only by the current user.
//!
//! **Three tiers.** Turned on by setting `SIMONIDES_TIER_POLICY` to a policy
//! file. L1 is the store above; L2 is a second store with its own key
//! (`SIMONIDES_L2_KEY` or `<home>/key-l2`); L3 is a third store that always
//! starts locked.
//!
//! **There is deliberately no `SIMONIDES_L3_KEY`.** An unattended restart must
//! never come up with the vault open, and the environment is exactly what an
//! unattended restart replays. Setting it is a fatal error.

use serde_json::{json, Value};
use std::io::{BufRead, Write};
use simonides_mcp::config;
use simonides_mcp::McpServer;

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("simonides: {msg}");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Out-of-process embed worker: reads one batch of texts from stdin, embeds
    // it, writes vectors to stdout, and exits, which is what frees the ONNX
    // runtime's retained memory. No store or key needed.
    if args.iter().any(|a| a == "--embed-batch") {
        if let Err(e) = simonides::run_embed_worker() {
            eprintln!("embed-worker: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{}", config::USAGE);
        return;
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("simonides {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let setup = args.iter().any(|a| a == "--setup");
    if let Some(unknown) = args.iter().find(|a| !matches!(a.as_str(), "--setup")) {
        fail(format!("unknown argument {unknown:?}; see simonides --help"));
    }

    let home = config::home_dir().unwrap_or_else(|e| fail(e));
    std::fs::create_dir_all(&home).unwrap_or_else(|e| fail(format!("create {}: {e}", home.display())));

    // Model cache and embed worker default to this install. Set before any
    // thread starts; child processes inherit them.
    if config::env_nonempty("FASTEMBED_CACHE_DIR").is_none() {
        std::env::set_var("FASTEMBED_CACHE_DIR", home.join("models"));
    }
    if config::env_nonempty("SIMONIDES_EMBED_WORKER").is_none() {
        if let Ok(exe) = std::env::current_exe() {
            std::env::set_var("SIMONIDES_EMBED_WORKER", exe);
        }
    }

    let db = config::path_or("SIMONIDES_DB", home.join("palace.db"));
    let vault = config::path_or("SIMONIDES_VAULT", home.join("vault"));
    let key = config::resolve_key("SIMONIDES_KEY", &home.join("key")).unwrap_or_else(|e| fail(e));

    // Tiered mode is selected by configuration, not a flag, so a deployment
    // cannot be half-configured: either a policy is given and all three tiers
    // are used, or none is and there is one store.
    let server = match config::env_nonempty("SIMONIDES_TIER_POLICY") {
        Some(policy_path) => {
            if std::env::var_os("SIMONIDES_L3_KEY").is_some() {
                fail(
                    "SIMONIDES_L3_KEY is set. The L3 key must never live in the environment: \
                     that would reopen the vault on every unattended restart. Remove it.",
                );
            }
            let policy = simonides::membrane::TierPolicy::load(std::path::Path::new(&policy_path))
                .unwrap_or_else(|e| fail(e));
            let l2_db = config::path_or("SIMONIDES_L2_DB", home.join("palace-l2.db"));
            let l2_vault = config::path_or("SIMONIDES_L2_VAULT", home.join("vault-l2"));
            let l2_key = config::resolve_key("SIMONIDES_L2_KEY", &home.join("key-l2"))
                .unwrap_or_else(|e| fail(e));
            let l3_db = config::path_or("SIMONIDES_L3_DB", home.join("palace-l3.db"));
            let l3_vault = config::path_or("SIMONIDES_L3_VAULT", home.join("vault-l3"));
            eprintln!("simonides: three-tier mode, home {} (L3 locked)", home.display());
            McpServer::open_tiered(
                &db, &vault, &key,
                &l2_db, &l2_vault, &l2_key,
                &l3_db, &l3_vault, policy,
            )
            .unwrap_or_else(|e| fail(format!("could not open the tiered store: {e}")))
        }
        None => {
            eprintln!("simonides: single store {}", db.display());
            McpServer::open(&db, &vault, &key)
                .unwrap_or_else(|e| fail(format!("could not open {}: {e}", db.display())))
        }
    };

    server.set_auto_index(true);

    if !setup {
        // Finish any indexing a previous run left undone (a crash, a failed
        // model download). A no-op when nothing is pending.
        let _ = server.handle_tool("rebuild", &json!({}));
    }

    if setup {
        let exe = std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "simonides".to_string());
        let block = json!({"mcpServers": {"simonides": {"command": exe}}});
        println!("Simonides is ready. Home folder: {}", home.display());
        println!();
        println!("Add this to your MCP client's configuration (for Claude Desktop, claude_desktop_config.json):");
        println!();
        println!("{}", serde_json::to_string_pretty(&block).unwrap());
        println!();
        println!("For Claude Code:  claude mcp add simonides -- \"{exe}\"");
        println!();
        println!("Back up the key file in the home folder. Without it the store cannot be opened.");
        drop(server);
        return;
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let resp = json!({"jsonrpc": "2.0", "id": Value::Null, "error": {"code": -32700, "message": format!("parse error: {e}")}});
                let _ = writeln!(out, "{resp}");
                let _ = out.flush();
                continue;
            }
        };
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
        // JSON-RPC notifications carry no id and MUST NOT get a response.
        if method.starts_with("notifications/") {
            continue;
        }
        let resp = match method {
            "initialize" => {
                // Agree to the client's requested protocol version; advertise the tools capability.
                // Answer with the client's version if we support it, else our latest.
                let asked = params.get("protocolVersion").and_then(|v| v.as_str()).unwrap_or("");
                let pv = simonides_mcp::protocol::negotiate_version(asked);
                json!({"jsonrpc": "2.0", "id": id, "result": {
                    "protocolVersion": pv,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "simonides", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": simonides_mcp::protocol::INSTRUCTIONS
                }})
            }
            "ping" => json!({"jsonrpc": "2.0", "id": id, "result": {}}),
            "tools/list" => json!({"jsonrpc": "2.0", "id": id, "result": {"tools": McpServer::tool_schemas()}}),
            "tools/call" => {
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
                let result = simonides_mcp::protocol::call_result(server.handle_tool(name, &args));
                json!({"jsonrpc": "2.0", "id": id, "result": result})
            }
            other => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("method not found: {other}")}}),
        };
        let _ = writeln!(out, "{}", serde_json::to_string(&resp).unwrap());
        let _ = out.flush();
    }
}
